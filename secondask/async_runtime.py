"""Async batch runtime.

The synchronous ``Runtime`` is a discrete event simulation over a virtual clock:
three weeks of recovery run in a few seconds because nothing actually blocks.
Making that loop async would add complexity and buy nothing, and it would put the
reproducibility guarantee at risk for no gain. So it stays as it is, and remains
the path every reported number comes from.

This module is for the case where the I/O is real: a production worker draining
a queue of due items, where each one needs a payment link created and a message
delivered, and where doing those one at a time means a batch of five thousand
takes as long as five thousand round trips.

The design constraint that shaped it:

    **Concurrency must not change the answer.**

So the loop is structured as *plan sequentially, execute concurrently, apply
sequentially*:

1. Walk the due items in a deterministic order and ask the agent for one action
   each. No I/O, no concurrency, so the decisions are identical to the sync path.
2. Gate every proposed action through the same ``PolicyEngine``. Still no I/O.
3. Perform the surviving actions concurrently, bounded by a semaphore.
4. Apply the results **in the original planning order**, not in completion
   order.

Step 4 is the whole trick. Applying results as they land would make state
transitions depend on network timing, and two runs of the same batch would
produce different ledgers. Sorting the results back into planning order costs
one list index and preserves the property that makes this system auditable.

What is genuinely gained is wall-clock time on real I/O. What is deliberately
not gained is throughput on the simulator, where there is no I/O to overlap.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from .clock import iso
from .execute.async_client import DEFAULT_CONCURRENCY, AsyncChannel, AsyncRazorpayClient, gather_bounded
from .execute.executor import Executor
from .ledger import AsyncLedger, BaseLedger
from .policy.engine import PolicyContext, PolicyEngine, ProposedAction
from .world.entities import ActionKind, ItemState, RiskItem


@dataclass
class AsyncBatchResult:
    planned: int = 0
    executed: int = 0
    denied: int = 0
    failed: int = 0
    recovered_paise: int = 0
    cost_paise: int = 0
    denials: dict[str, int] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "executed": self.executed,
            "denied": self.denied,
            "failed": self.failed,
            "recovered_paise": self.recovered_paise,
            "cost_paise": self.cost_paise,
            "denials": dict(sorted(self.denials.items())),
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class _Planned:
    """One action that survived the gate, waiting to be performed."""

    order: int
    item: RiskItem
    customer: Any
    action: ProposedAction


class AsyncRuntime:
    """Drains a batch of due items with concurrent I/O and ordered application."""

    def __init__(
        self,
        world: Any,
        agent: Any,
        policy: PolicyEngine,
        client: AsyncRazorpayClient,
        llm: Any,
        *,
        ledger: Optional[BaseLedger] = None,
        channel: Optional[AsyncChannel] = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        merchant_name: str = "Acme Retail",
    ) -> None:
        self.world = world
        self.agent = agent
        self.policy = policy
        self.client = client
        self.llm = llm
        self.concurrency = concurrency
        self.ledger: BaseLedger = ledger or AsyncLedger(run_id=f"{agent.name}@{world.seed}")
        self.channel = channel or AsyncChannel(concurrency=concurrency)
        # The synchronous executor is reused verbatim. It holds the ordering rule
        # that a payment link must exist before its message is sent, and that
        # rule is not one to reimplement in a second place.
        self.executor = Executor(world, client.client, llm, merchant_name=merchant_name)
        self._used_keys: set[str] = set()
        self._contacts: dict[str, list[datetime]] = {}
        self.escalation_daily_cap = max(3, len(getattr(world, "items", [])) // 150)

    # -- context -----------------------------------------------------------

    def _context(self, item: RiskItem, customer: Any, now: datetime) -> PolicyContext:
        history = self._contacts.get(customer.customer_id, [])
        day_ago = now - timedelta(hours=24)
        week_ago = now - timedelta(days=7)
        return PolicyContext(
            item=item,
            customer=customer,
            now=now,
            contacts_24h=sum(1 for t in history if t > day_ago),
            contacts_7d=sum(1 for t in history if t > week_ago),
            last_contact_at=history[-1] if history else None,
            used_idempotency_keys=frozenset(self._used_keys),
            consent_promotional=customer.tenure_days > 30,
            dnd_registered=customer.prior_ignores >= 3,
            escalation_daily_cap=self.escalation_daily_cap,
            merchant_name=self.executor.merchant_name,
        )

    # -- the loop ----------------------------------------------------------

    async def run_batch(self, items: list[RiskItem], now: datetime) -> AsyncBatchResult:
        """Plan sequentially, execute concurrently, apply in planning order."""
        import time as _time

        started = _time.perf_counter()
        result = AsyncBatchResult()

        # Deterministic planning order. Sorting by item id rather than trusting
        # the caller means two workers handed the same set in different orders
        # produce the same plan.
        ordered = sorted(items, key=lambda i: i.item_id)

        planned: list[_Planned] = []
        for order, item in enumerate(ordered):
            if item.state.is_terminal:
                continue
            customer = self.world.customers[item.customer_id]
            action = self.agent.decide(item, customer, now, self)
            result.planned += 1

            if action is None or action.kind in (ActionKind.WAIT, ActionKind.STOP):
                continue
            if action.scheduled_at > now:
                continue

            decision = self.policy.evaluate(action, self._context(item, customer, now))
            if not decision.allowed:
                result.denied += 1
                for verdict in decision.denials:
                    result.denials[verdict.rule_id] = result.denials.get(verdict.rule_id, 0) + 1
                self.agent.on_denied(item, action, decision)
                self.ledger.append(
                    now, "policy_denied", item.item_id,
                    {"action": action.to_dict(), "decision": decision.to_dict()},
                )
                continue

            # Reserve the key at planning time. Two concurrent actions on the
            # same item would otherwise both pass R-IDEMPOTENCY, because neither
            # is recorded until it completes.
            self._used_keys.add(action.idempotency_key)
            planned.append(_Planned(order=order, item=item, customer=customer, action=action))

        if not planned:
            result.wall_seconds = _time.perf_counter() - started
            return result

        outcomes = await gather_bounded(
            [
                (lambda p=p: asyncio.to_thread(self.executor.execute, p.action, p.item, p.customer))
                for p in planned
            ],
            limit=self.concurrency,
        )

        # Apply in planning order, never completion order. This is what keeps the
        # ledger identical between a concurrent run and a serial one.
        for plan, outcome in zip(planned, outcomes):
            if isinstance(outcome, BaseException):
                result.failed += 1
                self.ledger.append(
                    now, "execution_failed", plan.item.item_id,
                    {"action": plan.action.to_dict(), "error": f"{type(outcome).__name__}: {outcome}"},
                )
                continue
            self._apply(plan, outcome, now, result)

        result.wall_seconds = _time.perf_counter() - started
        return result

    def _apply(self, plan: _Planned, execution: Any, now: datetime, result: AsyncBatchResult) -> None:
        item, customer, action = plan.item, plan.customer, plan.action
        item.attempts += 1
        item.cost_paise += execution.cost_paise
        result.cost_paise += execution.cost_paise

        if not execution.executed:
            self.ledger.append(
                now, "execution_failed", item.item_id,
                {"action": action.to_dict(), "error": execution.error, "retryable": execution.retryable},
            )
            result.failed += 1
            return

        result.executed += 1
        if action.kind.is_contact:
            item.contacts += 1
            item.last_contact_at = now
            self._contacts.setdefault(customer.customer_id, []).append(now)
        if action.kind == ActionKind.PRENOTIFY:
            item.prenotified_at = now
        if action.kind == ActionKind.HUMAN_ESCALATION:
            item.escalated_at = now

        outcome = execution.outcome
        if outcome is not None:
            customer.annoyance += outcome.annoyance_delta
            if outcome.opted_out is not None:
                customer.opted_out_channels.add(outcome.opted_out)
            if outcome.success and outcome.recovered_paise > 0:
                item.recovered_paise = min(
                    item.amount_paise, item.recovered_paise + outcome.recovered_paise
                )
                result.recovered_paise += outcome.recovered_paise
                item.state = (
                    ItemState.RECOVERED if item.outstanding_paise <= 0 else ItemState.PARTIALLY_RECOVERED
                )
            elif item.state == ItemState.AT_RISK:
                item.state = ItemState.IN_RECOVERY

        self.ledger.append(
            now, "action", item.item_id,
            {"action": action.to_dict(), "result": execution.to_dict()},
        )


async def drain(
    world: Any,
    agent: Any,
    policy: PolicyEngine,
    llm: Any,
    *,
    now: Optional[datetime] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    ledger: Optional[BaseLedger] = None,
) -> AsyncBatchResult:
    """Convenience entry point: build the async stack and drain one batch."""
    from .execute.razorpay_client import RazorpayClient

    client = AsyncRazorpayClient(RazorpayClient(mode="mock", seed=world.seed), concurrency=concurrency)
    runtime = AsyncRuntime(world, agent, policy, client, llm, concurrency=concurrency, ledger=ledger)
    return await runtime.run_batch(list(world.items), now or world.start)
