"""The run loop.

Discrete event simulation over a virtual clock. Each item sits in a priority
queue keyed by the next instant something should be decided about it. Popping an
item asks the agent for one action, gates it, performs it, applies the
consequences and schedules the next visit.

The parts that took the most care are the loop guarantees, because a recovery
agent that can be talked into an infinite loop by its own policy engine is not a
recovery agent:

* **Progress.** An agent that returns ``WAIT`` for a time that is not strictly in
  the future is forced forward by one hour. Without this, an agent with an
  off-by-one in its scheduling spins on a single timestamp forever.
* **Bounded deferral.** A policy denial that offers a retry time reschedules the
  item, but only ``MAX_DEFERRALS`` times. Two rules can otherwise trade denials
  and bounce an item between them.
* **Bounded visits.** A hard cap per item, independent of both of the above.
* **Horizon.** Anything scheduled past the end of the world is dropped, and the
  item is settled into a terminal state so it appears in the exception list
  rather than silently vanishing from the accounting.

Every one of those limits is counted and reported. A guard that fires silently
is a bug that has been hidden rather than fixed.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from .clock import VirtualClock, ist_date, iso
from .execute.executor import Executor
from .execute.razorpay_client import RazorpayClient
from .ledger import Ledger
from .llm.gateway import LLMGateway
from .policy.engine import PolicyContext, PolicyEngine, ProposedAction
from .world.entities import (
    ACTION_CHANNEL,
    ActionKind,
    Channel,
    Customer,
    ItemState,
    Method,
    ReplyIntent,
    RiskItem,
)
from .world.generator import World

MAX_VISITS_PER_ITEM = 120
MAX_DEFERRALS_PER_ITEM = 24
MAX_PROPOSALS_PER_VISIT = 4
MIN_STEP = timedelta(hours=1)


@dataclass
class RunResult:
    agent_name: str
    seed: int
    at_risk_paise: int = 0
    recovered_paise: int = 0
    cost_paise: int = 0

    items_total: int = 0
    items_recovered: int = 0
    items_partial: int = 0
    items_escalated: int = 0
    items_stopped_early: int = 0
    items_exhausted: int = 0

    actions_executed: int = 0
    actions_by_kind: dict[str, int] = field(default_factory=dict)
    messages_sent: int = 0
    messages_by_channel: dict[str, int] = field(default_factory=dict)
    delivery_failures: int = 0
    silent_retries: int = 0
    wasted_retries: int = 0

    policy_denials: dict[str, int] = field(default_factory=dict)
    violations: dict[str, int] = field(default_factory=dict)

    opt_outs: int = 0
    complaints: int = 0
    churns: int = 0
    replies_received: int = 0
    promises_captured: int = 0
    partial_promises: int = 0
    disputes_detected: int = 0
    hardship_detected: int = 0

    guard_hits: dict[str, int] = field(default_factory=dict)
    decision_latencies_ms: list[float] = field(default_factory=list, repr=False)

    razorpay: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    ledger_head: str = ""
    ledger_entries: int = 0
    ledger_verified: bool = False

    @property
    def violating_actions(self) -> int:
        """Actions executed that broke at least one rule.

        Reported alongside the rule-level total because one bad send usually
        trips several rules at once, and quoting only the rule count would
        overstate how many customer-visible incidents occurred.
        """
        return self.violations.get("_actions", 0)

    @property
    def total_violations(self) -> int:
        return sum(v for k, v in self.violations.items() if k != "_actions")

    @property
    def net_paise(self) -> int:
        return self.recovered_paise - self.cost_paise

    def to_dict(self) -> dict[str, Any]:
        latencies = sorted(self.decision_latencies_ms)
        return {
            "agent": self.agent_name,
            "seed": self.seed,
            "at_risk_paise": self.at_risk_paise,
            "recovered_paise": self.recovered_paise,
            "cost_paise": self.cost_paise,
            "net_paise": self.net_paise,
            "items_total": self.items_total,
            "items_recovered": self.items_recovered,
            "items_partial": self.items_partial,
            "items_escalated": self.items_escalated,
            "items_stopped_early": self.items_stopped_early,
            "items_exhausted": self.items_exhausted,
            "actions_executed": self.actions_executed,
            "actions_by_kind": dict(sorted(self.actions_by_kind.items())),
            "messages_sent": self.messages_sent,
            "messages_by_channel": dict(sorted(self.messages_by_channel.items())),
            "delivery_failures": self.delivery_failures,
            "silent_retries": self.silent_retries,
            "wasted_retries": self.wasted_retries,
            "policy_denials": dict(sorted(self.policy_denials.items())),
            "violations": {k: v for k, v in sorted(self.violations.items()) if k != "_actions"},
            "total_violations": self.total_violations,
            "violating_actions": self.violating_actions,
            "opt_outs": self.opt_outs,
            "complaints": self.complaints,
            "churns": self.churns,
            "replies_received": self.replies_received,
            "promises_captured": self.promises_captured,
            "partial_promises": self.partial_promises,
            "disputes_detected": self.disputes_detected,
            "hardship_detected": self.hardship_detected,
            "guard_hits": dict(sorted(self.guard_hits.items())),
            "p50_decision_ms": _percentile(latencies, 0.50),
            "p95_decision_ms": _percentile(latencies, 0.95),
            "razorpay": self.razorpay,
            "llm": self.llm,
            "ledger_head": self.ledger_head,
            "ledger_entries": self.ledger_entries,
            "ledger_verified": self.ledger_verified,
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return round(sorted_values[index], 3)


def _bump(counter: dict[str, int], key: str, by: int = 1) -> None:
    counter[key] = counter.get(key, 0) + by


class Runtime:
    """Owns world state mutation. Agents propose; only this class applies."""

    def __init__(
        self,
        world: World,
        agent: Any,
        policy: PolicyEngine,
        client: RazorpayClient,
        llm: LLMGateway,
        *,
        spend_cap_paise: int = 0,
        escalation_daily_cap: int = 0,
        merchant_name: str = "Acme Retail",
    ) -> None:
        self.world = world
        self.agent = agent
        self.policy = policy
        self.client = client
        self.llm = llm
        self.clock = VirtualClock(world.start)
        self.ledger = Ledger(run_id=f"{agent.name}@{world.seed}")
        self.executor = Executor(world, client, llm, merchant_name=merchant_name)
        self.spend_cap_paise = spend_cap_paise
        self.spend_paise = 0
        # Human review capacity, expressed per day. Derived from the size of the
        # batch on the assumption that one ops person works about 25 recovery
        # cases a day and a team is staffed at roughly one person per 150 open
        # items. Escalation is the most effective action available for a
        # disputed or high value case, so without a capacity limit an expected
        # value planner correctly concludes that everything should go to a
        # person, and produces a plan no team could execute.
        self.escalation_daily_cap = (
            escalation_daily_cap if escalation_daily_cap > 0 else max(3, len(world.items) // 150)
        )
        self._escalations_by_day: dict[str, int] = {}

        self._contacts: dict[str, list[datetime]] = {}
        self._used_keys: set[str] = set()
        self._visits: dict[str, int] = {}
        self._deferrals: dict[str, int] = {}
        self.result = RunResult(agent_name=agent.name, seed=world.seed)

    # -- helpers ------------------------------------------------------------

    def _contact_counts(self, customer_id: str, now: datetime) -> tuple[int, int, Optional[datetime]]:
        history = self._contacts.get(customer_id, [])
        if not history:
            return 0, 0, None
        day_ago = now - timedelta(hours=24)
        week_ago = now - timedelta(days=7)
        in_24h = sum(1 for t in history if t > day_ago)
        in_7d = sum(1 for t in history if t > week_ago)
        return in_24h, in_7d, history[-1]

    def _context(self, item: RiskItem, customer: Customer, now: datetime) -> PolicyContext:
        in_24h, in_7d, last = self._contact_counts(customer.customer_id, now)
        return PolicyContext(
            item=item,
            customer=customer,
            now=now,
            contacts_24h=in_24h,
            contacts_7d=in_7d,
            last_contact_at=last,
            used_idempotency_keys=frozenset(self._used_keys),
            spend_paise=self.spend_paise,
            spend_cap_paise=self.spend_cap_paise,
            consent_promotional=customer.tenure_days > 30,
            dnd_registered=customer.prior_ignores >= 3,
            escalations_today=self._escalations_by_day.get(ist_date(now).isoformat(), 0),
            escalation_daily_cap=self.escalation_daily_cap,
            merchant_name=self.executor.merchant_name,
        )

    # -- main loop ----------------------------------------------------------

    def run(self) -> RunResult:
        world = self.world
        self.result.items_total = len(world.items)
        self.result.at_risk_paise = world.total_at_risk_paise

        queue: list[tuple[datetime, int, str]] = []
        by_id = {item.item_id: item for item in world.items}
        counter = 0
        for item in world.items:
            # First look happens a few minutes after the failure. Nothing useful
            # can be decided before the failure event itself exists.
            heapq.heappush(queue, (item.failed_at + timedelta(minutes=3), counter, item.item_id))
            counter += 1

        self.agent.on_run_start(world, self.clock)

        while queue:
            when, _, item_id = heapq.heappop(queue)
            if when >= world.end:
                _bump(self.result.guard_hits, "past_horizon")
                continue

            item = by_id[item_id]
            if item.state.is_terminal:
                continue

            visits = self._visits.get(item_id, 0)
            if visits >= MAX_VISITS_PER_ITEM:
                _bump(self.result.guard_hits, "max_visits")
                self._finalise(item, ItemState.STOPPED, when, "visit cap reached")
                continue
            self._visits[item_id] = visits + 1

            self.clock.advance_to(when)
            now = self.clock.now
            customer = world.customers[item.customer_id]

            import time as _time

            # A visit may involve several proposals. A refused action is not the
            # end of the item: the agent is told why and gets to propose
            # something else, which is how a real operator behaves and is the
            # only fair way to compare agents. Without this, an agent that
            # opened with one illegal action would be terminated on the spot
            # while an agent that opened legally ran to completion, and the
            # table would be measuring opening moves rather than policies.
            action: Optional[ProposedAction] = None
            decision = None
            outcome_of_visit = "none"
            # Earliest instant at which any refused action would become legal.
            # Collected across the whole visit so that if every proposal is
            # refused we defer to the soonest opening rather than the last one
            # tried.
            deferrals: list[datetime] = []
            for proposal in range(MAX_PROPOSALS_PER_VISIT):
                started = _time.perf_counter()
                action = self.agent.decide(item, customer, now, self)
                self.result.decision_latencies_ms.append((_time.perf_counter() - started) * 1000)

                if action is None or action.kind == ActionKind.STOP:
                    outcome_of_visit = "stop"
                    break

                if action.kind == ActionKind.WAIT:
                    outcome_of_visit = "wait"
                    break

                # An action scheduled in the future is a plan, not a request to
                # act now. Put it back and revisit at the chosen time.
                if action.scheduled_at > now:
                    outcome_of_visit = "future"
                    break

                ctx = self._context(item, customer, now)
                decision = self.policy.evaluate(action, ctx)
                for verdict in decision.denials:
                    _bump(self.result.policy_denials, verdict.rule_id)

                if decision.allowed:
                    outcome_of_visit = "execute"
                    break

                self.ledger.append(
                    now,
                    "policy_denied",
                    item_id,
                    {"action": action.to_dict(), "decision": decision.to_dict(), "proposal": proposal},
                )
                self.agent.on_denied(item, action, decision)
                outcome_of_visit = "denied"
                if decision.earliest_allowed_at is not None:
                    deferrals.append(decision.earliest_allowed_at)
                # Keep going. The refused action may have been the best one, but
                # a second-best action available now is usually worth more than
                # the best one tomorrow, and this is what lets a case blocked on
                # scarce human capacity fall through to a message instead of
                # queueing behind it for the rest of the run.
            else:
                _bump(self.result.guard_hits, "max_proposals_per_visit")

            if outcome_of_visit == "stop":
                reason = action.rationale if action else "agent returned no action"
                self._finalise(item, ItemState.STOPPED, now, reason)
                self.result.items_stopped_early += 1
                continue

            if outcome_of_visit == "wait":
                assert action is not None
                nxt = action.scheduled_at
                if nxt <= now:
                    nxt = now + MIN_STEP
                    _bump(self.result.guard_hits, "wait_not_advancing")
                self.ledger.append(now, "wait", item_id, {"until": iso(nxt), "why": action.rationale})
                counter += 1
                heapq.heappush(queue, (nxt, counter, item_id))
                continue

            if outcome_of_visit == "future":
                assert action is not None
                counter += 1
                heapq.heappush(queue, (action.scheduled_at, counter, item_id))
                continue

            if outcome_of_visit != "execute":
                assert decision is not None
                nxt = min(deferrals) if deferrals else None
                deferred_count = self._deferrals.get(item_id, 0)
                if nxt is not None and nxt > now and deferred_count < MAX_DEFERRALS_PER_ITEM:
                    self._deferrals[item_id] = deferred_count + 1
                    counter += 1
                    heapq.heappush(queue, (nxt, counter, item_id))
                    continue
                if nxt is not None and deferred_count >= MAX_DEFERRALS_PER_ITEM:
                    _bump(self.result.guard_hits, "max_deferrals")
                # Nothing legal is available and no later time fixes it. The
                # item becomes an exception and is reported as one.
                self._finalise(
                    item, ItemState.STOPPED, now, f"no permitted action: {decision.denied_rules}"
                )
                continue

            assert action is not None and decision is not None

            # The policy engine ran with the gate open (ablation). Anything it
            # would have denied and we did anyway is a measured violation.
            if not self.policy.enabled and decision.denials:
                _bump(self.result.violations, "_actions")
                for verdict in decision.denials:
                    _bump(self.result.violations, verdict.rule_id)

            result = self.executor.execute(action, item, customer)
            self._apply(item, customer, action, result, now)

            if not result.executed:
                self.ledger.append(
                    now,
                    "execution_failed",
                    item_id,
                    {"action": action.to_dict(), "error": result.error, "retryable": result.retryable},
                )
                if result.retryable:
                    # The gateway was unavailable. The item is not lost; it is
                    # tried again once the breaker has had time to reset.
                    counter += 1
                    heapq.heappush(queue, (now + timedelta(minutes=15), counter, item_id))
                else:
                    self._finalise(item, ItemState.STOPPED, now, result.error or "execution failed")
                continue

            if item.state.is_terminal:
                continue

            nxt = self.agent.next_visit(item, customer, now, self)
            if nxt is None:
                self._finalise(item, ItemState.STOPPED, now, "agent scheduled no further visit")
                self.result.items_stopped_early += 1
                continue
            if nxt <= now:
                nxt = now + MIN_STEP
                _bump(self.result.guard_hits, "next_visit_not_advancing")
            counter += 1
            heapq.heappush(queue, (nxt, counter, item_id))

        self._close_out()
        return self.result

    # -- state application --------------------------------------------------

    def _apply(
        self,
        item: RiskItem,
        customer: Customer,
        action: ProposedAction,
        result: Any,
        now: datetime,
    ) -> None:
        self._used_keys.add(action.idempotency_key)
        item.attempts += 1
        item.last_action_at = now
        item.cost_paise += result.cost_paise
        self.spend_paise += result.cost_paise
        self.result.cost_paise += result.cost_paise
        self.result.actions_executed += 1
        _bump(self.result.actions_by_kind, action.kind.value)

        if action.kind == ActionKind.PRENOTIFY:
            item.prenotified_at = now

        if action.kind == ActionKind.HUMAN_ESCALATION:
            item.escalated_at = now
            day = ist_date(now).isoformat()
            self._escalations_by_day[day] = self._escalations_by_day.get(day, 0) + 1

        channel = ACTION_CHANNEL[action.kind]
        if action.kind.is_contact:
            item.contacts += 1
            item.last_contact_at = now
            self._contacts.setdefault(customer.customer_id, []).append(now)
            self.result.messages_sent += 1
            _bump(self.result.messages_by_channel, channel.value)

        if action.kind in (ActionKind.SILENT_RETRY, ActionKind.MANDATE_DEBIT):
            self.result.silent_retries += 1

        if result.delivery_failed:
            self.result.delivery_failures += 1

        outcome = result.outcome
        entry = {
            "action": action.to_dict(),
            "result": result.to_dict(),
            "item_state_before": item.state.value,
        }

        if outcome is None:
            self.ledger.append(now, "action", item.item_id, entry)
            return

        if outcome.success and outcome.recovered_paise > 0:
            item.recovered_paise += outcome.recovered_paise
            # Clamp defensively. A partial-payment rounding path could otherwise
            # push recovered past the amount owed and inflate the headline
            # number, which is the one number that must never be wrong.
            if item.recovered_paise > item.amount_paise:
                item.recovered_paise = item.amount_paise
                _bump(self.result.guard_hits, "overpayment_clamped")
            self.result.recovered_paise += outcome.recovered_paise
            if item.outstanding_paise <= 0:
                item.state = ItemState.RECOVERED
                self.result.items_recovered += 1
            else:
                item.state = ItemState.PARTIALLY_RECOVERED
                self.result.items_partial += 1
        else:
            if action.kind in (ActionKind.SILENT_RETRY, ActionKind.MANDATE_DEBIT):
                self.result.wasted_retries += 1
            if item.state == ItemState.AT_RISK:
                item.state = ItemState.IN_RECOVERY

        customer.annoyance += outcome.annoyance_delta

        # Hard opt-out. The customer blocked the sender or registered a
        # preference. This applies to every agent whether or not it reads
        # replies, because it happens on the customer's side.
        if outcome.opted_out is not None:
            customer.opted_out_channels.add(outcome.opted_out)
            self.result.opt_outs += 1
        if outcome.complained:
            customer.complained = True
            self.result.complaints += 1
        if outcome.churned:
            customer.churned = True
            self.result.churns += 1

        if outcome.reply_text:
            self.result.replies_received += 1
            self._handle_reply(item, customer, outcome.reply_text, now, entry)

        # A handover is a handover. Whether or not the human collected the money,
        # the case now belongs to them and leaves the automated loop. Letting the
        # agent keep working an escalated case means two parties chasing the same
        # customer, which is worse than either doing it alone.
        if action.kind == ActionKind.HUMAN_ESCALATION and not item.state.is_terminal:
            if item.outstanding_paise <= 0:
                item.state = ItemState.RECOVERED
            else:
                item.state = ItemState.ESCALATED
                self.result.items_escalated += 1
            self.ledger.append(
                now, "closed", item.item_id,
                {"state": item.state.value, "reason": "handed to human review"},
            )

        self.ledger.append(now, "action", item.item_id, entry)

    def _handle_reply(
        self,
        item: RiskItem,
        customer: Customer,
        text: str,
        now: datetime,
        entry: dict[str, Any],
    ) -> None:
        """Read the customer's reply, if this agent reads replies at all.

        An agent with ``handles_replies = False`` still receives the message; it
        simply does nothing with it. That is the ablation: the cost of ignoring
        inbound text shows up as disputes never detected, promises never
        honoured, and complaints that follow from both.
        """
        entry["reply_text"] = text
        if not getattr(self.agent, "handles_replies", False):
            entry["reply_handled"] = False
            return

        parsed = self.llm.parse_reply(text, now)
        entry["reply_handled"] = True
        entry["reply_parsed"] = parsed.to_dict()

        if parsed.intent == ReplyIntent.OPT_OUT:
            # Honour it on every text channel, not just the one they replied on.
            # Somebody who says stop has said stop.
            for channel in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL, Channel.VOICE):
                customer.opted_out_channels.add(channel)
        elif parsed.intent == ReplyIntent.DISPUTE:
            customer.disputed = True
            self.result.disputes_detected += 1
        elif parsed.intent == ReplyIntent.HARDSHIP:
            customer.hardship = True
            self.result.hardship_detected += 1
        elif parsed.intent == ReplyIntent.WRONG_NUMBER:
            customer.opted_out_channels.add(Channel.SMS)
            customer.opted_out_channels.add(Channel.WHATSAPP)
            customer.opted_out_channels.add(Channel.VOICE)
        elif parsed.intent in (ReplyIntent.PROMISE_TO_PAY, ReplyIntent.PARTIAL_PAYMENT_PROMISE):
            # Cap how far a promise can push the next contact. Otherwise "I will
            # pay in December" silently writes the item off.
            promised = parsed.promise_date or (now + timedelta(days=3))
            item.promise_to_pay_at = min(promised, now + timedelta(days=21))
            self.result.promises_captured += 1
            if parsed.intent == ReplyIntent.PARTIAL_PAYMENT_PROMISE:
                self.result.partial_promises += 1
                # Recorded as a claim on the item, for the operator and for later
                # reconciliation against what actually arrives.
                #
                # It is deliberately NOT written to any field an action reads.
                # This number came out of a customer's message, and the whole
                # point of R-AMOUNT-BOUND is that a message cannot set the amount
                # on a money-moving action. Assigning it to outstanding_paise
                # here would be a one-line way to let anybody who owes money
                # decide what they owe.
                item.claimed_partial_paise = parsed.claimed_partial_paise
                entry["claimed_partial_paise"] = parsed.claimed_partial_paise
                entry["claim_note"] = (
                    "customer-stated amount, recorded as a claim only; "
                    "actions remain bound to the ledger balance"
                )
        elif parsed.intent == ReplyIntent.ALREADY_PAID:
            # A claim, not a fact. The ledger is the only authority on
            # settlement, so this records the claim and pauses contact rather
            # than marking anything paid.
            item.promise_to_pay_at = now + timedelta(days=2)
            entry["already_paid_claim"] = "recorded, awaiting payment event; no state change"

    def _finalise(self, item: RiskItem, state: ItemState, now: datetime, reason: str) -> None:
        if item.state.is_terminal:
            return
        if item.recovered_paise > 0 and item.outstanding_paise > 0:
            item.state = ItemState.PARTIALLY_RECOVERED
        else:
            item.state = state
        if state == ItemState.ESCALATED:
            self.result.items_escalated += 1
        self.ledger.append(now, "closed", item.item_id, {"state": item.state.value, "reason": reason})

    def _close_out(self) -> None:
        for item in self.world.items:
            if not item.state.is_terminal and item.state != ItemState.PARTIALLY_RECOVERED:
                self.result.items_exhausted += 1
                self.ledger.append(
                    self.world.end, "closed", item.item_id, {"state": "written_off", "reason": "horizon reached"}
                )
                item.state = ItemState.WRITTEN_OFF

        self.result.razorpay = self.client.to_dict()
        self.result.llm = {"backend": self.llm.backend_name, **self.llm.stats.to_dict()}
        self.result.ledger_head = self.ledger.head
        self.result.ledger_entries = len(self.ledger)
        ok, _ = self.ledger.verify()
        self.result.ledger_verified = ok
