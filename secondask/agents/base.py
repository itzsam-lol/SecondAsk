"""Agent interface.

An agent proposes one action at a time and says when it wants to be asked again.
It cannot mutate world state, cannot send anything, and cannot skip the policy
gate. Everything an agent returns is a request.

Keeping the interface this narrow is what makes the comparison table honest: the
baselines and SecondAsk differ only in what they return from ``decide``, so a
difference in outcomes is a difference in decisions rather than a difference in
what each was allowed to do.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from ..execute.executor import build_slots
from ..policy.engine import PolicyDecision, ProposedAction
from ..policy.templates import template_for
from ..world.entities import (
    ACTION_CHANNEL,
    ActionKind,
    Channel,
    Customer,
    Method,
    RiskItem,
)


class Agent:
    name = "agent"
    handles_replies = False

    def __init__(self) -> None:
        # Actions the policy engine has refused for a given item, mapped to the
        # time they become available again. ``None`` means never.
        #
        # Two different things land here and the distinction matters. A rail with
        # no standing authorisation is permanent, so the agent must stop
        # proposing it entirely. Human review capacity being used up for today
        # is temporary, and the right response is to take the next best action
        # now rather than to sit and wait for tomorrow. Collapsing both into a
        # single permanent block loses real recovery; treating both as temporary
        # spins the proposal loop.
        self._blocked: dict[str, dict[ActionKind, Optional[datetime]]] = {}

    def on_run_start(self, world: Any, clock: Any) -> None:
        """Hook for anything an agent needs to prepare once per run."""

    def blocked(self, item: RiskItem, kind: ActionKind, now: datetime) -> bool:
        """Is this action unavailable for this item at this instant?

        ``now`` is required rather than optional on purpose. An earlier version
        defaulted it, and every caller that forgot to pass it silently turned
        every temporary block into a permanent one, which quietly cost one
        baseline three quarters of its recovery. A required argument makes that
        mistake a TypeError instead of a number nobody questions.
        """
        entry = self._blocked.get(item.item_id)
        if entry is None or kind not in entry:
            return False
        until = entry[kind]
        if until is None:
            return True
        return now < until

    def decide(
        self, item: RiskItem, customer: Customer, now: datetime, runtime: Any
    ) -> Optional[ProposedAction]:
        raise NotImplementedError

    def next_visit(
        self, item: RiskItem, customer: Customer, now: datetime, runtime: Any
    ) -> Optional[datetime]:
        return None

    def on_denied(self, item: RiskItem, action: ProposedAction, decision: PolicyDecision) -> None:
        """Called when the policy engine refuses an action.

        Records when the action becomes available again so the agent can pick
        something else in the meantime. A denial with no retry time is
        structural and blocks the action permanently for this item.
        """
        self._blocked.setdefault(item.item_id, {})[action.kind] = decision.earliest_allowed_at

    # -- construction helpers ----------------------------------------------

    def build(
        self,
        kind: ActionKind,
        item: RiskItem,
        customer: Customer,
        now: datetime,
        runtime: Any,
        *,
        rationale: str = "",
        p_recover: float = 0.0,
        expected_value_paise: int = 0,
        template_id: Optional[str] = None,
    ) -> ProposedAction:
        """Assemble a well formed proposal.

        The amount always comes from ``item.outstanding_paise``. No agent, and in
        particular no model-driven agent, gets to choose it. ``R-AMOUNT-BOUND``
        would catch a deviation anyway; taking the value from the ledger here
        means there is nothing to catch.
        """
        channel = ACTION_CHANNEL[kind]
        amount = item.outstanding_paise
        key = f"{self.name}:{item.item_id}:{item.attempts}:{kind.value}"

        slots: dict[str, str] = {}
        resolved_template = template_id
        if channel in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL):
            if resolved_template is None:
                resolved_template = self._template_for(kind, item, customer)
            slots = build_slots(
                runtime.llm,
                resolved_template,
                item,
                customer,
                amount,
                now,
                runtime.executor.merchant_name,
            )

        return ProposedAction(
            item_id=item.item_id,
            customer_id=customer.customer_id,
            kind=kind,
            channel=channel,
            scheduled_at=now,
            amount_paise=amount,
            idempotency_key=key,
            template_id=resolved_template,
            slots=slots,
            rationale=rationale,
            p_recover=p_recover,
            expected_value_paise=expected_value_paise,
            proposed_by=self.name,
        )

    def _template_for(self, kind: ActionKind, item: RiskItem, customer: Customer) -> str:
        if kind == ActionKind.PRENOTIFY:
            return template_for("prenotify", customer.language)
        if kind == ActionKind.UPDATE_INSTRUMENT:
            return template_for("update_instrument", customer.language)
        if kind == ActionKind.INCENTIVE_OFFER:
            return "INCENTIVE_EN"
        if item.method == Method.INVOICE:
            return "INVOICE_MSME_EN" if item.is_msme_supplier else "INVOICE_DUE_EN"
        return template_for("payment_link", customer.language)

    def wait(self, item: RiskItem, until: datetime, why: str) -> ProposedAction:
        return ProposedAction(
            item_id=item.item_id,
            customer_id=item.customer_id,
            kind=ActionKind.WAIT,
            channel=Channel.NONE,
            scheduled_at=until,
            amount_paise=item.outstanding_paise,
            idempotency_key="",
            rationale=why,
            proposed_by=self.name,
        )

    def stop(self, item: RiskItem, why: str) -> ProposedAction:
        return ProposedAction(
            item_id=item.item_id,
            customer_id=item.customer_id,
            kind=ActionKind.STOP,
            channel=Channel.NONE,
            scheduled_at=item.failed_at,
            amount_paise=item.outstanding_paise,
            idempotency_key="",
            rationale=why,
            proposed_by=self.name,
        )
