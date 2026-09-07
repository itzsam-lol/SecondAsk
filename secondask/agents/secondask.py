"""SecondAsk.

The agent is deliberately thin. Almost everything it does is delegate:

* the **underwriter** says what each action is worth,
* the **planner** picks the best (action, time) pair or says stop,
* the **policy engine** decides whether that is permitted,
* the **model** handles only language.

What is left here is sequencing, and three pieces of domain logic that belong to
the agent rather than to any of those components:

**Mandates need a pre-debit notice.** RBI requires the customer be told at least
24 hours before a recurring debit. So when the plan says "debit this mandate",
the agent must first check whether a valid notice exists and, if not, send one
and come back a day later. The planner does not know this; the policy engine
would simply refuse the debit. The agent is what turns a refusal into a
sequence.

**A dead instrument needs the customer, not another attempt.** When the error
signature points at an instrument problem, an "update your card" message is a
different thing from a payment reminder, and the underwriter prices it
separately.

**A dispute belongs to a human.** Detected from the reply parser, and the agent
escalates rather than continuing to send.

Everything else, including when to give up, falls out of the expected value
being negative. There is no attempt counter driving the stopping decision.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Optional

from ..policy.engine import PolicyDecision, ProposedAction
from ..underwrite import planner as planning
from ..underwrite.model import Underwriter
from ..world.entities import ActionKind, Customer, ErrorReason, Method, RiskItem
from .base import Agent

DEFAULT_MODEL_PATH = os.path.join("models", "underwriter.json")

INSTRUMENT_REASONS = frozenset(
    {
        ErrorReason.CARD_EXPIRED,
        ErrorReason.CARD_DECLINED,
        ErrorReason.CARD_DISABLED_ONLINE,
        ErrorReason.MANDATE_REVOKED,
        ErrorReason.MANDATE_NOT_FOUND,
        ErrorReason.ACCOUNT_CLOSED,
        ErrorReason.ACCOUNT_FROZEN,
    }
)


class SecondAskAgent(Agent):
    name = "secondask"
    handles_replies = True

    def __init__(
        self,
        *,
        underwriter: Optional[Underwriter] = None,
        model_path: str = DEFAULT_MODEL_PATH,
        use_underwriter: bool = True,
        annoyance_price_paise: int = planning.ANNOYANCE_PRICE_PAISE,
        annoyance_budget: float = planning.ANNOYANCE_BUDGET,
    ) -> None:
        super().__init__()
        self.use_underwriter = use_underwriter
        self.annoyance_price_paise = annoyance_price_paise
        self.annoyance_budget = annoyance_budget
        self.underwriter = underwriter
        self.model_path = model_path
        self._last_plan: dict[str, planning.Plan] = {}
        # item -> (when the notice matures, the priced plan that justified it).
        # The valuation is carried across the 24 hour wait rather than recomputed,
        # so the receipt for the debit shows the numbers the decision was actually
        # made on. Without this the debit renders as p=0, ev=0, which looks broken
        # precisely when the agent has done the right thing.
        self._pending_debit: dict[str, tuple[datetime, float, int]] = {}
        if use_underwriter and self.underwriter is None:
            self.underwriter = self._load_or_train(model_path)
        if not use_underwriter:
            self.name = "secondask_no_underwriter"

    @staticmethod
    def _load_or_train(path: str) -> Underwriter:
        """Load the fitted model, training it on first use if absent.

        Training runs on seeds disjoint from every evaluation seed, so a
        first-run auto-train cannot leak into a reported number.
        """
        if os.path.exists(path):
            try:
                return Underwriter.load(path)
            except (ValueError, KeyError, OSError):
                # A stale model from an older feature layout. Refit rather than
                # silently scoring against mismatched coefficients.
                pass
        from ..underwrite.training import train

        underwriter = train()
        try:
            underwriter.save(path)
        except OSError:
            pass
        return underwriter

    # -- decision ----------------------------------------------------------

    def decide(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        if item.outstanding_paise <= 0:
            return self.stop(item, "nothing outstanding")

        # A dispute or declared hardship is not an optimisation problem. Hand it
        # to a person and stop the automated sequence, which is also what
        # R-STOP-DISPUTE would enforce if the agent tried anything else.
        if customer.disputed or customer.hardship:
            if not self.blocked(item, ActionKind.HUMAN_ESCALATION, now):
                return self.build(
                    ActionKind.HUMAN_ESCALATION, item, customer, now, runtime,
                    rationale="customer disputed or declared hardship, this needs a human",
                )
            return self.stop(item, "disputed or hardship, and escalation is unavailable")

        if not self.use_underwriter:
            return self._heuristic(item, customer, now, runtime)

        # A mandate debit was planned and the pre-debit notice has now matured.
        pending = self._pending_debit.get(item.item_id)
        if pending is not None and now >= pending[0]:
            _, planned_p, planned_ev = pending
            self._pending_debit.pop(item.item_id, None)
            if not self.blocked(item, ActionKind.MANDATE_DEBIT, now):
                return self.build(
                    ActionKind.MANDATE_DEBIT, item, customer, now, runtime,
                    rationale="pre-debit notice has matured past 24 hours, debiting",
                    p_recover=planned_p,
                    expected_value_paise=planned_ev,
                )

        assert self.underwriter is not None
        world = runtime.world
        result = planning.plan(
            self.underwriter,
            item,
            customer,
            now,
            world.downtime,
            world.bank_of(item),
            world.end,
            blocked=self._blocked_now(item, now),
            annoyance_price_paise=self.annoyance_price_paise,
            annoyance_budget=self.annoyance_budget,
        )
        self._last_plan[item.item_id] = result

        if result.best is None:
            return self.stop(item, result.reason)

        best = result.best
        if best.at > now:
            return self.wait(item, best.at, f"{best.note}; {best.action.value} is worth {best.ev_paise} paise then")

        # A mandate debit needs a valid pre-debit notice first. Send the notice
        # now and schedule the debit for just over 24 hours later.
        if best.action == ActionKind.MANDATE_DEBIT and not self._prenotice_valid(item, now):
            self._pending_debit[item.item_id] = (now + timedelta(hours=25), best.p_recover, best.ev_paise)
            return self.build(
                ActionKind.PRENOTIFY, item, customer, now, runtime,
                rationale="pre-debit notice required at least 24h before a mandate debit",
                p_recover=best.p_recover,
                expected_value_paise=best.ev_paise,
            )

        kind = best.action
        # Route an instrument problem to the message that can actually fix it.
        if kind == ActionKind.PAYMENT_LINK_SMS and self._looks_like_dead_instrument(item):
            if not self.blocked(item, ActionKind.UPDATE_INSTRUMENT, now):
                kind = ActionKind.UPDATE_INSTRUMENT

        return self.build(
            kind, item, customer, now, runtime,
            rationale=self._rationale(item, best),
            p_recover=best.p_recover,
            expected_value_paise=best.ev_paise,
        )

    def _blocked_now(self, item: RiskItem, now: datetime) -> set[ActionKind]:
        """Actions unavailable at this instant, permanent or merely for now."""
        entry = self._blocked.get(item.item_id, {})
        return {kind for kind, until in entry.items() if until is None or now < until}

    def _rationale(self, item: RiskItem, best: planning.Candidate) -> str:
        return (
            f"{item.error_source.value}/{item.error_reason.value} on {item.method.value}: "
            f"p={best.p_recover:.3f}, gross={best.gross_paise}p, "
            f"goodwill={best.goodwill_paise}p, ev={best.ev_paise}p"
        )

    def _prenotice_valid(self, item: RiskItem, now: datetime) -> bool:
        if item.prenotified_at is None:
            return False
        hours = (now - item.prenotified_at).total_seconds() / 3600.0
        return 24.0 <= hours <= 24.0 * 7

    def _looks_like_dead_instrument(self, item: RiskItem) -> bool:
        """Observable signature only. The latent blocker is not readable here."""
        return item.error_reason in INSTRUMENT_REASONS

    # -- ablation path ------------------------------------------------------

    def _heuristic(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        """The no-underwriter ablation.

        Keeps the policy gate and the model, and replaces expected value pricing
        with sensible domain rules of the kind a careful engineer would write
        without a model. It is not a straw man: it knows about dead instruments,
        it knows mandates need a notice, and it stops after a fixed number of
        attempts. What it cannot do is price anything, so it cannot tell a
        contact worth making from one that is not, and it has no basis for
        waiting until payday.
        """
        if item.attempts >= 5:
            return self.stop(item, "heuristic attempt budget exhausted")

        if item.method.is_mandate:
            if not self._prenotice_valid(item, now):
                if item.prenotified_at is None and not self.blocked(item, ActionKind.PRENOTIFY, now):
                    return self.build(
                        ActionKind.PRENOTIFY, item, customer, now, runtime,
                        rationale="notice before debit",
                    )
            elif not self.blocked(item, ActionKind.MANDATE_DEBIT, now):
                return self.build(
                    ActionKind.MANDATE_DEBIT, item, customer, now, runtime, rationale="retry the mandate"
                )

        if self._looks_like_dead_instrument(item) and not self.blocked(item, ActionKind.UPDATE_INSTRUMENT, now):
            return self.build(
                ActionKind.UPDATE_INSTRUMENT, item, customer, now, runtime,
                rationale="error signature indicates the instrument needs replacing",
            )

        for kind in (
            ActionKind.PAYMENT_LINK_WHATSAPP,
            ActionKind.PAYMENT_LINK_SMS,
            ActionKind.PAYMENT_LINK_EMAIL,
        ):
            if not self.blocked(item, kind, now):
                return self.build(kind, item, customer, now, runtime, rationale="follow up on the failure")
        return self.stop(item, "no available channel")

    # -- scheduling ---------------------------------------------------------

    def next_visit(self, item, customer, now, runtime) -> Optional[datetime]:
        pending = self._pending_debit.get(item.item_id)
        if pending is not None:
            return pending[0]
        if not self.use_underwriter:
            if item.attempts >= 5:
                return None
            return now + timedelta(hours=20)
        # Come back soon enough to re-plan against fresh state, since a downtime
        # window closing or a payday arriving changes the answer. The planner
        # will return WAIT if there is nothing worth doing yet, and a WAIT is
        # cheap: no gateway call, no message, one model evaluation per candidate.
        return now + timedelta(hours=6)

    def on_denied(self, item, action, decision) -> None:
        super().on_denied(item, action, decision)
        if action.kind == ActionKind.MANDATE_DEBIT:
            self._pending_debit.pop(item.item_id, None)

    def last_plan(self, item_id: str) -> Optional[planning.Plan]:
        """Exposed for the decision receipt in the dashboard."""
        return self._last_plan.get(item_id)
