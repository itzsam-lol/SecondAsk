"""The oracle. An upper bound, not an agent.

This one cheats. It reads ``item._blocker`` and ``item._resolves_at`` directly,
so it knows the true cause of every failure and the exact instant a bank outage
closes or a salary lands. No deployable system can do this.

It is in the repository because a recovery number with no ceiling is hard to
read. "SecondAsk recovered 11% of value" could mean it captured most of what was
available or a small fraction of it, and those are very different results. The
oracle answers that: it is what perfect information would achieve under the same
policy gate, the same capacity limits and the same contact caps.

It is deliberately still **policy gated**. An oracle that also ignored the rules
would measure two things at once and would not tell you what information alone
is worth.

The gap between SecondAsk and the oracle is the value of information the model
has not extracted. The gap between SecondAsk and the baselines is the value of
the information it has. Both are worth knowing, and the second is only
interpretable next to the first.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from ..policy.engine import ProposedAction
from ..world.entities import ActionKind, Blocker, Customer, RiskItem
from .base import Agent


class OracleAgent(Agent):
    name = "oracle_upper_bound"
    handles_replies = True

    def decide(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        if item.outstanding_paise <= 0:
            return self.stop(item, "nothing outstanding")

        blocker = item._blocker  # the cheat, and the whole point
        resolves = item._resolves_at

        if blocker == Blocker.UNREACHABLE:
            # Perfect information says this money is gone. Spending anything on
            # it is a certain loss, so the optimal action is to stop at once.
            return self.stop(item, "oracle: customer is unreachable, nothing will work")

        if blocker == Blocker.DISPUTE:
            if not self.blocked(item, ActionKind.HUMAN_ESCALATION, now):
                return self.build(
                    ActionKind.HUMAN_ESCALATION, item, customer, now, runtime,
                    rationale="oracle: disputed, only a human resolves this",
                )
            return self.stop(item, "oracle: disputed and no human capacity available")

        if blocker == Blocker.INSTRUMENT_DEAD:
            # Retries are worth exactly zero. Only the customer can fix it.
            for kind in (ActionKind.UPDATE_INSTRUMENT, ActionKind.PAYMENT_LINK_WHATSAPP):
                if not self.blocked(item, kind, now):
                    return self.build(
                        kind, item, customer, now, runtime,
                        rationale="oracle: instrument is dead, ask for a replacement",
                    )
            return self.stop(item, "oracle: instrument dead and no channel available")

        if blocker in (Blocker.TRANSIENT_INFRA, Blocker.LIQUIDITY):
            if resolves is None:
                return self.stop(item, "oracle: this never becomes recoverable")
            if now < resolves:
                # Wait for the exact instant. This is the information advantage
                # that SecondAsk has to infer from downtime feeds and calendars.
                return self.wait(item, resolves + timedelta(minutes=10), "oracle: waiting for the blocker to clear")
            if item.method.is_mandate:
                if not self._prenotified(item, now):
                    return self.build(
                        ActionKind.PRENOTIFY, item, customer, now, runtime,
                        rationale="oracle: notice before debit",
                    )
                if not self.blocked(item, ActionKind.MANDATE_DEBIT, now):
                    return self.build(
                        ActionKind.MANDATE_DEBIT, item, customer, now, runtime,
                        rationale="oracle: blocker has cleared, debit now",
                    )
            for kind in (ActionKind.PAYMENT_LINK_WHATSAPP, ActionKind.PAYMENT_LINK_SMS):
                if not self.blocked(item, kind, now):
                    return self.build(
                        kind, item, customer, now, runtime,
                        rationale="oracle: blocker has cleared, ask now",
                    )
            return self.stop(item, "oracle: no channel available")

        if blocker == Blocker.AUTH_FRICTION:
            # Intent was present and the window closes fast, so act immediately.
            for kind in (ActionKind.PAYMENT_LINK_WHATSAPP, ActionKind.PAYMENT_LINK_SMS):
                if not self.blocked(item, kind, now):
                    return self.build(
                        kind, item, customer, now, runtime,
                        rationale="oracle: authentication dropped, re-ask while intent is warm",
                    )
            return self.stop(item, "oracle: no channel available")

        # INTENT_LOST: only an incentive moves this, and only early.
        if (now - item.failed_at) > timedelta(days=5):
            return self.stop(item, "oracle: intent is gone and the window has closed")
        for kind in (ActionKind.INCENTIVE_OFFER, ActionKind.PAYMENT_LINK_WHATSAPP):
            if not self.blocked(item, kind, now):
                return self.build(
                    kind, item, customer, now, runtime,
                    rationale="oracle: lost intent, an incentive is the only lever",
                )
        return self.stop(item, "oracle: no channel available")

    def _prenotified(self, item: RiskItem, now: datetime) -> bool:
        if item.prenotified_at is None:
            return False
        hours = (now - item.prenotified_at).total_seconds() / 3600.0
        return 24.0 <= hours <= 24.0 * 7

    def next_visit(self, item, customer, now, runtime) -> Optional[datetime]:
        if item.attempts >= 5:
            return None
        if item.method.is_mandate and item.prenotified_at is not None:
            matured = item.prenotified_at + timedelta(hours=25)
            if matured > now:
                return matured
        return now + timedelta(hours=8)
