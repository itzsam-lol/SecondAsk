"""Baselines.

These exist so that SecondAsk's numbers mean something. A recovery rate quoted
without a comparison is unfalsifiable: it can always be made to look good by
choosing a friendly batch.

``DoNothing``
    The floor. Some money arrives anyway, because customers retry by themselves
    and outages end. Any agent that does not clear this by a wide margin has
    achieved nothing except spending money on messages.

``FixedSchedule``
    What most dunning systems actually do, and the honest comparison: retry at
    +1h, +24h and +72h, with a reminder alongside each attempt. It is not a straw
    man, it is the industry default.

``Aggressive``
    Maximum persistence. Every channel, every day, until something happens. Run
    with the policy gate open, this is what produces the violation counts.

``LLMOnly``
    A single model call in a loop chooses the next action from the full menu with
    no pricing and no gate. This is the shape of most agentic demos, and it is the
    most important baseline in the table: it isolates what the policy engine and
    the underwriter contribute, as distinct from what the model contributes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from ..policy.engine import ProposedAction
from ..world.entities import ActionKind, Customer, Method, RiskItem
from .base import Agent


class DoNothingAgent(Agent):
    name = "b0_do_nothing"

    def decide(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        return self.stop(item, "baseline takes no recovery action")


class FixedScheduleAgent(Agent):
    """Retry at +1h, +24h, +72h with a reminder alongside each.

    Note what it cannot do, which is the point. It has no way to tell an expired
    card from a temporary bank outage, so it spends its three attempts
    identically on both. On mandate rails it retries; on one-off rails the
    silent retry is refused by ``R-SILENT-RETRY-RAIL`` and only the message
    lands, which is also what happens in production.
    """

    name = "b1_fixed_schedule"
    OFFSETS_HOURS = (1, 24, 72)

    def decide(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        stage = self._stage(item, now)
        if stage is None:
            return self.stop(item, "fixed schedule exhausted")

        due = item.failed_at + timedelta(hours=self.OFFSETS_HOURS[stage])
        if now < due:
            return self.wait(item, due, f"waiting for scheduled attempt {stage + 1}")

        # Alternate retry and message. Note it attempts a silent retry on every
        # rail, including one-off UPI and card payments where no standing
        # authorisation exists. That is exactly what a rail-agnostic dunning
        # schedule does, and R-SILENT-RETRY-RAIL refuses it every time. The
        # denial count on that rule measures how often the naive design would
        # have charged somebody it had no mandate to charge.
        retry = ActionKind.MANDATE_DEBIT if item.method.is_mandate else ActionKind.SILENT_RETRY
        if item.attempts % 2 == 0 and not self.blocked(item, retry, now):
            if item.method.is_mandate and item.prenotified_at is None:
                return self.build(
                    ActionKind.PRENOTIFY, item, customer, now, runtime,
                    rationale="pre-debit notice required before a mandate debit",
                )
            return self.build(
                retry, item, customer, now, runtime,
                rationale=f"scheduled retry {stage + 1} at +{self.OFFSETS_HOURS[stage]}h",
            )
        return self.build(
            ActionKind.PAYMENT_LINK_SMS, item, customer, now, runtime,
            rationale=f"scheduled reminder {stage + 1} at +{self.OFFSETS_HOURS[stage]}h",
        )

    def _stage(self, item: RiskItem, now: datetime) -> Optional[int]:
        elapsed = (now - item.failed_at).total_seconds() / 3600.0
        for index, offset in enumerate(self.OFFSETS_HOURS):
            if elapsed < offset + 6:  # a six hour grace band per stage
                return index
        return None

    def next_visit(self, item, customer, now, runtime) -> Optional[datetime]:
        elapsed = (now - item.failed_at).total_seconds() / 3600.0
        for offset in self.OFFSETS_HOURS:
            if elapsed < offset:
                return item.failed_at + timedelta(hours=offset)
        if elapsed < self.OFFSETS_HOURS[-1] + 6:
            return now + timedelta(hours=2)
        return None


class AggressiveAgent(Agent):
    """Contact on every channel, every day, until the money arrives.

    Recovers more than doing nothing. Also generates the opt-outs, the
    complaints and, with the gate open, the regulatory violations. Included
    because "we tried harder" is the intuitive answer to a recovery problem and
    the results table is the argument against it.
    """

    name = "b2_aggressive"
    ROTATION = (
        ActionKind.PAYMENT_LINK_SMS,
        ActionKind.PAYMENT_LINK_WHATSAPP,
        ActionKind.PAYMENT_LINK_EMAIL,
        ActionKind.VOICE_CALL,
    )

    def decide(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        retry = ActionKind.MANDATE_DEBIT if item.method.is_mandate else ActionKind.SILENT_RETRY
        if item.attempts % 3 == 0 and not self.blocked(item, retry, now):
            if item.method.is_mandate and item.prenotified_at is None:
                return self.build(
                    ActionKind.PRENOTIFY, item, customer, now, runtime,
                    rationale="pre-debit notice",
                )
            return self.build(retry, item, customer, now, runtime, rationale="retry every cycle")
        for offset in range(len(self.ROTATION)):
            kind = self.ROTATION[(item.attempts + offset) % len(self.ROTATION)]
            if not self.blocked(item, kind, now):
                return self.build(
                    kind, item, customer, now, runtime, rationale="contact on rotation, every cycle"
                )
        return self.stop(item, "every contact channel is blocked")

    def next_visit(self, item, customer, now, runtime) -> Optional[datetime]:
        # Deliberately ignores contact hours. With the gate closed the policy
        # engine defers these into the permitted window; with the gate open they
        # go out at 3 AM and are counted as violations.
        return now + timedelta(hours=8)


class LLMOnlyAgent(Agent):
    """A model picks the next action from the full menu. No pricing, no gate.

    This is the honest version of the common agentic demo, and it is run with
    the policy engine disabled so that what it would have violated is measured.

    When no API key is present it runs a deterministic stand-in whose behaviour
    matches what these loops empirically do: reach for contact because contact
    feels like progress, escalate when a customer pushes back, and keep going
    because nothing in the loop prices the next message. The point of the
    baseline is the missing gate and the missing price, and that is reproduced
    faithfully without a key. Runs that used a real model are labelled as such
    in the results.
    """

    name = "b3_llm_only"
    handles_replies = True

    MENU = (
        ActionKind.SILENT_RETRY,
        ActionKind.MANDATE_DEBIT,
        ActionKind.PAYMENT_LINK_SMS,
        ActionKind.PAYMENT_LINK_WHATSAPP,
        ActionKind.PAYMENT_LINK_EMAIL,
        ActionKind.UPDATE_INSTRUMENT,
        ActionKind.INCENTIVE_OFFER,
        ActionKind.VOICE_CALL,
        ActionKind.HUMAN_ESCALATION,
    )

    SYSTEM = """You are a payment recovery agent. Given one failed payment, choose \
the single next action.

Reply with ONE JSON object: {"action": "<one of the listed actions>", \
"wait_hours": <number>, "why": "<short reason>"}. No prose."""

    def decide(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        backend = getattr(runtime.llm, "backend", None)
        if runtime.llm.enabled and backend is not None:
            chosen = self._ask_model(item, customer, now, runtime, backend)
            if chosen is not None:
                return chosen
        return self._heuristic(item, customer, now, runtime)

    def _ask_model(self, item, customer, now, runtime, backend) -> Optional[ProposedAction]:
        payload = {
            "method": item.method.value,
            "error_source": item.error_source.value,
            "error_reason": item.error_reason.value,
            "amount_rupees": item.outstanding_paise // 100,
            "attempts_so_far": item.attempts,
            "hours_since_failure": round((now - item.failed_at).total_seconds() / 3600.0, 1),
            "allowed_actions": [k.value for k in self.MENU],
        }
        try:
            raw = backend.complete_json(self.SYSTEM, json.dumps(payload), max_tokens=150)
        except Exception:  # noqa: BLE001
            return None
        from ..llm.gateway import _extract_json

        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return None
        try:
            kind = ActionKind(str(parsed.get("action", "")).strip().lower())
        except ValueError:
            return None
        if kind not in self.MENU:
            return None
        return self.build(
            kind, item, customer, now, runtime,
            rationale=str(parsed.get("why", "model choice"))[:120],
        )

    def _heuristic(self, item, customer, now, runtime) -> Optional[ProposedAction]:
        if item.attempts >= 6:
            return self.build(
                ActionKind.HUMAN_ESCALATION, item, customer, now, runtime,
                rationale="nothing has worked, hand to a human",
            )
        if customer.disputed:
            return self.build(
                ActionKind.HUMAN_ESCALATION, item, customer, now, runtime,
                rationale="customer disputed",
            )
        if item.method.is_mandate and item.attempts == 0:
            return self.build(
                ActionKind.PRENOTIFY, item, customer, now, runtime, rationale="notify before debit"
            )
        retry = ActionKind.MANDATE_DEBIT if item.method.is_mandate else ActionKind.SILENT_RETRY
        if item.attempts <= 2 and not self.blocked(item, retry, now):
            return self.build(retry, item, customer, now, runtime, rationale="try the charge again")
        rotation = (
            ActionKind.PAYMENT_LINK_SMS,
            ActionKind.PAYMENT_LINK_WHATSAPP,
            ActionKind.INCENTIVE_OFFER,
            ActionKind.VOICE_CALL,
        )
        for offset in range(len(rotation)):
            kind = rotation[(item.attempts + offset) % len(rotation)]
            if not self.blocked(item, kind, now):
                return self.build(
                    kind, item, customer, now, runtime, rationale="follow up with the customer"
                )
        return self.build(
            ActionKind.HUMAN_ESCALATION, item, customer, now, runtime,
            rationale="no channel left, hand to a human",
        )

    def next_visit(self, item, customer, now, runtime) -> Optional[datetime]:
        if item.attempts >= 8:
            return None
        return now + timedelta(hours=12)
