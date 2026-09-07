"""Action execution.

By the time anything reaches here it has already been priced by the underwriter
and approved by the policy engine. The executor's job is to perform the side
effect and report honestly what happened, including when it fails.

The ordering inside ``execute`` is deliberate and is the part worth reading:

1. Render the message from a registered template. If rendering fails, stop. No
   send happens.
2. Create the payment link through Razorpay. If the gateway is unavailable, stop
   and report ``retryable``. The caller reschedules; the item is not lost.
3. Only then deliver, and only then consult the world for the outcome.

Doing the gateway call before the delivery matters: a message containing a link
that does not exist is worse than no message, because the customer clicks it,
fails, and now believes the merchant is broken as well as owed money.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..llm.gateway import LLMGateway
from ..money import to_rupees
from ..policy.templates import TemplateError, get_template, template_for
from ..rng import crn_uniform
from ..world.entities import (
    ACTION_CHANNEL,
    ACTION_COST_PAISE,
    ActionKind,
    Channel,
    Customer,
    Method,
    RiskItem,
)
from ..world.generator import World
from ..world.outcomes import ActionOutcome, execute_counterfactual
from ..policy.engine import ProposedAction
from .razorpay_client import RazorpayClient, RazorpayError


@dataclass
class ExecutionResult:
    executed: bool
    action: ProposedAction
    outcome: Optional[ActionOutcome] = None
    message_text: Optional[str] = None
    provider_ref: Optional[str] = None
    cost_paise: int = 0
    error: Optional[str] = None
    retryable: bool = False
    delivery_failed: bool = False
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Ledger projection. Deliberately excludes wall-clock latency.

        ``latency_ms`` is measured with ``perf_counter`` and varies between runs
        on the same machine, let alone across machines. It was originally
        included here, which quietly broke the reproducibility guarantee: two
        runs of the same seed produced identical decisions, identical recovered
        amounts and identical message counts, but different chain heads. The
        determinism test caught it. Timing stays available on the object for
        metrics; it does not go into anything that is hashed.
        """
        return {
            "executed": self.executed,
            "provider_ref": self.provider_ref,
            "cost_paise": self.cost_paise,
            "error": self.error,
            "retryable": self.retryable,
            "delivery_failed": self.delivery_failed,
            "message": self.message_text,
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }


# Carrier level delivery failure. Independent of whether the customer engages:
# an SMS can simply not arrive. Charged for anyway, which is realistic and is
# part of why contact volume is expensive.
DELIVERY_FAILURE_RATE: dict[Channel, float] = {
    Channel.SMS: 0.035,
    Channel.WHATSAPP: 0.018,
    Channel.EMAIL: 0.055,
    Channel.VOICE: 0.14,
    Channel.NONE: 0.0,
    Channel.HUMAN: 0.0,
}


class Executor:
    def __init__(
        self,
        world: World,
        client: RazorpayClient,
        llm: LLMGateway,
        *,
        merchant_name: str = "Acme Retail",
    ) -> None:
        self.world = world
        self.client = client
        self.llm = llm
        self.merchant_name = merchant_name
        self.messages_sent = 0
        self.messages_failed = 0
        self.link_failures = 0

    def execute(self, action: ProposedAction, item: RiskItem, customer: Customer) -> ExecutionResult:
        started = time.perf_counter()
        now = action.scheduled_at

        if action.kind in (ActionKind.WAIT, ActionKind.STOP):
            return ExecutionResult(executed=False, action=action, notes=["no-op action"])

        result = ExecutionResult(executed=False, action=action)
        message: Optional[str] = None
        provider_ref: Optional[str] = None

        needs_link = action.kind in (
            ActionKind.PAYMENT_LINK_SMS,
            ActionKind.PAYMENT_LINK_WHATSAPP,
            ActionKind.PAYMENT_LINK_EMAIL,
            ActionKind.UPDATE_INSTRUMENT,
            ActionKind.INCENTIVE_OFFER,
            ActionKind.PRENOTIFY,
        )

        if needs_link:
            try:
                link = self.client.create_payment_link(
                    amount_paise=action.amount_paise,
                    description=f"Recovery for {item.order_id or item.item_id}",
                    reference_id=item.item_id,
                    idempotency_key=action.idempotency_key,
                    now=now,
                    expire_by=now + timedelta(days=7),
                )
                provider_ref = link.get("short_url") or link.get("id")
            except RazorpayError as exc:
                self.link_failures += 1
                result.error = f"payment link creation failed: {exc}"
                result.retryable = exc.retryable
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

            try:
                message = self._render(action, item, customer, provider_ref or "", now)
            except TemplateError as exc:
                # Should be unreachable: the policy engine already rendered this
                # action's template. Kept because "unreachable" and "cannot
                # happen" are different claims, and a send is not something to
                # attempt on the strength of the weaker one.
                result.error = f"template render failed at execution time: {exc}"
                result.retryable = False
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

        channel = ACTION_CHANNEL[action.kind]
        cost = ACTION_COST_PAISE.get(action.kind, 0)

        if channel in DELIVERY_FAILURE_RATE and DELIVERY_FAILURE_RATE[channel] > 0:
            roll = crn_uniform(self.world.seed, item.item_id, "delivery", action.kind.value, action.idempotency_key)
            if roll < DELIVERY_FAILURE_RATE[channel]:
                self.messages_failed += 1
                result.executed = True  # it was attempted and it was billed
                result.delivery_failed = True
                result.cost_paise = cost
                result.message_text = message
                result.provider_ref = provider_ref
                result.outcome = ActionOutcome()  # no engagement, no annoyance
                result.notes.append(f"{channel.value} delivery failed at the carrier")
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

        if action.kind.is_contact:
            self.messages_sent += 1

        outcome = execute_counterfactual(self.world, item, customer, action.kind, now)

        result.executed = True
        result.outcome = outcome
        result.message_text = message
        result.provider_ref = provider_ref
        result.cost_paise = cost
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    # -- message construction ----------------------------------------------

    def _render(
        self,
        action: ProposedAction,
        item: RiskItem,
        customer: Customer,
        link: str,
        now: datetime,
    ) -> str:
        """Render the already approved slots, substituting the real link.

        The slots were filled and frozen before the policy engine saw them, so
        what goes out is exactly what was validated. The single exception is
        ``link``, which cannot exist until the gateway has issued it. That is
        safe because the link is generated by us rather than by a model, and
        because ``Template.render`` re-applies every length and character check
        on the way out. Nothing else is re-derived here.
        """
        template = get_template(action.template_id or template_for("payment_link", customer.language))
        values = dict(action.slots)
        if "link" in template.slots:
            values["link"] = link[:MAX_SLOT_LEN]
        return template.render(values)


# Kept in sync with policy.templates.MAX_SLOT_LEN, imported rather than repeated.
from ..policy.templates import MAX_SLOT_LEN  # noqa: E402


PLACEHOLDER_LINK = "https://rzp.io/i/XXXXXXXX"
"""Stand-in used at proposal time.

Deliberately the same shape and length as a real Razorpay short link, so that a
message which passes template validation with the placeholder cannot fail length
validation once the real link is substituted.
"""


def build_slots(
    llm: LLMGateway,
    template_id: str,
    item: RiskItem,
    customer: Customer,
    amount_paise: int,
    now: datetime,
    merchant_name: str,
) -> dict[str, str]:
    """Produce the final slot values for a message, before it is gated.

    This runs at proposal time, not at send time, so the policy engine validates
    the exact text that will leave the system. The model may write only the
    slots that are genuinely editorial. Amount, link, date, invoice number and
    the MSMED day count are facts computed here and locked afterwards: letting a
    model paraphrase a rupee figure is precisely the failure this architecture
    exists to make impossible.
    """
    template = get_template(template_id)
    amount = f"{to_rupees(amount_paise):.2f}"
    debit_date = (now + timedelta(hours=26)).strftime("%d-%m-%Y")
    overdue_days = max(0, (now - (item.invoice_due_at or item.failed_at)).days)
    # MSMED Act section 15 caps an agreed credit period at 45 days, so the
    # statutory overrun is measured against the agreement and floored at zero.
    msme_days = max(0, overdue_days - min(item.agreement_days, 45))

    facts = {
        "amount": amount,
        "link": PLACEHOLDER_LINK,
        "date": debit_date,
        "invoice": (item.order_id or item.item_id)[:MAX_SLOT_LEN],
        "days": str(msme_days),
        "merchant": merchant_name[:MAX_SLOT_LEN],
        "discount": str(max(1, amount_paise // 100 // 20)),
    }
    defaults = {slot: facts.get(slot, "") for slot in template.slots}

    editorial = tuple(s for s in template.slots if s not in ("amount", "link", "date", "invoice", "days"))
    if editorial:
        filled = llm.fill_slots(
            template.template_id,
            editorial,
            context={
                "method": item.method.value,
                "reason": item.error_reason.value,
                "days_overdue": overdue_days,
            },
            language=customer.language,
            defaults={s: defaults[s] for s in editorial},
        )
        for slot in editorial:
            value = filled.get(slot)
            if isinstance(value, str) and value.strip():
                defaults[slot] = value.strip()[:MAX_SLOT_LEN]

    # Re-lock the facts unconditionally, after the model has had its turn.
    for slot in template.slots:
        if slot in ("amount", "link", "date", "invoice", "days"):
            defaults[slot] = facts[slot]
    return defaults
