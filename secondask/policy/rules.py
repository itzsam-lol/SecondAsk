"""The rule set.

Each rule is a pure function of ``(ProposedAction, PolicyContext)`` returning a
``RuleVerdict``. Each carries a ``citation`` naming the obligation it implements,
so the audit trail says *why* an action was refused in terms a compliance
reviewer can check, not just a rule number.

Citations are to the public position of each regulator as summarised in
COMPLIANCE.md. They are engineering-grade, not legal advice: the point is that
the constraint is machine-checked and traceable, not that this file is a legal
opinion.

Boundary conditions are stated explicitly everywhere a regulator specifies a
time. "Between 8 AM and 7 PM" is implemented as the half-open interval
``[08:00, 19:00)``, so 18:59:59 is permitted and 19:00:00 is not. Tests pin all
four boundary instants, because "between" is exactly the kind of word that
produces an off-by-one that only shows up in a regulator's sample.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from ..clock import ist_time_of_day, next_ist_time, to_ist
from ..world.entities import ACTION_CHANNEL, ActionKind, Channel, ItemState
from .engine import PolicyContext, ProposedAction, RuleVerdict
from .templates import TemplateError, get_template

# Contact window, IST. Half-open: [CONTACT_START, CONTACT_END).
CONTACT_START_HOUR = 8
CONTACT_END_HOUR = 19

# Voice is more intrusive than text, so it gets a narrower window.
VOICE_START_HOUR = 10
VOICE_END_HOUR = 18

MAX_CONTACTS_24H = 2
MAX_CONTACTS_7D = 4
MIN_CONTACT_GAP_HOURS = 6.0
MAX_ATTEMPTS_PER_ITEM = 6

# RBI e-mandate: the pre-debit notification must precede the debit by at least
# 24 hours. An upper bound is our own addition: a notice sent three weeks ago
# does not meaningfully inform anyone, so we treat it as stale and require a
# fresh one. Failing closed on staleness costs an SMS; failing open risks an
# unnotified debit.
PREDEBIT_MIN_HOURS = 24.0
PREDEBIT_MAX_HOURS = 24.0 * 7


def _rule(rule_id: str, citation: str = ""):
    def decorate(fn):
        fn.rule_id = rule_id
        fn.citation = citation
        return fn

    return decorate


def _ok(rule_id: str) -> RuleVerdict:
    return RuleVerdict(rule_id=rule_id, allowed=True)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@_rule("R-RBI-HOURS", "RBI Fair Practices Code / recovery agent conduct: no customer contact outside 08:00-19:00 local time, across calls, SMS and instant messaging.")
def rbi_contact_hours(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """No customer contact outside 08:00-19:00 IST.

    Applies to every contact channel, not just voice. A 10 PM automated SMS is a
    violation even though nobody's phone rang. Silent retries are exempt: no
    human is contacted, so the restriction does not apply.
    """
    rid = "R-RBI-HOURS"
    if not action.kind.is_contact:
        return _ok(rid)
    hour = ist_time_of_day(action.scheduled_at)
    if CONTACT_START_HOUR <= hour < CONTACT_END_HOUR:
        return _ok(rid)
    local = to_ist(action.scheduled_at)
    return RuleVerdict(
        rid,
        False,
        reason=f"contact at {local.strftime('%H:%M')} IST is outside the 08:00-19:00 window",
        retry_at=next_ist_time(action.scheduled_at, CONTACT_START_HOUR),
        citation=rbi_contact_hours.citation,
    )


@_rule("R-VOICE-WINDOW", "Internal control: voice is the most intrusive channel, so it is restricted more tightly than the regulatory minimum.")
def voice_window(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-VOICE-WINDOW"
    if action.kind != ActionKind.VOICE_CALL:
        return _ok(rid)
    hour = ist_time_of_day(action.scheduled_at)
    if VOICE_START_HOUR <= hour < VOICE_END_HOUR:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"voice call at {to_ist(action.scheduled_at).strftime('%H:%M')} IST is outside 10:00-18:00",
        retry_at=next_ist_time(action.scheduled_at, VOICE_START_HOUR),
        citation=voice_window.citation,
    )


@_rule("R-COOLDOWN", "Internal control: minimum interval between contacts to the same customer.")
def contact_cooldown(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-COOLDOWN"
    if not action.kind.is_contact or ctx.last_contact_at is None:
        return _ok(rid)
    gap = (action.scheduled_at - ctx.last_contact_at).total_seconds() / 3600.0
    if gap >= MIN_CONTACT_GAP_HOURS:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"only {gap:.1f}h since last contact, minimum is {MIN_CONTACT_GAP_HOURS:.0f}h",
        retry_at=ctx.last_contact_at + timedelta(hours=MIN_CONTACT_GAP_HOURS),
        citation=contact_cooldown.citation,
    )


# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------


@_rule("R-FREQ-24H", "TRAI TCCCPA preference framework and RBI conduct norms: contact frequency must be bounded.")
def frequency_cap_24h(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-FREQ-24H"
    if not action.kind.is_contact:
        return _ok(rid)
    if ctx.contacts_24h < MAX_CONTACTS_24H:
        return _ok(rid)
    retry = None
    if ctx.last_contact_at is not None:
        retry = ctx.last_contact_at + timedelta(hours=24)
    return RuleVerdict(
        rid,
        False,
        reason=f"{ctx.contacts_24h} contacts in the last 24h, cap is {MAX_CONTACTS_24H}",
        retry_at=retry,
        citation=frequency_cap_24h.citation,
    )


@_rule("R-FREQ-7D", "TRAI TCCCPA preference framework: weekly contact volume must be bounded.")
def frequency_cap_7d(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-FREQ-7D"
    if not action.kind.is_contact:
        return _ok(rid)
    if ctx.contacts_7d < MAX_CONTACTS_7D:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"{ctx.contacts_7d} contacts in the last 7 days, cap is {MAX_CONTACTS_7D}",
        citation=frequency_cap_7d.citation,
    )


@_rule("R-MAX-ATTEMPTS", "Internal control: bounded total attempts per item, so no item can be pursued indefinitely.")
def max_attempts(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-MAX-ATTEMPTS"
    if action.kind in (ActionKind.WAIT, ActionKind.STOP):
        return _ok(rid)
    if ctx.item.attempts < MAX_ATTEMPTS_PER_ITEM:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"item already has {ctx.item.attempts} attempts, cap is {MAX_ATTEMPTS_PER_ITEM}",
        citation=max_attempts.citation,
    )


# ---------------------------------------------------------------------------
# Consent and channel eligibility
# ---------------------------------------------------------------------------


@_rule("R-OPTOUT", "DPDP Act 2023 consent withdrawal, and TRAI preference registration: an opt-out is absolute.")
def respect_optout(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """An opt-out is permanent and channel-specific.

    Note this is checked against the *customer*, not the item. Somebody who says
    STOP about one failed payment has said STOP, full stop. Systems that scope
    opt-out to a single dunning sequence keep messaging them about the next one.
    """
    rid = "R-OPTOUT"
    channel = ACTION_CHANNEL[action.kind]
    if channel in (Channel.NONE, Channel.HUMAN):
        return _ok(rid)
    if channel in ctx.customer.opted_out_channels:
        return RuleVerdict(
            rid,
            False,
            reason=f"customer has opted out of {channel.value}",
            citation=respect_optout.citation,
        )
    return _ok(rid)


@_rule("R-CHANNEL-EXISTS", "Internal control: do not attempt a channel the customer has no address for.")
def channel_reachable(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-CHANNEL-EXISTS"
    channel = ACTION_CHANNEL[action.kind]
    if channel in (Channel.NONE, Channel.HUMAN):
        return _ok(rid)
    if ctx.customer.channel_available(channel):
        return _ok(rid)
    return RuleVerdict(
        rid, False, reason=f"no usable {channel.value} address for this customer",
        citation=channel_reachable.citation,
    )


@_rule("R-DLT-CONSENT", "TRAI TCCCPA: promotional content requires prior consent and DND scrubbing; transactional content does not.")
def promotional_consent(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """Promotional sends need consent and must respect DND.

    Classification comes from the template, so an agent cannot relabel a
    discount offer as transactional to slip past this.
    """
    rid = "R-DLT-CONSENT"
    if action.template_id is None:
        return _ok(rid)
    template = get_template(action.template_id)  # raises -> engine fails closed
    if not template.promotional:
        return _ok(rid)
    if not ctx.consent_promotional:
        return RuleVerdict(
            rid, False, reason=f"template {action.template_id} is promotional and no consent is on record",
            citation=promotional_consent.citation,
        )
    if ctx.dnd_registered:
        return RuleVerdict(
            rid, False, reason="customer is on the DND register; promotional content is barred",
            citation=promotional_consent.citation,
        )
    return _ok(rid)


@_rule("R-DLT-TEMPLATE", "TRAI TCCCPA: commercial messages must match a template registered on a DLT platform against a registered header.")
def registered_template(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """Every message must render cleanly from a registered template.

    This rule is the reason the language model cannot send free text. It does
    not merely check that a template id was supplied: it performs the render, so
    a malformed slot value, an over-long substitution or an injected newline is
    caught here and denied rather than transmitted.
    """
    rid = "R-DLT-TEMPLATE"
    channel = ACTION_CHANNEL[action.kind]
    if channel not in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL):
        return _ok(rid)
    if not action.template_id:
        return RuleVerdict(
            rid, False, reason="message action carries no template id; free-text sending is not permitted",
            citation=registered_template.citation,
        )
    try:
        template = get_template(action.template_id)
        template.render(action.slots)
    except TemplateError as exc:
        return RuleVerdict(rid, False, reason=str(exc), citation=registered_template.citation)
    return _ok(rid)


# ---------------------------------------------------------------------------
# Mandates
# ---------------------------------------------------------------------------


@_rule("R-EMANDATE-PRENOTIFY", "RBI e-mandate framework: a pre-debit notification must reach the customer at least 24 hours before a recurring debit.")
def emandate_prenotification(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """No mandate debit without a valid pre-debit notice.

    Both bounds are enforced. Too recent (under 24h) is a regulatory breach; too
    old (over 7 days) is treated as stale and requires a fresh notice, which is
    our own stricter reading.
    """
    rid = "R-EMANDATE-PRENOTIFY"
    if action.kind != ActionKind.MANDATE_DEBIT:
        return _ok(rid)
    if not ctx.item.method.is_mandate:
        return _ok(rid)
    sent = ctx.item.prenotified_at
    if sent is None:
        return RuleVerdict(
            rid, False, reason="no pre-debit notification has been sent for this mandate",
            citation=emandate_prenotification.citation,
        )
    hours = (action.scheduled_at - sent).total_seconds() / 3600.0
    if hours < PREDEBIT_MIN_HOURS:
        return RuleVerdict(
            rid,
            False,
            reason=f"pre-debit notice was sent {hours:.1f}h ago; the minimum is {PREDEBIT_MIN_HOURS:.0f}h",
            retry_at=sent + timedelta(hours=PREDEBIT_MIN_HOURS),
            citation=emandate_prenotification.citation,
        )
    if hours > PREDEBIT_MAX_HOURS:
        return RuleVerdict(
            rid,
            False,
            reason=f"pre-debit notice is {hours / 24:.1f} days old and is stale; send a fresh one",
            citation=emandate_prenotification.citation,
        )
    return _ok(rid)


@_rule("R-SILENT-RETRY-RAIL", "Internal control: only mandate rails can be debited without the payer present.")
def silent_retry_rail(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """A one-off UPI or card payment cannot be silently re-attempted.

    This looks like a technical detail and is actually a consent boundary: a
    customer who abandoned a checkout has not authorised anybody to charge them
    later. Systems that model "retry" uniformly across rails are describing an
    action they are not entitled to take.
    """
    rid = "R-SILENT-RETRY-RAIL"
    if action.kind not in (ActionKind.SILENT_RETRY, ActionKind.MANDATE_DEBIT):
        return _ok(rid)
    if ctx.item.method.supports_silent_retry:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"{ctx.item.method.value} has no standing authorisation; re-attempting requires the payer",
        citation=silent_retry_rail.citation,
    )


# ---------------------------------------------------------------------------
# Stopping rules
# ---------------------------------------------------------------------------


@_rule("R-STOP-SETTLED", "Internal control: stop on settlement. Contacting a customer who has already paid is the fastest route to a complaint.")
def stop_when_settled(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-STOP-SETTLED"
    if action.kind in (ActionKind.WAIT, ActionKind.STOP):
        return _ok(rid)
    if ctx.item.outstanding_paise <= 0 or ctx.item.state == ItemState.RECOVERED:
        return RuleVerdict(
            rid, False, reason="nothing outstanding on this item", citation=stop_when_settled.citation
        )
    if ctx.item.state.is_terminal:
        return RuleVerdict(
            rid,
            False,
            reason=f"item is in terminal state {ctx.item.state.value}",
            citation=stop_when_settled.citation,
        )
    return _ok(rid)


@_rule("R-STOP-DISPUTE", "RBI Fair Practices Code: a disputed amount must go to a human process, not an automated collection sequence.")
def stop_on_dispute(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """Once disputed, only a human may act.

    Escalation is explicitly still permitted. The rule routes the case, it does
    not abandon it.
    """
    rid = "R-STOP-DISPUTE"
    if not ctx.customer.disputed:
        return _ok(rid)
    if action.kind in (ActionKind.HUMAN_ESCALATION, ActionKind.WAIT, ActionKind.STOP):
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason="customer has disputed; only human escalation is permitted",
        citation=stop_on_dispute.citation,
    )


@_rule("R-STOP-HARDSHIP", "RBI Fair Practices Code: declared hardship requires forbearance, not intensified collection.")
def stop_on_hardship(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-STOP-HARDSHIP"
    if not ctx.customer.hardship:
        return _ok(rid)
    if action.kind in (ActionKind.HUMAN_ESCALATION, ActionKind.WAIT, ActionKind.STOP):
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason="customer has declared hardship; automated collection is suspended",
        citation=stop_on_hardship.citation,
    )


@_rule("R-HONOUR-PROMISE", "Internal control: a promise to pay must be honoured before chasing again.")
def honour_promise_to_pay(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """Do not chase before a promised date.

    Cheap to implement, and it removes the single most infuriating dunning
    behaviour: being asked again for money you have already committed a date
    for. Silent retries are exempt because they cost the customer nothing and
    they are not a contact.
    """
    rid = "R-HONOUR-PROMISE"
    promised = ctx.item.promise_to_pay_at
    if promised is None or not action.kind.is_contact:
        return _ok(rid)
    if action.scheduled_at >= promised:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"customer promised to pay by {promised.date()}; do not chase before then",
        retry_at=promised,
        citation=honour_promise_to_pay.citation,
    )


# ---------------------------------------------------------------------------
# Money and integrity
# ---------------------------------------------------------------------------


@_rule("R-AMOUNT-BOUND", "Internal control: a money-moving action's amount must equal the ledger's outstanding balance.")
def amount_matches_ledger(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """The amount is taken from the ledger, never from a model.

    This is the specific control that makes a hallucinated or injected amount
    unable to move money. Even if every other defence failed and something
    proposed a debit of a different figure, it dies here. Also rejects
    non-positive amounts, which are otherwise a route to a negative-value
    "refund" through a collection channel.
    """
    rid = "R-AMOUNT-BOUND"
    if action.kind in (ActionKind.WAIT, ActionKind.STOP, ActionKind.HUMAN_ESCALATION):
        return _ok(rid)
    if action.amount_paise <= 0:
        return RuleVerdict(
            rid, False, reason=f"non-positive amount {action.amount_paise}",
            citation=amount_matches_ledger.citation,
        )
    if action.amount_paise != ctx.item.outstanding_paise:
        return RuleVerdict(
            rid,
            False,
            reason=(
                f"amount {action.amount_paise} paise does not match the outstanding balance "
                f"{ctx.item.outstanding_paise} paise"
            ),
            citation=amount_matches_ledger.citation,
        )
    return _ok(rid)


@_rule("R-IDEMPOTENCY", "Internal control: every side-effecting action carries a key that may be used once.")
def idempotency(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """Replaying a key is refused.

    Guards against duplicate webhook delivery and against a retry loop that
    re-sends after a timeout where the first call actually succeeded.
    """
    rid = "R-IDEMPOTENCY"
    if action.kind in (ActionKind.WAIT, ActionKind.STOP):
        return _ok(rid)
    if not action.idempotency_key:
        return RuleVerdict(rid, False, reason="missing idempotency key", citation=idempotency.citation)
    if action.idempotency_key in ctx.used_idempotency_keys:
        return RuleVerdict(
            rid,
            False,
            reason=f"idempotency key {action.idempotency_key} has already been used",
            citation=idempotency.citation,
        )
    return _ok(rid)


@_rule("R-SPEND-CAP", "Internal control: a hard ceiling on recovery spend, so a bug cannot spend without bound.")
def spend_cap(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    rid = "R-SPEND-CAP"
    if ctx.spend_cap_paise <= 0:
        return _ok(rid)
    from ..world.entities import ACTION_COST_PAISE

    cost = ACTION_COST_PAISE.get(action.kind, 0)
    if ctx.spend_paise + cost <= ctx.spend_cap_paise:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=f"spend cap reached ({ctx.spend_paise} + {cost} > {ctx.spend_cap_paise} paise)",
        citation=spend_cap.citation,
    )


@_rule("R-ESCALATE-ONCE", "Internal control: an item handed to a human leaves the automated loop and does not come back to it.")
def escalate_once(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """One escalation per item.

    Without this the planner discovers that a human is effective and escalates
    the same case repeatedly, which in the real world means several people
    working the same account and calling the same customer. Escalation is a
    handover, not a retry.
    """
    rid = "R-ESCALATE-ONCE"
    if action.kind != ActionKind.HUMAN_ESCALATION:
        return _ok(rid)
    if ctx.item.escalated_at is None:
        return _ok(rid)
    return RuleVerdict(
        rid, False, reason="this item has already been handed to a human",
        citation=escalate_once.citation,
    )


@_rule("R-ESCALATION-CAPACITY", "Internal control: human review is a fixed daily capacity, not a resource that scales with the size of the backlog.")
def escalation_capacity(action: ProposedAction, ctx: PolicyContext) -> RuleVerdict:
    """A daily cap on how many cases a human team can absorb.

    This is the constraint that stops expected value planning from concluding
    that everything should go to a person. Escalation is genuinely the most
    effective action for a disputed or high value case, so a planner that prices
    it without a capacity limit will choose it for the entire batch. Pricing
    scarce capacity as though it were elastic is how an optimiser produces a
    plan that cannot be executed.

    Denials carry tomorrow morning as a retry time, so a case that misses
    today's capacity queues rather than being dropped.
    """
    rid = "R-ESCALATION-CAPACITY"
    if action.kind != ActionKind.HUMAN_ESCALATION:
        return _ok(rid)
    if ctx.escalation_daily_cap <= 0:
        return _ok(rid)
    if ctx.escalations_today < ctx.escalation_daily_cap:
        return _ok(rid)
    return RuleVerdict(
        rid,
        False,
        reason=(
            f"human review capacity for today is used up "
            f"({ctx.escalations_today}/{ctx.escalation_daily_cap})"
        ),
        retry_at=next_ist_time(action.scheduled_at, CONTACT_START_HOUR),
        citation=escalation_capacity.citation,
    )


DEFAULT_RULES = [
    rbi_contact_hours,
    voice_window,
    contact_cooldown,
    frequency_cap_24h,
    frequency_cap_7d,
    max_attempts,
    respect_optout,
    channel_reachable,
    promotional_consent,
    registered_template,
    emandate_prenotification,
    silent_retry_rail,
    stop_when_settled,
    stop_on_dispute,
    stop_on_hardship,
    honour_promise_to_pay,
    escalate_once,
    escalation_capacity,
    amount_matches_ledger,
    idempotency,
    spend_cap,
]

RULE_CITATIONS = {getattr(r, "rule_id"): getattr(r, "citation", "") for r in DEFAULT_RULES}
