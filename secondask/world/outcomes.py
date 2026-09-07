"""The counterfactual response model.

This answers the one question that makes offline measurement possible: *if this
action were taken on this item at this time, would the money arrive?*

In production you cannot answer that. You observe the branch you took and never
the branch you didn't, so "we recovered X" is uninterpretable without a holdout.
Here the world knows the latent blocker, so every counterfactual is available,
and every agent is scored against the same ground truth.

Three properties keep this honest:

1. **The agent never reads any of this.** Success probabilities are computed
   inside the simulator from latent state. Agents see only outcomes.

2. **Common random numbers.** The draw is keyed on
   ``(seed, item, action_kind, hour_bucket)``, so if two agents take the same
   action on the same item in the same hour, they get the same coin. Differences
   between agents are attributable to decisions, not luck.

3. **The probabilities are shaped by mechanism, not by convenience.** Each
   blocker responds to the intervention that actually addresses it. Nothing here
   is conditioned on which agent is running.

The structural facts this encodes, which are what the whole thesis rests on:

* ``INSTRUMENT_DEAD`` has a **hard zero** for every retry. Not "low". Zero. You
  cannot retry your way past an expired card, and any number of attempts is
  exactly as useless as one.
* ``LIQUIDITY`` is governed by the salary calendar, not by persistence. Before
  payday nothing works; after payday it works and then decays again as the money
  is spent.
* ``TRANSIENT_INFRA`` resolves without anyone being contacted, so a free silent
  retry beats a paid message.
* ``DISPUTE`` gets *worse* with automated contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..clock import hour_bucket, ist_time_of_day
from ..rng import crn_uniform
from .entities import (
    ACTION_CHANNEL,
    ActionKind,
    Blocker,
    Channel,
    Customer,
    ReplyIntent,
    RiskItem,
)
from .generator import World

# Salience multiplier by channel. WhatsApp beats SMS, email is largely ignored,
# a voice call is hard to miss and correspondingly expensive and irritating.
CHANNEL_LIFT: dict[Channel, float] = {
    Channel.NONE: 1.0,
    Channel.SMS: 1.00,
    Channel.WHATSAPP: 1.25,
    Channel.EMAIL: 0.55,
    Channel.VOICE: 1.60,
    Channel.HUMAN: 1.80,
}

# Annoyance added per contact, before modifiers.
ANNOYANCE_DELTA: dict[Channel, float] = {
    Channel.NONE: 0.0,
    Channel.SMS: 0.50,
    Channel.WHATSAPP: 0.58,
    Channel.EMAIL: 0.18,
    Channel.VOICE: 1.50,
    Channel.HUMAN: 0.35,
}

# Time constants for decay, in hours.
TAU_AUTH_FRICTION = 9.0
TAU_INTENT_LOST = 74.0
TAU_POST_PAYDAY = 120.0  # salary gets spent; the recovery window closes


@dataclass
class ActionOutcome:
    """Everything that happened as a result of one executed action."""

    success: bool = False
    recovered_paise: int = 0
    partial: bool = False
    annoyance_delta: float = 0.0
    opted_out: Optional[Channel] = None
    complained: bool = False
    churned: bool = False
    reply_text: Optional[str] = None
    reply_intent_latent: ReplyIntent = ReplyIntent.NONE
    promise_date: Optional[datetime] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "recovered_paise": self.recovered_paise,
            "partial": self.partial,
            "annoyance_delta": round(self.annoyance_delta, 3),
            "opted_out": self.opted_out.value if self.opted_out else None,
            "complained": self.complained,
            "reply_intent_latent": self.reply_intent_latent.value,
        }


def _fatigue(customer: Customer) -> float:
    """Diminishing returns on contact.

    An exponential in accumulated annoyance. The fifth message in a week is worth
    a fraction of the first, which is the mechanism that makes "blast everything"
    lose money rather than merely spend it.
    """
    return math.exp(-0.42 * max(0.0, customer.annoyance))


def _time_of_day_lift(ts: datetime) -> float:
    """Response is better late morning and early evening.

    Two Gaussian bumps at 11:00 and 18:30 IST. Both sit inside the RBI 08:00-19:00
    contact window, so this pushes toward compliant hours anyway, but the model
    does not know that. The constraint is enforced separately and absolutely.
    """
    hour = ist_time_of_day(ts)
    morning = math.exp(-((hour - 11.0) ** 2) / 8.0)
    evening = math.exp(-((hour - 18.5) ** 2) / 4.0)
    return 0.62 + 0.38 * max(morning, evening)


def _decay(hours: float, tau: float) -> float:
    if hours <= 0:
        return 1.0
    return math.exp(-hours / tau)


def _hours_between(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 3600.0


def success_probability(
    world: World,
    item: RiskItem,
    customer: Customer,
    action: ActionKind,
    ts: datetime,
) -> float:
    """p(money arrives | action taken on item at ts). Simulator-internal."""
    if action in (ActionKind.WAIT, ActionKind.STOP):
        return 0.0

    blocker = item._blocker
    channel = ACTION_CHANNEL[action]
    is_retry = action in (ActionKind.SILENT_RETRY, ActionKind.MANDATE_DEBIT)
    resolves = item._resolves_at

    # A customer who has opted out of the channel cannot be reached on it, and a
    # churned customer is gone. Both are enforced by policy too; this is the
    # world's own consistency check.
    if channel in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL, Channel.VOICE):
        if not customer.channel_available(channel):
            return 0.0
        if customer.churned:
            return 0.0

    fatigue = _fatigue(customer)
    lift = CHANNEL_LIFT[channel] * _time_of_day_lift(ts)
    responsiveness = customer._responsiveness

    if action == ActionKind.HUMAN_ESCALATION:
        # A human can resolve a dispute, arrange a plan, or fix a dead
        # instrument. Expensive, and correspondingly effective.
        base = {
            Blocker.DISPUTE: 0.58,
            Blocker.INSTRUMENT_DEAD: 0.44,
            Blocker.LIQUIDITY: 0.30,
            Blocker.INTENT_LOST: 0.22,
            Blocker.TRANSIENT_INFRA: 0.50,
            Blocker.UNREACHABLE: 0.05,
        }[blocker]
        return min(0.95, base * (0.6 + 0.4 * responsiveness))

    if action == ActionKind.PRENOTIFY:
        # A pre-debit notice does not itself collect money. Its value is legal
        # (it is required before a mandate debit) plus a small real effect: some
        # customers top up the account when told a debit is coming.
        return 0.0

    if blocker == Blocker.UNREACHABLE:
        return 0.02 if is_retry else 0.008

    if blocker == Blocker.INSTRUMENT_DEAD:
        if is_retry:
            return 0.0  # hard zero, by construction
        if action == ActionKind.UPDATE_INSTRUMENT:
            return min(0.92, 0.34 * responsiveness * fatigue * lift * 2.0)
        if action == ActionKind.INCENTIVE_OFFER:
            return min(0.92, 0.22 * responsiveness * fatigue * lift * 2.0)
        return min(0.92, 0.20 * responsiveness * fatigue * lift * 2.0)

    if blocker == Blocker.DISPUTE:
        # Automated contact does not collect from a disputing customer. It
        # produces complaints. Only escalation works, handled above.
        return 0.02 if not is_retry else 0.01

    if blocker == Blocker.TRANSIENT_INFRA:
        resolved = resolves is not None and ts >= resolves
        if is_retry:
            return 0.86 if resolved else 0.04
        if resolved:
            return min(0.92, 0.55 * responsiveness * fatigue * lift * 1.6)
        return min(0.92, 0.10 * responsiveness * fatigue * lift)

    if blocker == Blocker.AUTH_FRICTION:
        if is_retry:
            return 0.05
        age = _hours_between(ts, item.failed_at)
        base = 0.62 * _decay(age, TAU_AUTH_FRICTION)
        return min(0.92, base * (0.45 + 0.55 * responsiveness) * fatigue * lift)

    if blocker == Blocker.LIQUIDITY:
        if resolves is None:
            return 0.012 if is_retry else 0.02
        if ts < resolves:
            # Before payday. A retry cannot work. A message is not useless,
            # since it can secure a promise to pay, but it does not collect today.
            return 0.03 if is_retry else min(0.92, 0.06 * responsiveness * fatigue * lift)
        since_payday = _hours_between(ts, resolves)
        window = _decay(since_payday, TAU_POST_PAYDAY)
        if is_retry:
            return min(0.95, 0.72 * window)
        return min(0.92, 0.55 * window * (0.45 + 0.55 * responsiveness) * fatigue * lift)

    # INTENT_LOST
    age = _hours_between(ts, item.failed_at)
    decay = _decay(age, TAU_INTENT_LOST)
    if is_retry:
        return 0.02
    base = 0.30 if action == ActionKind.INCENTIVE_OFFER else 0.12
    return min(0.92, base * decay * (0.4 + 0.6 * responsiveness) * fatigue * lift)


def _annoyance_for(action: ActionKind, item: RiskItem, customer: Customer, ts: datetime) -> float:
    channel = ACTION_CHANNEL[action]
    delta = ANNOYANCE_DELTA[channel]
    if delta == 0.0:
        return 0.0
    # Contacting a disputing customer with an automated message is the single
    # most inflammatory thing this system can do.
    if item._blocker == Blocker.DISPUTE:
        delta *= 3.0
    # Rapid repeat contact compounds.
    if item.last_contact_at is not None:
        gap_hours = _hours_between(ts, item.last_contact_at)
        if gap_hours < 24:
            delta *= 1.0 + (24.0 - max(0.0, gap_hours)) / 24.0
    # Being asked for money you have already promised is irritating.
    if item.promise_to_pay_at is not None and ts < item.promise_to_pay_at:
        delta *= 1.8
    return delta


# Reply corpus. Deliberately multilingual and messy, because that is what the
# parser has to survive. The last group is adversarial: see llm/injection.py.
_REPLY_BANK: dict[ReplyIntent, list[str]] = {
    ReplyIntent.PROMISE_TO_PAY: [
        "salary aane ke baad kar dunga, 3 tarikh tak",
        "will pay by 5th, please wait",
        "abhi paisa nahi hai, next week pakka",
        "Kindly give me time till month end. I will clear it.",
        "2 din me ho jayega",
        "paycheck comes on the 1st, I'll do it then",
    ],
    ReplyIntent.ALREADY_PAID: [
        "already paid this yesterday, check your records",
        "maine payment kar diya hai, screenshot bhej raha hoon",
        "This was settled last week. Please verify.",
        "paid via UPI ref 4471829",
    ],
    ReplyIntent.DISPUTE: [
        "I never ordered this. Remove the charge.",
        "ye galat hai, maine cancel kiya tha",
        "The service was never delivered. I am not paying.",
        "raising a complaint with my bank",
    ],
    ReplyIntent.OPT_OUT: [
        "STOP",
        "stop messaging me",
        "band karo ye message bhejna",
        "unsubscribe",
        "do not contact me again",
    ],
    ReplyIntent.WRONG_NUMBER: [
        "wrong number bhai",
        "you have the wrong person",
        "ye number kisi aur ka hai",
    ],
    ReplyIntent.HARDSHIP: [
        "I lost my job last month, please give me some time",
        "medical emergency chal raha hai, abhi possible nahi",
        "hospital me hoon, baad me dekhta hoon",
    ],
    ReplyIntent.NEEDS_HELP: [
        "card expire ho gaya hai, kaise update karun",
        "how do I change my card?",
        "link kaam nahi kar raha",
        "payment page not opening",
    ],
    ReplyIntent.UNINTELLIGIBLE: [
        "??",
        "k",
        "...",
        "hmm",
        "\U0001f937",
    ],
}


def _reply_for(
    world: World,
    item: RiskItem,
    customer: Customer,
    action: ActionKind,
    ts: datetime,
    success: bool,
) -> tuple[Optional[str], ReplyIntent]:
    """Decide whether the customer writes back, and what they say."""
    channel = ACTION_CHANNEL[action]
    if channel not in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL):
        return None, ReplyIntent.NONE
    if success:
        return None, ReplyIntent.NONE

    u = crn_uniform(world.seed, item.item_id, "reply", hour_bucket(ts), action.value)
    reply_rate = 0.22 * (0.4 + 0.6 * customer._responsiveness)
    if channel == Channel.WHATSAPP:
        reply_rate *= 1.5
    elif channel == Channel.EMAIL:
        reply_rate *= 0.4
    if u >= reply_rate:
        return None, ReplyIntent.NONE

    # Which intent, given the latent blocker and the customer's state.
    if customer.annoyance > customer._annoyance_tolerance:
        intent = ReplyIntent.OPT_OUT
    elif item._blocker == Blocker.DISPUTE:
        intent = ReplyIntent.DISPUTE
    elif item._blocker == Blocker.UNREACHABLE:
        intent = ReplyIntent.WRONG_NUMBER
    elif item._blocker == Blocker.INSTRUMENT_DEAD:
        intent = ReplyIntent.NEEDS_HELP
    elif item._blocker == Blocker.LIQUIDITY:
        # Genuine hardship in the subset that never becomes liquid.
        intent = (
            ReplyIntent.HARDSHIP if item._resolves_at is None else ReplyIntent.PROMISE_TO_PAY
        )
    else:
        pick = crn_uniform(world.seed, item.item_id, "reply_kind", hour_bucket(ts))
        if pick < 0.35:
            intent = ReplyIntent.PROMISE_TO_PAY
        elif pick < 0.5:
            intent = ReplyIntent.ALREADY_PAID
        elif pick < 0.7:
            intent = ReplyIntent.NEEDS_HELP
        else:
            intent = ReplyIntent.UNINTELLIGIBLE

    corpus = _REPLY_BANK[intent]
    idx = int(crn_uniform(world.seed, item.item_id, "reply_text", intent.value) * len(corpus))
    text = corpus[min(idx, len(corpus) - 1)]
    return text, intent


def execute_counterfactual(
    world: World,
    item: RiskItem,
    customer: Customer,
    action: ActionKind,
    ts: datetime,
) -> ActionOutcome:
    """Run one action against the world and return everything that happened.

    Mutates neither item nor customer, the caller applies the outcome. Keeping
    this function free of side effects is what allows the same code path to be
    used for the exploration runs that train the underwriter.
    """
    outcome = ActionOutcome()

    if action in (ActionKind.WAIT, ActionKind.STOP):
        return outcome

    p = success_probability(world, item, customer, action, ts)
    u = crn_uniform(world.seed, item.item_id, action.value, hour_bucket(ts))
    outcome.success = u < p

    if outcome.success:
        # Partial settlement happens mostly on invoices, where a buyer pays what
        # they have agreed and disputes the rest.
        partial_u = crn_uniform(world.seed, item.item_id, "partial", hour_bucket(ts))
        partial_rate = 0.18 if item.method.value == "invoice" else 0.03
        if partial_u < partial_rate:
            fraction = 0.35 + 0.5 * crn_uniform(world.seed, item.item_id, "partial_frac")
            outcome.recovered_paise = int(item.outstanding_paise * fraction)
            outcome.partial = True
        else:
            outcome.recovered_paise = item.outstanding_paise
        # Guard: never report recovering more than is owed, and never a negative
        # amount. Both are reachable through rounding on a partial payment.
        outcome.recovered_paise = max(0, min(outcome.recovered_paise, item.outstanding_paise))

    outcome.annoyance_delta = _annoyance_for(action, item, customer, ts)

    projected = customer.annoyance + outcome.annoyance_delta
    if projected > customer._annoyance_tolerance and action.is_contact:
        channel = ACTION_CHANNEL[action]
        over = projected - customer._annoyance_tolerance
        opt_u = crn_uniform(world.seed, customer.customer_id, "optout", hour_bucket(ts))
        if opt_u < min(0.85, 0.30 * over):
            outcome.opted_out = channel
        comp_u = crn_uniform(world.seed, customer.customer_id, "complaint", hour_bucket(ts))
        complaint_p = 0.06 * over
        if item._blocker == Blocker.DISPUTE:
            complaint_p *= 4.0
        if comp_u < min(0.7, complaint_p):
            outcome.complained = True
        churn_u = crn_uniform(world.seed, customer.customer_id, "churn", hour_bucket(ts))
        if churn_u < min(0.5, 0.08 * over):
            outcome.churned = True

    if not outcome.success:
        text, intent = _reply_for(world, item, customer, action, ts, outcome.success)
        outcome.reply_text = text
        outcome.reply_intent_latent = intent
        if intent == ReplyIntent.PROMISE_TO_PAY and item._resolves_at is not None:
            outcome.promise_date = item._resolves_at

    return outcome
