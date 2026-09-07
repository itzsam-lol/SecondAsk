"""Expected value planning.

The whole thesis of the project is in this file. Most recovery systems answer
"what is the next step in the sequence". This one answers a different question:

    over every action I could take and every time I could take it, which single
    (action, time) pair maximises expected recovered rupees net of what it costs
    to make the attempt, and is that number greater than zero?

If the answer to the last part is no, the correct action is to stop. That is
where the stopping rule comes from. It is not a fixed attempt count, it is the
point at which pursuing the money is worth less than leaving it alone.

Three terms in the valuation, and the second is the one that is usually missing:

``gross``
    ``p_recover * outstanding``. What the attempt is worth if it works.

``goodwill``
    Contact consumes a finite resource. Every message raises the probability
    that this customer opts out, complains, or leaves, and those outcomes cost
    real money on every future interaction, not just this one. Systems that
    price only the SMS at 18 paise conclude that messaging is nearly free and
    then send twenty of them. Pricing goodwill is what makes the agent send
    fewer messages and recover more, and it is the single change that most
    moves the results table.

``delay``
    Money recovered on day 14 is worth less than the same money on day 1:
    working capital has a cost, and the longer an item is open the more likely
    it ages out entirely. A mild exponential discount, which is what stops the
    planner parking every liquidity case on a payday three weeks away.

The candidate time grid is deliberately not uniform. It contains the moments
that actually matter: the instant a bank outage closes, the next few salary
credit dates, and otherwise the two times of day when people respond. Searching
a uniform hourly grid over three weeks would cost five hundred model evaluations
per decision to find the same answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from ..clock import days_in_month, to_ist, to_utc
from ..world.downtime import DowntimeFeed
from ..world.entities import (
    ACTION_CHANNEL,
    ACTION_COST_PAISE,
    ActionKind,
    Channel,
    Customer,
    Method,
    RiskItem,
)
from ..world.outcomes import ANNOYANCE_DELTA
from .model import Underwriter

# Price of one unit of accumulated annoyance, in paise.
#
# Derivation, stated so the number can be argued with rather than taken on
# faith: at the tolerance boundary each additional unit of annoyance raises the
# probability of an opt-out by roughly 0.3 and of outright churn by roughly 0.08.
# Against a conservative annual contribution of about 1,200 rupees for a
# repeat customer, 0.08 x 120000 paise is around 9,600 paise of expected
# lifetime value destroyed, plus the option value of the channels an opt-out
# closes. 4,000 paise per unit is deliberately below that estimate, so the agent
# is if anything too willing to contact rather than too reluctant.
#
# The sensitivity of the results to this number is reported in the evaluation,
# because a headline that only holds at one arbitrary constant is not a result.
ANNOYANCE_PRICE_PAISE = 4000

# Per-customer goodwill budget, as a hard constraint on top of the soft price.
#
# It scales with the amount at stake, because a flat budget is wrong in both
# directions at once: it tolerates too much friction over a 200 rupee wallet
# top-up and gives up far too early on a 1.5 lakh receivable. Nobody would run a
# collections process that treats those identically, and an earlier version of
# this planner did, which cost it most of the invoice value in the batch.
#
# The growth is logarithmic and hard-capped. That cap is the ethical line rather
# than an economic one: past a certain point the answer to "this customer is not
# responding" is a human conversation or a write-off, not more messages, however
# large the invoice. No amount of money buys unlimited contact.
ANNOYANCE_BUDGET = 2.6
ANNOYANCE_BUDGET_CAP = 4.6
ANNOYANCE_BUDGET_PIVOT_RUPEES = 500.0

# Working capital discount. exp(-days / TAU).
TAU_DELAY_DAYS = 34.0

# Opportunity cost of consuming one slot of scarce human review capacity, in
# paise. Properly this is the shadow price of the capacity constraint, which
# would mean solving the dual of the allocation problem across the whole batch.
# A fixed premium is a deliberate approximation: the hard cap in
# R-ESCALATION-CAPACITY is what actually enforces the constraint, and this
# number only has to be large enough that the planner reaches for a person when
# the case genuinely warrants one rather than by default. Set to roughly the
# gross value of a mid-sized item, so escalation has to clearly beat messaging
# rather than merely tie with it.
ESCALATION_SCARCITY_PREMIUM_PAISE = 40000

PLANNING_HORIZON_DAYS = 14
CONTACT_HOURS_IST = (11, 18)


@dataclass
class Candidate:
    action: ActionKind
    at: datetime
    p_recover: float
    gross_paise: int
    cost_paise: int
    goodwill_paise: int
    discount: float
    ev_paise: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        from ..clock import iso

        return {
            "action": self.action.value,
            "at": iso(self.at),
            "p_recover": round(self.p_recover, 4),
            "gross_paise": self.gross_paise,
            "cost_paise": self.cost_paise,
            "goodwill_paise": self.goodwill_paise,
            "discount": round(self.discount, 4),
            "ev_paise": self.ev_paise,
            "note": self.note,
        }


@dataclass
class Plan:
    best: Optional[Candidate]
    alternatives: list[Candidate] = field(default_factory=list)
    considered: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.to_dict() if self.best else None,
            "alternatives": [c.to_dict() for c in self.alternatives],
            "considered": self.considered,
            "reason": self.reason,
        }


def candidate_times(
    item: RiskItem,
    now: datetime,
    downtime: DowntimeFeed,
    bank: str,
    horizon_end: datetime,
) -> list[datetime]:
    """The instants worth evaluating.

    Not a uniform grid. It contains, in order of importance:

    * now, if we are inside contactable hours;
    * the moment a bank outage covering this failure closes, which is when a
      free silent retry becomes worth roughly 0.86;
    * the next few salary credit dates, which is the entire liquidity story;
    * a coarse daily grid at the two times of day people actually respond.
    """
    times: list[datetime] = [now]

    resolution = downtime.resolution_after(item.failed_at, bank, item.method)
    if resolution is not None and resolution > now:
        times.append(resolution + timedelta(minutes=15))
    active = downtime.resolution_after(now, bank, item.method)
    if active is not None and active > now:
        times.append(active + timedelta(minutes=15))

    for day in _payday_candidates(now, horizon_end):
        times.append(day)

    cursor = to_ist(now).replace(minute=0, second=0, microsecond=0)
    for offset in range(0, PLANNING_HORIZON_DAYS + 1):
        base = cursor + timedelta(days=offset)
        for hour in CONTACT_HOURS_IST:
            times.append(to_utc(base.replace(hour=hour)))

    unique = sorted({t for t in times if now <= t < horizon_end})
    return unique


def _payday_candidates(now: datetime, horizon_end: datetime) -> list[datetime]:
    """Next occurrences of the salary credit days, at 11:00 IST.

    The agent does not know any individual customer's payday, exactly as it
    would not in production. It knows the population clusters on the 1st and 2nd,
    again around the 7th, and on the last day of the month for month-end
    payrolls, so it evaluates those instants and lets the model decide whether
    this particular item's features make them worth waiting for.
    """
    out: list[datetime] = []
    local = to_ist(now)
    year, month = local.year, local.month
    for _ in range(2):
        last = days_in_month(year, month)
        for day in (1, 2, 7, last):
            try:
                candidate = local.replace(
                    year=year, month=month, day=day, hour=11, minute=0, second=0, microsecond=0
                )
            except ValueError:
                continue
            stamp = to_utc(candidate)
            if now < stamp < horizon_end:
                out.append(stamp)
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def available_actions(item: RiskItem, customer: Customer, blocked: set[ActionKind]) -> list[ActionKind]:  # noqa: D401
    """Actions worth pricing.

    Filters on physical possibility only. Regulatory permission is the policy
    engine's job, and the planner deliberately does not duplicate that logic:
    two implementations of the same rule drift apart, and the one that drifts is
    always the shadow copy.
    """
    out: list[ActionKind] = []
    for action in (
        ActionKind.MANDATE_DEBIT,
        ActionKind.PAYMENT_LINK_SMS,
        ActionKind.PAYMENT_LINK_WHATSAPP,
        ActionKind.PAYMENT_LINK_EMAIL,
        ActionKind.UPDATE_INSTRUMENT,
        ActionKind.INCENTIVE_OFFER,
        ActionKind.VOICE_CALL,
        ActionKind.HUMAN_ESCALATION,
    ):
        if action in blocked:
            continue
        if action == ActionKind.MANDATE_DEBIT and not item.method.is_mandate:
            continue
        # Already handed over. Proposing it again would be refused by
        # R-ESCALATE-ONCE, and pricing an action that cannot happen just
        # displaces a real option out of the ranking.
        if action == ActionKind.HUMAN_ESCALATION and item.escalated_at is not None:
            continue
        channel = ACTION_CHANNEL[action]
        if channel in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL, Channel.VOICE):
            if not customer.channel_available(channel):
                continue
        out.append(action)
    return out


def annoyance_budget_for(item: RiskItem, base: float = ANNOYANCE_BUDGET) -> float:
    """Goodwill a single item may consume, scaled by the amount outstanding."""
    rupees = max(1.0, item.outstanding_paise / 100.0)
    bonus = 0.55 * math.log10(max(1.0, rupees / ANNOYANCE_BUDGET_PIVOT_RUPEES))
    return min(ANNOYANCE_BUDGET_CAP, base + max(0.0, bonus))


def goodwill_cost(action: ActionKind, customer: Customer) -> int:
    """Expected lifetime value destroyed by taking this action, in paise.

    Escalates with accumulated annoyance rather than staying flat, because the
    tenth message does more damage than the first. That convexity is what makes
    the planner spread contact across customers instead of hammering whichever
    one has the largest balance.
    """
    if not action.is_contact:
        return 0
    delta = ANNOYANCE_DELTA[ACTION_CHANNEL[action]]
    escalation = 1.0 + max(0.0, customer.annoyance)
    return int(ANNOYANCE_PRICE_PAISE * delta * escalation)


def plan(
    underwriter: Underwriter,
    item: RiskItem,
    customer: Customer,
    now: datetime,
    downtime: DowntimeFeed,
    bank: str,
    horizon_end: datetime,
    *,
    blocked: Optional[set[ActionKind]] = None,
    annoyance_price_paise: int = ANNOYANCE_PRICE_PAISE,
    annoyance_budget: float = ANNOYANCE_BUDGET,
    keep_alternatives: int = 3,
) -> Plan:
    """Price every (action, time) pair and return the best, if any beats zero."""
    blocked = blocked or set()
    outstanding = item.outstanding_paise
    if outstanding <= 0:
        return Plan(best=None, reason="nothing outstanding")

    actions = available_actions(item, customer, blocked)
    if not actions:
        return Plan(best=None, reason="no physically available action")

    times = candidate_times(item, now, downtime, bank, horizon_end)
    if not times:
        return Plan(best=None, reason="no candidate time inside the horizon")

    candidates: list[Candidate] = []
    budget = annoyance_budget_for(item, annoyance_budget)
    over_budget = customer.annoyance >= budget

    for action in actions:
        contact = action.is_contact
        # A customer already at their goodwill budget is not contacted again at
        # any price. Non-contact actions such as a mandate debit remain
        # available, which is the right outcome: it costs them nothing.
        if contact and over_budget:
            continue
        base_cost = ACTION_COST_PAISE.get(action, 0)
        if action == ActionKind.HUMAN_ESCALATION:
            base_cost += ESCALATION_SCARCITY_PREMIUM_PAISE
        goodwill = int(
            annoyance_price_paise
            * ANNOYANCE_DELTA[ACTION_CHANNEL[action]]
            * (1.0 + max(0.0, customer.annoyance))
        ) if contact else 0

        probabilities = underwriter.p_recover_batch(item, customer, times, downtime, bank, action)
        for when, p in zip(times, probabilities):
            days_out = max(0.0, (when - now).total_seconds() / 86400.0)
            discount = math.exp(-days_out / TAU_DELAY_DAYS)
            gross = int(p * outstanding * discount)
            ev = gross - base_cost - goodwill
            candidates.append(
                Candidate(
                    action=action,
                    at=when,
                    p_recover=p,
                    gross_paise=gross,
                    cost_paise=base_cost,
                    goodwill_paise=goodwill,
                    discount=discount,
                    ev_paise=ev,
                )
            )

    if not candidates:
        return Plan(
            best=None,
            considered=0,
            reason=(
                "goodwill budget exhausted for this customer"
                if over_budget
                else "no candidate action"
            ),
        )

    # Sort by value, then earliest, then a stable name. The tie-breaks matter:
    # without them two runs of the same seed could order equal-EV candidates
    # differently and the ledger hash would stop being reproducible.
    candidates.sort(key=lambda c: (-c.ev_paise, c.at, c.action.value))
    best = candidates[0]

    if best.ev_paise <= 0:
        return Plan(
            best=None,
            alternatives=candidates[:keep_alternatives],
            considered=len(candidates),
            reason=(
                f"best available action is worth {best.ev_paise} paise, "
                f"which does not justify the attempt"
            ),
        )

    # Keep only genuinely different alternatives for the receipt, so the top
    # three are three different ideas rather than the same action at three
    # adjacent times.
    alternatives: list[Candidate] = []
    seen: set[str] = {best.action.value}
    for candidate in candidates[1:]:
        if candidate.action.value in seen:
            continue
        seen.add(candidate.action.value)
        alternatives.append(candidate)
        if len(alternatives) >= keep_alternatives:
            break

    best.note = _explain(best, item, now)
    return Plan(best=best, alternatives=alternatives, considered=len(candidates), reason="positive expected value")


def _explain(candidate: Candidate, item: RiskItem, now: datetime) -> str:
    """One human sentence about the timing choice. Deterministic, not generated."""
    delay_hours = (candidate.at - now).total_seconds() / 3600.0
    if delay_hours < 1:
        return "acting now, the value of waiting is lower than acting immediately"
    if delay_hours < 36:
        return f"waiting {delay_hours:.0f}h for a better response window"
    day = to_ist(candidate.at)
    if day.day <= 3 or day.day == 7:
        return f"holding until {day.strftime('%d %b')}, the next likely salary credit"
    return f"holding {delay_hours / 24:.1f} days for a higher probability window"
