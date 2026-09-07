"""Analytic reference bounds.

The oracle agent turned out not to be an upper bound. It has perfect information
but plays greedily, one item at a time, and under the same contact caps as
everything else it recovered less than SecondAsk did. That is a real result
about greedy play, not a ceiling, and calling it a ceiling would have been
wrong. It is reported under its own name.

This module computes the reference properly, and states plainly what it is and
is not.

``best_single_action``
    For each item, the maximum over every action and every candidate time of
    ``p_success * outstanding``, evaluated against a pristine customer with no
    accumulated annoyance and with the world's own response model.

    This is an upper bound on **one attempt per item**. It is not a bound on the
    whole problem, because agents get several attempts and independent contact
    draws can compound. It is also unreachable in practice: it ignores contact
    caps, human review capacity, the goodwill budget and the fact that several
    items share a customer.

    What it is good for is scale. It says what fraction of the money in a batch
    is reachable at all, as opposed to structurally lost to dead instruments,
    unreachable customers and liquidity that never arrives. Reading a recovery
    rate without that denominator is how people conclude a system is
    underperforming when it is close to the physical limit.

``structurally_lost``
    The share of value whose best single action is worth almost nothing. Money
    that no policy recovers, so no agent should be judged on it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..world.entities import ActionKind, Customer, Method
from ..world.generator import World
from ..world.outcomes import success_probability

# Actions considered for the bound. WAIT and STOP recover nothing; PRENOTIFY is
# a precondition rather than a collection step.
_ACTIONS = (
    ActionKind.MANDATE_DEBIT,
    ActionKind.PAYMENT_LINK_SMS,
    ActionKind.PAYMENT_LINK_WHATSAPP,
    ActionKind.PAYMENT_LINK_EMAIL,
    ActionKind.UPDATE_INSTRUMENT,
    ActionKind.INCENTIVE_OFFER,
    ActionKind.VOICE_CALL,
    ActionKind.HUMAN_ESCALATION,
)

# An item counts as structurally hard when even its single best action, taken at
# its single best moment against a customer with no accumulated annoyance, is
# worth less than this. Set at 0.08 rather than something tiny because human
# escalation has a floor probability of roughly 0.05 against every blocker,
# including genuinely unreachable customers, so a lower threshold reports zero
# hard items and says nothing.
_UNREACHABLE_THRESHOLD = 0.08


def _pristine(customer: Customer) -> Customer:
    """A copy with contact history cleared.

    The bound asks what the best first attempt is worth, so it must not be
    charged for annoyance that a particular run happened to accumulate.
    """
    return Customer(
        customer_id=customer.customer_id,
        has_phone=customer.has_phone,
        has_email=customer.has_email,
        has_whatsapp=customer.has_whatsapp,
        language=customer.language,
        tenure_days=customer.tenure_days,
        prior_recoveries=customer.prior_recoveries,
        prior_ignores=customer.prior_ignores,
        _responsiveness=customer._responsiveness,
        _annoyance_tolerance=customer._annoyance_tolerance,
        _payday_day=customer._payday_day,
    )


def _times(item: Any, world: World) -> list:
    """Candidate instants, including the ones only the world knows about."""
    out = [item.failed_at + timedelta(hours=h) for h in (1, 4, 12, 24, 48, 96, 168, 240, 336)]
    if item._resolves_at is not None:
        out.append(item._resolves_at + timedelta(minutes=10))
        out.append(item._resolves_at + timedelta(hours=6))
    return [t for t in out if world.start <= t < world.end]


def best_single_action(world: World) -> dict[str, Any]:
    """Upper bound on one attempt per item. See the module docstring for caveats."""
    total_at_risk = 0
    total_bound = 0
    lost_value = 0
    lost_items = 0
    per_method: dict[str, dict[str, int]] = {}

    for item in world.items:
        customer = _pristine(world.customers[item.customer_id])
        amount = item.amount_paise
        total_at_risk += amount

        best_p = 0.0
        for action in _ACTIONS:
            if action == ActionKind.MANDATE_DEBIT and not item.method.is_mandate:
                continue
            for when in _times(item, world):
                p = success_probability(world, item, customer, action, when)
                if p > best_p:
                    best_p = p

        contribution = int(best_p * amount)
        total_bound += contribution

        bucket = per_method.setdefault(
            item.method.value, {"at_risk": 0, "bound": 0, "items": 0, "lost": 0}
        )
        bucket["at_risk"] += amount
        bucket["bound"] += contribution
        bucket["items"] += 1

        if best_p < _UNREACHABLE_THRESHOLD:
            lost_value += amount
            lost_items += 1
            bucket["lost"] += 1

    return {
        "at_risk_paise": total_at_risk,
        "best_single_action_paise": total_bound,
        "best_single_action_share": round(total_bound / total_at_risk, 4) if total_at_risk else 0.0,
        "structurally_hard_paise": lost_value,
        "structurally_hard_items": lost_items,
        "structurally_hard_threshold": _UNREACHABLE_THRESHOLD,
        "structurally_hard_share": round(lost_value / total_at_risk, 4) if total_at_risk else 0.0,
        "per_method": {k: v for k, v in sorted(per_method.items())},
        "caveat": (
            "upper bound on a single attempt per item, ignoring contact caps, human "
            "review capacity, the goodwill budget and shared customers. Agents take "
            "several attempts, so exceeding this figure is possible and is not a bug."
        ),
    }
