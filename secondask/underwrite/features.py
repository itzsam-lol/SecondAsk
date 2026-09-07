"""Feature extraction.

Strictly observable. Every value here can be computed in production from a
Razorpay webhook, the downtime feed, a calendar and our own contact history.
Nothing reads ``item._blocker`` or ``item._resolves_at``, and
``test_no_latent_leakage`` asserts that by constructing a world, corrupting every
latent field, and checking the feature vectors are unchanged.

That test matters more than it looks. A simulator-trained model that peeks at
latent state produces beautiful benchmark numbers and is worth nothing, and the
failure is invisible unless something checks for it explicitly.

Three feature groups do the real work:

**Error signature.** ``error_source`` plus a coarse class over ``error_reason``.
This is what separates "the bank was down" from "the card is dead", and it is
free on every webhook.

**Downtime context.** Whether the failure landed inside a recorded outage,
whether that outage has since closed, and how degraded the recent window was.
Turns a guess about transience into an observation.

**Calendar.** Day of month, distance to the start of the month and to month end.
The agent does not know any individual customer's payday, exactly as it would not
in production. It knows the population-level salary cycle, and that is enough to
learn that an insufficient-funds failure retried on the 2nd behaves differently
from the same failure retried on the 26th.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..clock import day_of_month_ist, days_in_month, days_to_month_end_ist, ist_time_of_day, to_ist
from ..world.entities import (
    ActionKind,
    Customer,
    ErrorReason,
    ErrorSource,
    ErrorStep,
    Method,
    RiskItem,
)
from ..world.downtime import DowntimeFeed
import math

# Coarse classes over error_reason. Grouping matters: the raw vocabulary is long,
# sparse and provider-specific, whereas these five classes are stable and are
# what actually determines whether a retry can work.
REASON_CLASSES = ("liquidity", "instrument", "auth", "infra", "abandon", "other")

_REASON_CLASS: dict[ErrorReason, str] = {
    ErrorReason.INSUFFICIENT_FUNDS: "liquidity",
    ErrorReason.LIMIT_EXCEEDED: "liquidity",
    ErrorReason.CARD_EXPIRED: "instrument",
    ErrorReason.CARD_DECLINED: "instrument",
    ErrorReason.CARD_DISABLED_ONLINE: "instrument",
    ErrorReason.MANDATE_REVOKED: "instrument",
    ErrorReason.MANDATE_NOT_FOUND: "instrument",
    ErrorReason.ACCOUNT_CLOSED: "instrument",
    ErrorReason.ACCOUNT_FROZEN: "instrument",
    ErrorReason.INVALID_OTP: "auth",
    ErrorReason.AUTHENTICATION_FAILED: "auth",
    ErrorReason.PAYMENT_TIMED_OUT: "auth",
    ErrorReason.GATEWAY_TECHNICAL_ERROR: "infra",
    ErrorReason.ISSUER_DOWN: "infra",
    ErrorReason.CHECKOUT_ABANDONED: "abandon",
    ErrorReason.INVOICE_OVERDUE: "abandon",
    ErrorReason.PAYMENT_FAILED: "other",
    ErrorReason.UNKNOWN: "other",
}

SOURCES = tuple(s.value for s in ErrorSource)
METHODS = tuple(m.value for m in Method)
STEPS = tuple(s.value for s in ErrorStep)


def reason_class(reason: ErrorReason) -> str:
    """Never raises on an unseen reason.

    A payment provider adding a new error code should degrade the model's
    accuracy on those payments, not crash the recovery loop for every payment.
    """
    return _REASON_CLASS.get(reason, "other")


FEATURE_NAMES: list[str] = (
    [f"method={m}" for m in METHODS]
    + [f"source={s}" for s in SOURCES]
    + [f"reason_class={r}" for r in REASON_CLASSES]
    + [f"step={s}" for s in STEPS]
    + [
        "log_amount",
        "is_mandate",
        "silent_retry_allowed",
        "attempts",
        "contacts",
        "log_hours_since_failure",
        "downtime_at_failure",
        "downtime_active_now",
        "downtime_resolved_since",
        "log_hours_since_resolution",
        "recent_degradation_ratio",
        "day_of_month_norm",
        "days_since_month_start",
        "days_to_month_end",
        "in_payday_window",
        "crossed_payday_since_failure",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "log_tenure_days",
        "prior_recoveries",
        "prior_ignores",
        "annoyance",
        "has_phone",
        "has_whatsapp",
        "has_email",
        "opted_out_count",
        "promise_outstanding",
        "is_msme_invoice",
    ]
)

N_FEATURES = len(FEATURE_NAMES)


def _one_hot(value: str, options: tuple[str, ...]) -> list[float]:
    """Unknown values produce an all-zero block rather than an exception.

    That is the right degradation: the model falls back on the remaining
    features instead of the pipeline dying on an unrecognised enum from a
    provider update.
    """
    return [1.0 if value == option else 0.0 for option in options]


def extract(
    item: RiskItem,
    customer: Customer,
    now: datetime,
    downtime: DowntimeFeed,
    bank: str,
) -> list[float]:
    """Build the observable feature vector for one item at one instant."""
    hours_since_failure = max(0.0, (now - item.failed_at).total_seconds() / 3600.0)

    at_failure = downtime.active(item.failed_at, bank, item.method)
    active_now = downtime.active(now, bank, item.method)
    resolution = max((w.end for w in at_failure), default=None)
    resolved_since = 1.0 if (resolution is not None and now >= resolution) else 0.0
    hours_since_resolution = (
        max(0.0, (now - resolution).total_seconds() / 3600.0) if resolution is not None and now >= resolution else 0.0
    )

    local = to_ist(now)
    dom = day_of_month_ist(now)
    month_len = days_in_month(local.year, local.month)
    to_month_end = days_to_month_end_ist(now)
    # The payday window covers the start-of-month salary cluster and the
    # last day, which is when month-end payrolls land.
    in_payday_window = 1.0 if (dom <= 5 or to_month_end == 0) else 0.0
    # Did a plausible salary credit fall between the failure and now? This is
    # the single feature that separates "wait two days" from "give up".
    crossed = 0.0
    if hours_since_failure > 0:
        failed_local = to_ist(item.failed_at)
        if failed_local.month != local.month or failed_local.year != local.year:
            crossed = 1.0
        elif day_of_month_ist(item.failed_at) < 5 <= dom:
            crossed = 1.0

    hour = ist_time_of_day(now)
    angle = 2.0 * math.pi * hour / 24.0

    vector: list[float] = []
    vector += _one_hot(item.method.value, METHODS)
    vector += _one_hot(item.error_source.value, SOURCES)
    vector += _one_hot(reason_class(item.error_reason), REASON_CLASSES)
    vector += _one_hot(item.error_step.value, STEPS)
    vector += [
        math.log1p(max(0, item.outstanding_paise) / 100.0),
        1.0 if item.method.is_mandate else 0.0,
        1.0 if item.method.supports_silent_retry else 0.0,
        float(item.attempts),
        float(item.contacts),
        math.log1p(hours_since_failure),
        1.0 if at_failure else 0.0,
        1.0 if active_now else 0.0,
        resolved_since,
        math.log1p(hours_since_resolution),
        downtime.recent_degradation_ratio(now, bank, item.method),
        dom / float(month_len),
        float(min(dom - 1, 15)),
        float(min(to_month_end, 15)),
        in_payday_window,
        crossed,
        math.sin(angle),
        math.cos(angle),
        1.0 if local.weekday() >= 5 else 0.0,
        math.log1p(max(0, customer.tenure_days)),
        float(customer.prior_recoveries),
        float(customer.prior_ignores),
        float(customer.annoyance),
        1.0 if customer.has_phone else 0.0,
        1.0 if customer.has_whatsapp else 0.0,
        1.0 if customer.has_email else 0.0,
        float(len(customer.opted_out_channels)),
        1.0 if (item.promise_to_pay_at is not None and now < item.promise_to_pay_at) else 0.0,
        1.0 if item.is_msme_supplier else 0.0,
    ]

    if len(vector) != N_FEATURES:
        raise AssertionError(f"feature vector is {len(vector)} long, FEATURE_NAMES declares {N_FEATURES}")
    return vector
