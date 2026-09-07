"""World generation.

Generates a batch of at-risk items with latent blockers, then emits an
*observable* error signature from ``p(observable | blocker, method)``. Working in
that direction (cause first, symptom second) is what produces honest aliasing:
several blockers can emit the same visible error, so the agent faces a real
inference problem rather than a decode table.

Parameter provenance is documented in METHODOLOGY.md. Rates are anchored to
published aggregates (UPI's very low technical decline rate versus card and
netbanking, liquidity dominating auto-debit failure, salary credit clustering at
the start of the month) and were fixed *before* any agent was written. They
were never tuned to make SecondAsk win, and the ablation table is the check on
that: if the parameters had been reverse-engineered to favour the agent, the
no-underwriter ablation would not degrade in the specific, mechanism-shaped way
it does.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..clock import DEFAULT_EPOCH, day_of_month_ist, days_in_month, to_ist, to_utc
from ..money import rupees
from ..rng import Stream
from .downtime import BANKS, DowntimeFeed, generate_downtime
from .entities import (
    Blocker,
    Customer,
    ErrorReason,
    ErrorSource,
    ErrorStep,
    Method,
    RiskItem,
)

# ---------------------------------------------------------------------------
# Method mix and ticket sizes
# ---------------------------------------------------------------------------

# Weighted by how often each rail appears in a *failure* batch, which is not the
# same as its share of traffic. UPI carries the most volume but fails least, so
# it is under-represented here relative to its transaction share.
METHOD_MIX: list[tuple[Method, float]] = [
    (Method.UPI, 0.24),
    (Method.CARD, 0.22),
    (Method.NETBANKING, 0.10),
    (Method.WALLET, 0.04),
    (Method.EMANDATE_UPI, 0.14),
    (Method.EMANDATE_CARD, 0.08),
    (Method.NACH, 0.10),
    (Method.INVOICE, 0.08),
]

# (median rupees, sigma, floor, ceiling) per method.
TICKET: dict[Method, tuple[int, float, int, int]] = {
    Method.UPI: (480, 1.15, 20, 100_000),
    Method.CARD: (1_850, 1.20, 50, 300_000),
    Method.NETBANKING: (3_200, 1.10, 100, 500_000),
    Method.WALLET: (320, 0.95, 20, 20_000),
    Method.EMANDATE_UPI: (699, 0.80, 49, 15_000),
    Method.EMANDATE_CARD: (899, 0.85, 49, 100_000),
    Method.NACH: (2_400, 0.95, 100, 100_000),
    # Held well below a true B2B distribution, deliberately.
    #
    # At the original 85,000 rupee median, invoices were 7% of items and 69% of
    # the value in the batch, so the value-weighted headline was decided by
    # whether about twenty items happened to land. That is a property of the
    # benchmark rather than of any agent, and it made the comparison unreadable.
    # These parameters keep every method under roughly half the total value, so
    # the metric measures policy quality instead of tail luck. The per-method
    # breakdown is reported regardless, because an aggregate that hides a
    # distribution is how this went unnoticed in the first place.
    Method.INVOICE: (35_000, 0.80, 5_000, 600_000),
}

# ---------------------------------------------------------------------------
# Latent blocker priors
# ---------------------------------------------------------------------------

ONE_OFF_BLOCKERS: list[tuple[Blocker, float]] = [
    (Blocker.AUTH_FRICTION, 0.30),
    (Blocker.TRANSIENT_INFRA, 0.25),
    (Blocker.INTENT_LOST, 0.18),
    (Blocker.LIQUIDITY, 0.15),
    (Blocker.INSTRUMENT_DEAD, 0.07),
    (Blocker.UNREACHABLE, 0.03),
    (Blocker.DISPUTE, 0.02),
]

# Liquidity dominates auto-debit failure. This is the single most consequential
# prior in the file: it is why timing beats persistence on mandate rails, and
# why a fixed +1h/+24h/+72h schedule wastes almost all of its attempts.
MANDATE_BLOCKERS: list[tuple[Blocker, float]] = [
    (Blocker.LIQUIDITY, 0.45),
    (Blocker.INSTRUMENT_DEAD, 0.20),
    (Blocker.TRANSIENT_INFRA, 0.20),
    (Blocker.INTENT_LOST, 0.08),
    (Blocker.DISPUTE, 0.04),
    (Blocker.UNREACHABLE, 0.03),
]

INVOICE_BLOCKERS: list[tuple[Blocker, float]] = [
    (Blocker.LIQUIDITY, 0.35),
    (Blocker.INTENT_LOST, 0.30),
    (Blocker.DISPUTE, 0.20),
    (Blocker.UNREACHABLE, 0.08),
    (Blocker.INSTRUMENT_DEAD, 0.05),
    (Blocker.TRANSIENT_INFRA, 0.02),
]

# ---------------------------------------------------------------------------
# Observable emission: p(error_source, error_reason | blocker)
# ---------------------------------------------------------------------------
#
# The overlaps are the point. Read down the `reason` columns:
#   payment_timed_out   <- TRANSIENT_INFRA and AUTH_FRICTION
#   payment_failed      <- every blocker, at some rate
#   insufficient_funds  <- mostly LIQUIDITY, but INSTRUMENT_DEAD leaks in
# so no single observable identifies a blocker, and a classifier has to combine
# the error signature with method, downtime and calendar context to do better
# than the base rate.

EMISSION: dict[Blocker, dict[str, list[tuple[Any, float]]]] = {
    Blocker.TRANSIENT_INFRA: {
        "source": [(ErrorSource.BANK, 0.50), (ErrorSource.GATEWAY, 0.35), (ErrorSource.INTERNAL, 0.15)],
        "reason": [
            (ErrorReason.GATEWAY_TECHNICAL_ERROR, 0.32),
            (ErrorReason.PAYMENT_FAILED, 0.28),
            (ErrorReason.ISSUER_DOWN, 0.22),
            (ErrorReason.PAYMENT_TIMED_OUT, 0.18),
        ],
        "step": [(ErrorStep.AUTHORIZATION, 0.55), (ErrorStep.RESPONSE, 0.30), (ErrorStep.INITIATION, 0.15)],
    },
    Blocker.LIQUIDITY: {
        "source": [(ErrorSource.CUSTOMER, 0.85), (ErrorSource.BANK, 0.15)],
        "reason": [
            (ErrorReason.INSUFFICIENT_FUNDS, 0.78),
            (ErrorReason.PAYMENT_FAILED, 0.16),
            (ErrorReason.LIMIT_EXCEEDED, 0.06),
        ],
        "step": [(ErrorStep.AUTHORIZATION, 0.90), (ErrorStep.RESPONSE, 0.10)],
    },
    Blocker.INSTRUMENT_DEAD: {
        "source": [(ErrorSource.CUSTOMER, 0.68), (ErrorSource.BANK, 0.32)],
        "reason": [
            (ErrorReason.CARD_EXPIRED, 0.24),
            (ErrorReason.ACCOUNT_CLOSED, 0.16),
            (ErrorReason.MANDATE_REVOKED, 0.18),
            (ErrorReason.CARD_DISABLED_ONLINE, 0.14),
            (ErrorReason.MANDATE_NOT_FOUND, 0.08),
            (ErrorReason.ACCOUNT_FROZEN, 0.06),
            (ErrorReason.CARD_DECLINED, 0.08),
            # The leak that makes this hard: 6% look like plain liquidity.
            (ErrorReason.INSUFFICIENT_FUNDS, 0.06),
        ],
        "step": [(ErrorStep.AUTHORIZATION, 0.70), (ErrorStep.INITIATION, 0.30)],
    },
    Blocker.AUTH_FRICTION: {
        "source": [(ErrorSource.CUSTOMER, 0.80), (ErrorSource.GATEWAY, 0.20)],
        "reason": [
            (ErrorReason.INVALID_OTP, 0.34),
            (ErrorReason.AUTHENTICATION_FAILED, 0.30),
            (ErrorReason.PAYMENT_TIMED_OUT, 0.26),
            (ErrorReason.PAYMENT_FAILED, 0.10),
        ],
        "step": [(ErrorStep.AUTHENTICATION, 0.85), (ErrorStep.INITIATION, 0.15)],
    },
    Blocker.INTENT_LOST: {
        "source": [(ErrorSource.CUSTOMER, 0.55), (ErrorSource.NA, 0.45)],
        "reason": [
            (ErrorReason.CHECKOUT_ABANDONED, 0.55),
            (ErrorReason.PAYMENT_FAILED, 0.30),
            (ErrorReason.INVOICE_OVERDUE, 0.15),
        ],
        "step": [(ErrorStep.INITIATION, 0.60), (ErrorStep.NA, 0.40)],
    },
    Blocker.DISPUTE: {
        "source": [(ErrorSource.CUSTOMER, 0.60), (ErrorSource.NA, 0.25), (ErrorSource.BUSINESS, 0.15)],
        "reason": [
            (ErrorReason.PAYMENT_FAILED, 0.45),
            (ErrorReason.INVOICE_OVERDUE, 0.35),
            (ErrorReason.CHECKOUT_ABANDONED, 0.20),
        ],
        "step": [(ErrorStep.NA, 0.60), (ErrorStep.INITIATION, 0.40)],
    },
    Blocker.UNREACHABLE: {
        # Deliberately uninformative: an unreachable customer's failure looks
        # like any other failure. They are only identifiable from the *absence*
        # of any response, which is exactly the inference the agent must make.
        "source": [
            (ErrorSource.CUSTOMER, 0.40),
            (ErrorSource.BANK, 0.25),
            (ErrorSource.NA, 0.20),
            (ErrorSource.UNKNOWN, 0.15),
        ],
        "reason": [
            (ErrorReason.PAYMENT_FAILED, 0.40),
            (ErrorReason.INSUFFICIENT_FUNDS, 0.20),
            (ErrorReason.CHECKOUT_ABANDONED, 0.20),
            (ErrorReason.UNKNOWN, 0.20),
        ],
        "step": [(ErrorStep.NA, 0.50), (ErrorStep.AUTHORIZATION, 0.50)],
    },
}

# Salary credit days. The 1st dominates, with a secondary cluster at the 7th and
# a month-end group (last working day). This is the prior that makes
# "retry insufficient_funds on the 2nd, not the 28th" a winning move.
PAYDAY_MIX: list[tuple[int, float]] = [
    (1, 0.44),
    (2, 0.09),
    (5, 0.07),
    (7, 0.14),
    (10, 0.06),
    (0, 0.20),  # 0 is a sentinel for "last day of month"
]

LANGUAGES: list[tuple[str, float]] = [
    ("en", 0.42),
    ("hi", 0.31),
    ("hinglish", 0.15),
    ("ta", 0.05),
    ("te", 0.04),
    ("mr", 0.03),
]


class World:
    """A generated batch plus everything needed to evaluate a policy against it."""

    def __init__(
        self,
        seed: int,
        items: list[RiskItem],
        customers: dict[str, Customer],
        downtime: DowntimeFeed,
        start: datetime,
        horizon_days: int,
        item_banks: dict[str, str],
    ) -> None:
        self.seed = seed
        self.items = items
        self.customers = customers
        self.downtime = downtime
        self.start = start
        self.horizon_days = horizon_days
        self.item_banks = item_banks

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.horizon_days)

    @property
    def total_at_risk_paise(self) -> int:
        return sum(i.amount_paise for i in self.items)

    def bank_of(self, item: RiskItem) -> str:
        return self.item_banks.get(item.item_id, "HDFC")

    def summary(self) -> dict[str, Any]:
        by_method: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_blocker: dict[str, int] = {}
        for item in self.items:
            by_method[item.method.value] = by_method.get(item.method.value, 0) + 1
            by_source[item.error_source.value] = by_source.get(item.error_source.value, 0) + 1
            by_blocker[item._blocker.value] = by_blocker.get(item._blocker.value, 0) + 1
        return {
            "seed": self.seed,
            "items": len(self.items),
            "customers": len(self.customers),
            "at_risk_paise": self.total_at_risk_paise,
            "horizon_days": self.horizon_days,
            "downtime_windows": len(self.downtime),
            "by_method": dict(sorted(by_method.items())),
            "by_error_source": dict(sorted(by_source.items())),
            "by_blocker_latent": dict(sorted(by_blocker.items())),
        }


def _next_payday(after: datetime, payday_day: int, stream: Stream) -> datetime:
    """Next salary credit strictly after ``after``.

    ``payday_day == 0`` means the last day of the month. Handles 28/29/30/31 and
    February in leap years by deriving the day count rather than assuming 30.
    """
    local = to_ist(after)
    year, month = local.year, local.month
    for _ in range(3):  # at most three months ahead
        last = days_in_month(year, month)
        day = last if payday_day == 0 else min(payday_day, last)
        candidate = local.replace(
            year=year, month=month, day=day, hour=11, minute=0, second=0, microsecond=0
        )
        # Salary lands over a few hours, not on the stroke of a fixed time.
        candidate = candidate + timedelta(minutes=stream.randint(0, 600))
        if candidate > local:
            return to_utc(candidate)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return to_utc(local + timedelta(days=30))


def generate_world(
    seed: int,
    n_items: int = 1000,
    horizon_days: int = 21,
    start: datetime = DEFAULT_EPOCH,
    customer_ratio: float = 0.82,
) -> World:
    """Build a world.

    ``customer_ratio`` below 1.0 means some customers own several failed items.
    That is deliberate and it is where per-customer annoyance budgeting earns its
    keep: a naive per-item system sends this cohort two or three simultaneous
    dunning sequences.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    root = Stream(seed, "world")
    cust_stream = root.sub("customers")
    item_stream = root.sub("items")

    n_customers = max(1, int(n_items * customer_ratio))
    customers: dict[str, Customer] = {}
    for i in range(n_customers):
        cid = f"cust_{seed}_{i:05d}"
        # ~4% have no usable phone, ~18% no email. Both matter: a customer with
        # neither can only be reached by escalation, and the agent must notice
        # rather than burn attempts on channels that do not exist.
        has_phone = cust_stream.chance(0.96)
        customers[cid] = Customer(
            customer_id=cid,
            has_phone=has_phone,
            has_email=cust_stream.chance(0.82),
            has_whatsapp=has_phone and cust_stream.chance(0.88),
            language=cust_stream.weighted(LANGUAGES),
            tenure_days=cust_stream.lognormal_int(180, 1.1, low=1, high=3000),
            prior_recoveries=cust_stream.weighted([(0, 0.6), (1, 0.22), (2, 0.11), (3, 0.07)]),
            prior_ignores=cust_stream.weighted([(0, 0.55), (1, 0.24), (2, 0.13), (3, 0.08)]),
            _responsiveness=min(0.97, max(0.03, cust_stream.uniform(0.08, 0.92))),
            _annoyance_tolerance=cust_stream.uniform(1.8, 5.5),
            _payday_day=cust_stream.weighted(PAYDAY_MIX),
        )

    customer_ids = list(customers.keys())
    downtime = generate_downtime(root, start, horizon_days)

    items: list[RiskItem] = []
    item_banks: dict[str, str] = {}

    for i in range(n_items):
        item_id = f"item_{seed}_{i:05d}"
        method = item_stream.weighted(METHOD_MIX)
        median, sigma, low, high = TICKET[method]
        amount = rupees(item_stream.lognormal_int(median, sigma, low=low, high=high))
        cid = item_stream.choice(customer_ids)
        customer = customers[cid]
        bank = item_stream.choice(BANKS)

        if method == Method.INVOICE:
            priors = INVOICE_BLOCKERS
        elif method.is_mandate:
            priors = MANDATE_BLOCKERS
        else:
            priors = ONE_OFF_BLOCKERS
        blocker = item_stream.weighted(priors)

        # Failures are spread over the first 40% of the horizon so that every
        # item has room for a recovery sequence inside the window. An item that
        # fails on the last day cannot be recovered by anyone, and including
        # such items would just add noise that is identical across agents.
        max_offset_minutes = max(60, int(horizon_days * 24 * 60 * 0.40))
        failed_at = start + timedelta(minutes=item_stream.randint(0, max_offset_minutes))

        # A share of TRANSIENT_INFRA failures are pinned into a real downtime
        # window, so the downtime feed is genuinely predictive rather than
        # decorative. The rest fail outside any recorded window, which is also
        # true in production, since not every outage gets a webhook.
        resolves_at: datetime | None = None
        if blocker == Blocker.TRANSIENT_INFRA:
            if item_stream.chance(0.62):
                candidates = [
                    w
                    for w in downtime.windows
                    if w.method == method and w.start >= start and w.end <= start + timedelta(days=horizon_days)
                ]
                if candidates:
                    window = item_stream.choice(candidates)
                    bank = window.bank
                    span = max(1, int((window.end - window.start).total_seconds() // 60))
                    failed_at = window.start + timedelta(minutes=item_stream.randint(0, span - 1) if span > 1 else 0)
                    resolves_at = window.end
            if resolves_at is None:
                resolves_at = failed_at + timedelta(minutes=item_stream.randint(20, 300))

        elif blocker == Blocker.LIQUIDITY:
            # 12% never become liquid inside the horizon: job loss, genuine
            # distress. These are unrecoverable and the correct action is to
            # stop early, which is precisely what a persistence-based policy
            # will not do.
            if item_stream.chance(0.88):
                resolves_at = _next_payday(failed_at, customer._payday_day, item_stream)
            else:
                resolves_at = None

        error_source = item_stream.weighted(EMISSION[blocker]["source"])
        error_reason = item_stream.weighted(EMISSION[blocker]["reason"])
        error_step = item_stream.weighted(EMISSION[blocker]["step"])

        # Mandate rails have no authentication step, so an auth-shaped reason
        # there would be a tell. Rewrite it to the generic failure.
        if method.is_mandate and error_reason in (
            ErrorReason.INVALID_OTP,
            ErrorReason.AUTHENTICATION_FAILED,
        ):
            error_reason = ErrorReason.PAYMENT_FAILED
            error_step = ErrorStep.AUTHORIZATION
        if method == Method.INVOICE:
            error_step = ErrorStep.NA
            if error_reason in (ErrorReason.CHECKOUT_ABANDONED, ErrorReason.PAYMENT_FAILED):
                error_reason = ErrorReason.INVOICE_OVERDUE

        is_msme = method == Method.INVOICE and item_stream.chance(0.58)
        due_at = None
        agreement_days = 45
        if method == Method.INVOICE:
            agreement_days = item_stream.weighted([(15, 0.2), (30, 0.35), (45, 0.35), (60, 0.10)])
            due_at = failed_at - timedelta(days=item_stream.randint(1, 40))

        item = RiskItem(
            item_id=item_id,
            customer_id=cid,
            amount_paise=amount,
            method=method,
            error_source=error_source,
            error_reason=error_reason,
            error_step=error_step,
            failed_at=failed_at,
            order_id=f"order_{seed}_{i:05d}",
            is_msme_supplier=is_msme,
            invoice_due_at=due_at,
            agreement_days=agreement_days,
            _blocker=blocker,
            _resolves_at=resolves_at,
        )
        items.append(item)
        item_banks[item_id] = bank

    items.sort(key=lambda x: (x.failed_at, x.item_id))
    return World(seed, items, customers, downtime, start, horizon_days, item_banks)
