"""Circuit breaker and bounded retry.

A recovery agent talks to a payment gateway thousands of times per run. When that
gateway degrades, the naive behaviour (retry every call, immediately, forever)
converts a partial outage into a self-inflicted outage and, worse, can duplicate
side effects.

Standard three-state breaker:

    CLOSED     calls pass; consecutive failures are counted
    OPEN       calls are refused immediately for ``reset_after`` seconds
    HALF_OPEN  a single probe is allowed; success closes, failure re-opens

The breaker runs on the **virtual clock**, not wall time, so its behaviour is
reproducible and a 21-day simulated run does not take 21 real days to exercise
the reset path.

Backoff is exponential with full jitter, derived deterministically from the
idempotency key rather than from ``random``, so a replayed run produces exactly
the same delays. Jitter without determinism would break reproducibility;
determinism without jitter would synchronise every retry in the batch into a
thundering herd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..rng import crn_uniform


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(RuntimeError):
    """Raised when a call is refused because the breaker is open."""


@dataclass
class CircuitBreaker:
    name: str = "razorpay"
    failure_threshold: int = 5
    reset_after_seconds: int = 120
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: datetime | None = None
    trips: int = 0
    refused: int = 0

    def before_call(self, now: datetime) -> None:
        if self.state == BreakerState.OPEN:
            assert self.opened_at is not None
            if now - self.opened_at >= timedelta(seconds=self.reset_after_seconds):
                self.state = BreakerState.HALF_OPEN
            else:
                self.refused += 1
                raise CircuitOpen(
                    f"circuit '{self.name}' is open until "
                    f"{(self.opened_at + timedelta(seconds=self.reset_after_seconds)).isoformat()}"
                )

    def on_success(self) -> None:
        self.consecutive_failures = 0
        self.state = BreakerState.CLOSED
        self.opened_at = None

    def on_failure(self, now: datetime) -> None:
        self.consecutive_failures += 1
        # A failure while probing re-opens immediately; do not wait for the
        # threshold again, the probe was the test.
        if self.state == BreakerState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = now
            self.trips += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "trips": self.trips,
            "refused_calls": self.refused,
            "consecutive_failures": self.consecutive_failures,
        }


def backoff_delay(attempt: int, key: str, *, base_seconds: float = 0.5, cap_seconds: float = 30.0) -> float:
    """Exponential backoff with deterministic full jitter.

    ``attempt`` is 1-based. The exponent is clamped before the shift so that a
    runaway attempt counter cannot produce an overflow or a delay measured in
    years.
    """
    if attempt < 1:
        attempt = 1
    exponent = min(attempt - 1, 10)
    ceiling = min(cap_seconds, base_seconds * (2**exponent))
    return ceiling * crn_uniform("backoff", key, attempt)
