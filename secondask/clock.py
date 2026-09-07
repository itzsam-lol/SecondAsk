"""Time handling and the virtual clock.

Design decisions, both of which are load-bearing:

1. **IST is a fixed ``+05:30`` offset, not ``zoneinfo``.**
   India has observed no daylight saving since 1945, so a fixed offset is exactly
   correct for every timestamp this system will ever see. ``ZoneInfo`` would add a
   dependency on system tzdata, which is absent on a stock Windows install and
   would raise ``ZoneInfoNotFoundError`` at import. Fixed offset is both more
   correct here and more portable.

2. **Everything is stored in UTC and converted to IST only at policy boundaries.**
   Contact-hour rules are defined in local time; storage and ordering are not.
   Mixing the two is how you end up sending a 2 AM SMS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")
UTC = timezone.utc

# Epoch used for deterministic runs. Chosen as a Monday so that weekday-dependent
# behaviour in tests is stable and easy to reason about.
DEFAULT_EPOCH = datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)  # Monday


def to_ist(ts: datetime) -> datetime:
    """Convert an aware timestamp to IST. Naive input is a programming error."""
    if ts.tzinfo is None:
        raise ValueError("naive datetime crossed a timezone boundary; all timestamps must be aware")
    return ts.astimezone(IST)


def to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("naive datetime crossed a timezone boundary; all timestamps must be aware")
    return ts.astimezone(UTC)


def ist_time_of_day(ts: datetime) -> float:
    """Hours past IST midnight as a float, e.g. 19.5 for 7:30 PM."""
    local = to_ist(ts)
    return local.hour + local.minute / 60.0 + local.second / 3600.0


def ist_date(ts: datetime) -> date:
    return to_ist(ts).date()


def hour_bucket(ts: datetime) -> int:
    """Whole hours since the Unix epoch.

    This is the granularity at which counterfactual outcome draws are keyed, so
    that two different policies taking the same action in the same hour observe
    the same random draw (common random numbers).
    """
    return int(to_utc(ts).timestamp()) // 3600


def days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days


def day_of_month_ist(ts: datetime) -> int:
    return to_ist(ts).day


def days_to_month_end_ist(ts: datetime) -> int:
    """Days remaining in the IST calendar month. Handles 28/29/30/31 correctly."""
    local = to_ist(ts).date()
    return days_in_month(local.year, local.month) - local.day


def is_month_end_window(ts: datetime, window: int = 3) -> bool:
    return days_to_month_end_ist(ts) < window


def next_ist_time(ts: datetime, hour: int, minute: int = 0) -> datetime:
    """The next instant at IST ``hour:minute`` strictly after ``ts``.

    Used to defer an action out of a forbidden contact window. Correct across
    midnight, month ends and year ends because it delegates to date arithmetic
    rather than manipulating hour fields.
    """
    local = to_ist(ts)
    candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local:
        candidate = candidate + timedelta(days=1)
        # Re-normalise: adding a day to an aware fixed-offset datetime cannot
        # produce a nonexistent local time (no DST), so this is safe.
    return to_utc(candidate)


@dataclass
class VirtualClock:
    """A deterministic clock.

    The whole system reads time through this object. Nothing calls
    ``datetime.now()``. That is what makes a run reproducible, makes "advance to
    19:01 and watch the agent refuse" a one-line test, and makes a 30-day
    recovery horizon simulate in under a second.
    """

    now: datetime = DEFAULT_EPOCH
    _start: datetime = field(init=False)

    def __post_init__(self) -> None:
        if self.now.tzinfo is None:
            raise ValueError("VirtualClock requires an aware datetime")
        self.now = to_utc(self.now)
        self._start = self.now

    @property
    def start(self) -> datetime:
        return self._start

    def advance_to(self, ts: datetime) -> None:
        """Move the clock forward. Time never runs backwards.

        Out-of-order webhook delivery is a real occurrence, but it must be
        handled by ordering on *event* timestamps, not by rewinding the clock.
        """
        ts = to_utc(ts)
        if ts < self.now:
            return
        self.now = ts

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + timedelta(**kwargs)

    def elapsed(self) -> timedelta:
        return self.now - self._start


def iso(ts: datetime) -> str:
    """Canonical serialisation. Always UTC, always second precision.

    Microseconds are dropped deliberately: they would otherwise leak into ledger
    hashes and make cross-platform reproducibility depend on float rounding.
    """
    return to_utc(ts).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    cleaned = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return to_utc(parsed)
