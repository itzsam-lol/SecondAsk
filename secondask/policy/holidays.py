"""Indian holiday calendar for collection outreach.

Automated commercial collection on Diwali morning is legal and stupid. It is the
kind of thing that produces a screenshot on social media rather than a payment,
and no amount of expected value justifies it.

Two tiers, deliberately separated because their reliability differs:

``FIXED_NATIONAL``
    The three gazetted national holidays plus Christmas. These fall on the same
    Gregorian date every year, so they are computed rather than tabulated and are
    correct for any year.

``OBSERVED``
    Major festivals whose dates follow lunar calendars and therefore move every
    year: Holi, the two Eids, Dussehra, Diwali, Guru Nanak Jayanti and others.
    These are **tabulated per year and must be refreshed annually** from the
    central government gazette. There is no algorithm here that derives them,
    because writing an approximate lunar calculation and presenting its output as
    a compliance control would be worse than a table with a known expiry.

``is_holiday`` fails **open** for years outside the table, and says so via
``coverage``. That is the deliberate choice: a missing calendar year should not
silently suspend all collections for a merchant, but it must be visible. The
runtime surfaces ``holiday_calendar_expired`` in its guard counters so a stale
table is noticed rather than discovered.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

from ..clock import ist_date

# Gazetted national holidays. Fixed Gregorian dates, valid for every year.
_FIXED: dict[tuple[int, int], str] = {
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
    (12, 25): "Christmas Day",
}

# Lunar and observed festivals, by year. MUST be refreshed annually.
#
# Sourced against the central government holiday list. Where a festival is
# observed across two days in different states, the principal day is listed.
# Regional-only holidays (Pongal, Onam, Bihu, Gudi Padwa) are deliberately
# absent: they would need a state mapping per customer, which this system does
# not have, and guessing would be worse than omitting.
_OBSERVED: dict[int, dict[tuple[int, int], str]] = {
    2025: {
        (3, 14): "Holi",
        (3, 31): "Eid-ul-Fitr",
        (6, 7): "Bakrid",
        (8, 16): "Janmashtami",
        (10, 2): "Dussehra",
        (10, 20): "Diwali",
        (11, 5): "Guru Nanak Jayanti",
    },
    2026: {
        (3, 4): "Holi",
        (3, 21): "Eid-ul-Fitr",
        (5, 27): "Bakrid",
        (9, 4): "Janmashtami",
        (10, 20): "Dussehra",
        (11, 8): "Diwali",
        (11, 24): "Guru Nanak Jayanti",
    },
    2027: {
        (3, 22): "Holi",
        (3, 11): "Eid-ul-Fitr",
        (5, 17): "Bakrid",
        (8, 25): "Janmashtami",
        (10, 9): "Dussehra",
        (10, 29): "Diwali",
        (11, 14): "Guru Nanak Jayanti",
    },
}

COVERED_YEARS = tuple(sorted(_OBSERVED))


def coverage() -> tuple[int, int]:
    """First and last year for which the observed-festival table is populated."""
    return COVERED_YEARS[0], COVERED_YEARS[-1]


def is_covered(year: int) -> bool:
    return year in _OBSERVED


def holiday_name(day: date) -> str | None:
    """Name of the holiday on ``day``, or None.

    Fixed national holidays resolve for any year. Observed festivals resolve
    only for tabulated years; outside those the function returns None, which
    means "not known to be a holiday" rather than "known not to be one". Callers
    that care about the difference should check ``is_covered``.
    """
    fixed = _FIXED.get((day.month, day.day))
    if fixed is not None:
        return fixed
    return _OBSERVED.get(day.year, {}).get((day.month, day.day))


def is_holiday(day: date) -> bool:
    return holiday_name(day) is not None


def is_holiday_ist(ts: datetime) -> bool:
    """Holiday check against the IST calendar date of an aware timestamp.

    The timezone matters: a 19:00 UTC action on 1 October is already 2 October in
    IST, which is Gandhi Jayanti. Checking the UTC date would miss it.
    """
    return is_holiday(ist_date(ts))


def next_working_day(day: date, *, limit: int = 10) -> date:
    """First non-holiday date strictly after ``day``.

    Weekends are not skipped. Collections legitimately run on Saturdays and
    Sundays in India, and treating them as closed would suspend a fifth of the
    recovery window for no regulatory reason.
    """
    from datetime import timedelta

    cursor = day
    for _ in range(limit):
        cursor = cursor + timedelta(days=1)
        if not is_holiday(cursor):
            return cursor
    return cursor


def holidays_in(year: int) -> list[tuple[date, str]]:
    """Every known holiday in a year, ordered. Useful for a compliance report."""
    out: list[tuple[date, str]] = []
    for (month, day), name in _FIXED.items():
        try:
            out.append((date(year, month, day), name))
        except ValueError:  # pragma: no cover, all fixed dates are valid
            continue
    for (month, day), name in _OBSERVED.get(year, {}).items():
        stamp = date(year, month, day)
        if not any(existing == stamp for existing, _ in out):
            out.append((stamp, name))
    return sorted(out)
