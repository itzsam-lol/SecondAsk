"""Money handling.

Every monetary value in this system is an ``int`` number of **paise**. Floats are
never used for money anywhere in the codebase. This is not pedantry: a float
rupee amount accumulates representation error across the tens of thousands of
arithmetic operations an evaluation run performs, and a recovery agent whose
reported "money recovered" drifts from the ledger is worthless.

The only place a float is permitted is a *ratio* (recovery rate, probability),
never an amount.
"""

from __future__ import annotations

import sys
from decimal import ROUND_HALF_UP, Decimal

# Rupee symbol. Windows consoles frequently run cp1252, which cannot encode
# U+20B9, so we resolve a safe symbol once at import time (see setup_stdout).
_RUPEE = "₹"
_ASCII_RUPEE = "Rs."

_symbol = _RUPEE


def setup_stdout() -> None:
    """Make stdout UTF-8 if possible, else fall back to an ASCII rupee symbol.

    Confirmed failure mode on a stock Windows 11 console::

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u20b9'

    A traceback on the reviewer's first run is an unacceptable first impression,
    so we degrade instead of crashing.
    """
    global _symbol
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass
    try:
        _RUPEE.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError, TypeError):
        _symbol = _ASCII_RUPEE


def force_ascii() -> None:
    """Force the ASCII rupee symbol regardless of console capability."""
    global _symbol
    _symbol = _ASCII_RUPEE


def rupees(amount: float | str | Decimal) -> int:
    """Convert a rupee amount to integer paise, half-up.

    Accepts str/Decimal for exactness. A float input is routed through str() so
    that ``rupees(0.07)`` is 700 paise rather than 699.
    """
    d = Decimal(str(amount))
    return int((d * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def to_rupees(paise: int) -> Decimal:
    """Exact rupee value of a paise amount, for display only."""
    return (Decimal(paise) / 100).quantize(Decimal("0.01"))


def fmt(paise: int, *, symbol: bool = True, compact: bool = False) -> str:
    """Format paise using the Indian digit grouping (lakh/crore)."""
    sign = "-" if paise < 0 else ""
    value = abs(paise)
    if compact:
        return f"{sign}{_symbol if symbol else ''}{_compact(value)}"
    whole, frac = divmod(value, 100)
    grouped = _indian_group(whole)
    body = f"{grouped}.{frac:02d}"
    return f"{sign}{_symbol if symbol else ''}{body}"


def _compact(paise: int) -> str:
    """Render large amounts as e.g. ``12.4L`` or ``3.1Cr``."""
    whole = paise // 100
    if whole >= 10_000_000:
        return f"{whole / 10_000_000:.2f}Cr"
    if whole >= 100_000:
        return f"{whole / 100_000:.2f}L"
    if whole >= 1_000:
        return f"{whole / 1_000:.1f}K"
    return _indian_group(whole)


def _indian_group(n: int) -> str:
    """2,2,3 grouping: 1234567 -> '12,34,567'."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def pct(numerator: float, denominator: float, *, places: int = 1) -> str:
    """Percentage that degrades gracefully instead of dividing by zero.

    Empty batches and zero-message runs are both real states this system must
    report on, so ``0/0`` must render rather than raise.
    """
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.{places}f}%"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of raising on a zero divisor."""
    if denominator == 0:
        return default
    return numerator / denominator
