"""Bank and gateway downtime feed.

Razorpay emits ``payment.downtime.started`` / ``.resolved`` webhooks. This is a
free, high-value causal signal that almost no dunning system consumes: a failure
that lands inside a known issuer outage is somebody else's problem, it will
resolve on its own, and a silent retry after resolution costs nothing and works.
Contacting the customer about it is pure waste and pure annoyance.

Downtime is modelled as intervals per (bank, method). The agent may query this
feed for any timestamp, exactly as it could in production. What the agent may
*not* see is which specific failures were caused by an outage, because in
production you never know that either. You only know the failure landed in the
window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..clock import iso
from ..rng import Stream
from .entities import Method

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB", "YESB", "IDFC"]


@dataclass(frozen=True)
class DowntimeWindow:
    bank: str
    method: Method
    start: datetime
    end: datetime
    severity: float  # 0..1, share of traffic affected

    def covers(self, ts: datetime) -> bool:
        """Half-open interval ``[start, end)``.

        Half-open avoids the double-counting that a closed interval causes when
        one outage ends exactly as the next begins.
        """
        return self.start <= ts < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank": self.bank,
            "method": self.method.value,
            "start": iso(self.start),
            "end": iso(self.end),
            "severity": round(self.severity, 3),
        }


class DowntimeFeed:
    def __init__(self, windows: list[DowntimeWindow]) -> None:
        self._windows = sorted(windows, key=lambda w: w.start)

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def windows(self) -> tuple[DowntimeWindow, ...]:
        return tuple(self._windows)

    def active(self, ts: datetime, bank: str | None = None, method: Method | None = None) -> list[DowntimeWindow]:
        return [
            w
            for w in self._windows
            if w.covers(ts)
            and (bank is None or w.bank == bank)
            and (method is None or w.method == method)
        ]

    def is_degraded(self, ts: datetime, bank: str, method: Method) -> bool:
        return bool(self.active(ts, bank, method))

    def resolution_after(self, ts: datetime, bank: str, method: Method) -> datetime | None:
        """When the outage covering ``ts`` ends, or None if there isn't one."""
        windows = self.active(ts, bank, method)
        if not windows:
            return None
        return max(w.end for w in windows)

    def recent_degradation_ratio(self, ts: datetime, bank: str, method: Method, hours: int = 6) -> float:
        """Fraction of the trailing window that was degraded.

        A useful observable even when the failure itself sits just outside an
        outage: issuers rarely fail cleanly, and the hour either side of a
        recorded window is usually also bad.
        """
        span_start = ts - timedelta(hours=hours)
        degraded = timedelta(0)
        for w in self._windows:
            if w.method != method or w.bank != bank:
                continue
            overlap_start = max(w.start, span_start)
            overlap_end = min(w.end, ts)
            if overlap_end > overlap_start:
                degraded += overlap_end - overlap_start
        total = timedelta(hours=hours)
        if total.total_seconds() <= 0:
            return 0.0
        return min(1.0, degraded.total_seconds() / total.total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {"windows": [w.to_dict() for w in self._windows]}


def generate_downtime(stream: Stream, start: datetime, days: int) -> DowntimeFeed:
    """Generate a plausible outage calendar.

    Two shapes, because both occur and they demand different responses:

    * frequent short blips (5-40 minutes, low severity)
    * occasional real outages (1-4 hours, high severity) which produce a visible
      cluster of failures and are what the root-cause narration is for
    """
    windows: list[DowntimeWindow] = []
    sub = stream.sub("downtime")
    methods = [Method.UPI, Method.CARD, Method.NETBANKING, Method.EMANDATE_UPI, Method.NACH]

    blips = int(days * 3.5)
    for i in range(blips):
        bank = sub.choice(BANKS)
        method = sub.choice(methods)
        offset_minutes = sub.randint(0, max(1, days * 24 * 60 - 60))
        begin = start + timedelta(minutes=offset_minutes)
        length = sub.randint(5, 40)
        windows.append(
            DowntimeWindow(bank, method, begin, begin + timedelta(minutes=length), sub.uniform(0.1, 0.4))
        )

    outages = max(1, int(days * 0.4))
    for i in range(outages):
        bank = sub.choice(BANKS)
        method = sub.choice(methods)
        offset_minutes = sub.randint(0, max(1, days * 24 * 60 - 240))
        begin = start + timedelta(minutes=offset_minutes)
        length = sub.randint(60, 240)
        windows.append(
            DowntimeWindow(bank, method, begin, begin + timedelta(minutes=length), sub.uniform(0.5, 0.95))
        )

    return DowntimeFeed(windows)
