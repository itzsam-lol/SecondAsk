"""Confidence intervals across seeds.

The previous benchmark reported "1.94x the money" from three seeds with no
variance. That is a fair thing to be sceptical of: three draws is enough to see a
large effect and nowhere near enough to say how large, and quoting a ratio to two
decimals from three samples implies a precision that is not there.

Two intervals, because they answer different questions:

``bootstrap_ci``
    Percentile bootstrap over the per-seed values. Makes no distributional
    assumption, which matters because recovered-rupees per seed is right-skewed:
    a seed containing one large invoice that happened to land is not drawn from
    anything symmetric.

``paired_bootstrap_ci``
    For a *difference* between two agents. This is the one that matters, and it
    resamples **seeds, not agents**: both agents saw the same world on a given
    seed, so their results are paired and the pairing removes almost all the
    variance that comes from the batch rather than from the policy. Treating the
    two as independent samples would throw that away and give an interval several
    times too wide.

Small-sample honesty is enforced rather than documented. Below
``MIN_SEEDS_FOR_CI`` the report says so instead of printing an interval that
looks authoritative and is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..rng import stable_hash

MIN_SEEDS_FOR_CI = 5
DEFAULT_RESAMPLES = 20000


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n: int
    method: str = "bootstrap"

    @property
    def reliable(self) -> bool:
        return self.n >= MIN_SEEDS_FOR_CI

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0 and self.high > 0) or (self.low < 0 and self.high < 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "n_seeds": self.n,
            "method": self.method,
            "reliable": self.reliable,
        }

    def render(self, fmt: Callable[[float], str] = lambda v: f"{v:.4g}") -> str:
        if not self.reliable:
            return f"{fmt(self.point)} (n={self.n}, too few seeds for an interval)"
        return f"{fmt(self.point)}  [{fmt(self.low)}, {fmt(self.high)}]"


def _resample_indices(n: int, draw: int, salt: str) -> list[int]:
    """Deterministic resample.

    Seeded from a stable hash rather than ``random``, so a reported interval is
    reproducible. An interval that moves between runs of the same data is not a
    statistic, it is a rumour.
    """
    return [stable_hash(salt, draw, k) % n for k in range(n)]


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    salt: str = "ci",
) -> Interval:
    """Percentile bootstrap for the mean of ``values``."""
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    point = sum(data) / n
    if n == 1:
        return Interval(point, point, point, 1)

    means = []
    for draw in range(resamples):
        idx = _resample_indices(n, draw, salt)
        means.append(sum(data[i] for i in idx) / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * resamples))]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(point, lo, hi, n)


def paired_bootstrap_ci(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    salt: str = "paired",
) -> Interval:
    """Bootstrap for the mean per-seed difference, resampling seeds.

    Both agents ran the same seeds under common random numbers, so element i of
    each sequence describes the same world. Resampling that pairing keeps the
    batch-to-batch variance out of the interval, which is the entire reason the
    benchmark uses shared seeds in the first place.
    """
    if len(treatment) != len(control):
        raise ValueError("paired bootstrap needs equal-length sequences")
    deltas = [float(a) - float(b) for a, b in zip(treatment, control)]
    interval = bootstrap_ci(deltas, resamples=resamples, alpha=alpha, salt=salt)
    interval.method = "paired bootstrap"
    return interval


def ratio_ci(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    salt: str = "ratio",
) -> Interval:
    """Bootstrap for the ratio of totals, resampling seeds jointly.

    The ratio of sums, not the mean of per-seed ratios. Those differ, and the
    second is the wrong one: a seed with a tiny denominator produces an enormous
    per-seed ratio and drags the average somewhere no individual seed went.
    """
    if len(treatment) != len(control):
        raise ValueError("ratio bootstrap needs equal-length sequences")
    t = [float(v) for v in treatment]
    c = [float(v) for v in control]
    n = len(t)
    if n == 0 or sum(c) == 0:
        return Interval(0.0, 0.0, 0.0, n, "ratio bootstrap")
    point = sum(t) / sum(c)
    if n == 1:
        return Interval(point, point, point, 1, "ratio bootstrap")

    ratios = []
    for draw in range(resamples):
        idx = _resample_indices(n, draw, salt)
        denominator = sum(c[i] for i in idx)
        if denominator == 0:
            continue
        ratios.append(sum(t[i] for i in idx) / denominator)
    if not ratios:
        return Interval(point, point, point, n, "ratio bootstrap")
    ratios.sort()
    lo = ratios[max(0, int((alpha / 2) * len(ratios)))]
    hi = ratios[min(len(ratios) - 1, int((1 - alpha / 2) * len(ratios)))]
    return Interval(point, lo, hi, n, "ratio bootstrap")


def sign_test_p(deltas: Sequence[float]) -> float:
    """Two-sided exact sign test on per-seed differences.

    Deliberately not a t-test. Per-seed recovery is skewed and n is small, which
    is where a t-test's normality assumption does the most damage. The sign test
    assumes almost nothing and costs a little power, which is the right trade
    when the alternative is a p-value nobody should believe.

    Ties are dropped, which is the standard treatment and is why the reported n
    can be smaller than the number of seeds.
    """
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2**n))
