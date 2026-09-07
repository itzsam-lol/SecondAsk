"""Calibration measurement.

Discrimination and calibration are different properties and this system needs the
second one. AUC says whether the model ranks a good action above a bad one.
Calibration says whether "0.3" means it happens three times in ten.

The underwriter multiplies its probability by a rupee amount to get an expected
value, and then compares that number against the cost of a message and against
other items competing for the same contact budget. A model that ranks perfectly
but reports 0.9 whenever it means 0.3 would order actions correctly and price
every one of them at triple its worth, and the budget allocation built on top
would be nonsense. So calibration is the property that has to be reported, and
it is reported on held-out seeds.

Metrics:

``Brier score``
    Mean squared error of the probabilities. Lower is better. Reported against
    the base-rate baseline, because a Brier score alone is uninterpretable.

``ECE``
    Expected calibration error: average gap between predicted and observed
    frequency across bins, weighted by bin population.

``Reliability curve``
    The bins themselves, so the shape of any miscalibration is visible rather
    than compressed into one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Bin:
    lower: float
    upper: float
    count: int = 0
    predicted_sum: float = 0.0
    observed_sum: int = 0

    @property
    def mean_predicted(self) -> float:
        return self.predicted_sum / self.count if self.count else 0.0

    @property
    def mean_observed(self) -> float:
        return self.observed_sum / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "range": [round(self.lower, 3), round(self.upper, 3)],
            "count": self.count,
            "predicted": round(self.mean_predicted, 4),
            "observed": round(self.mean_observed, 4),
        }


@dataclass
class CalibrationReport:
    n: int = 0
    brier: float = 0.0
    brier_baseline: float = 0.0
    ece: float = 0.0
    max_ce: float = 0.0
    auc: float = 0.0
    base_rate: float = 0.0
    bins: list[Bin] = field(default_factory=list)

    @property
    def brier_skill(self) -> float:
        """1 minus the ratio to the base-rate model. Above zero beats the prior."""
        if self.brier_baseline <= 0:
            return 0.0
        return 1.0 - (self.brier / self.brier_baseline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "base_rate": round(self.base_rate, 4),
            "brier": round(self.brier, 5),
            "brier_baseline": round(self.brier_baseline, 5),
            "brier_skill": round(self.brier_skill, 4),
            "ece": round(self.ece, 4),
            "max_calibration_error": round(self.max_ce, 4),
            "auc": round(self.auc, 4),
            "bins": [b.to_dict() for b in self.bins if b.count > 0],
        }


def evaluate(predictions: Sequence[float], outcomes: Sequence[int], *, n_bins: int = 10) -> CalibrationReport:
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes differ in length")
    report = CalibrationReport(n=len(predictions))
    if not predictions:
        return report

    n = len(predictions)
    base = sum(outcomes) / n
    report.base_rate = base
    report.brier = sum((p - y) ** 2 for p, y in zip(predictions, outcomes)) / n
    report.brier_baseline = sum((base - y) ** 2 for y in outcomes) / n

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [Bin(edges[i], edges[i + 1]) for i in range(n_bins)]
    for p, y in zip(predictions, outcomes):
        # Clamp so that p == 1.0 lands in the last bin rather than out of range.
        index = min(n_bins - 1, max(0, int(p * n_bins)))
        b = bins[index]
        b.count += 1
        b.predicted_sum += p
        b.observed_sum += y

    ece = 0.0
    max_ce = 0.0
    for b in bins:
        if not b.count:
            continue
        gap = abs(b.mean_predicted - b.mean_observed)
        ece += (b.count / n) * gap
        max_ce = max(max_ce, gap)
    report.ece = ece
    report.max_ce = max_ce
    report.bins = bins
    report.auc = auc_score(predictions, outcomes)
    return report


def auc_score(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
    """AUC via the rank-sum identity, with ties given half credit.

    O(n log n) rather than the O(n^2) pair enumeration, which matters because
    this runs over tens of thousands of held-out samples.
    """
    pairs = sorted(zip(predictions, outcomes), key=lambda p: p[0])
    n_pos = sum(o for _, o in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        # Undefined rather than zero. A degenerate label set is a fact about the
        # data, and returning 0.5 says "no information" honestly.
        return 0.5

    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1

    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
