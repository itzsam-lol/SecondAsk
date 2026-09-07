"""Logistic regression, pure Python.

Written out rather than imported because the project holds a zero-dependency
line, and because a reviewer should be able to see exactly what the "model" in
this system is. It is a linear model with an L2 penalty, fitted by mini-batch
SGD. That is a deliberate choice, not a limitation:

* It is **calibrated**. The underwriter needs a probability that can be
  multiplied by a rupee amount to give an expected value. A model that ranks well
  but is poorly calibrated would rank actions correctly and price them wrongly,
  and pricing is the entire point. Reliability curves are reported.
* It is **inspectable**. Every coefficient can be read, and the README quotes
  several of them, because a claim like "error_source carries real signal" should
  be checkable rather than asserted.
* It is **enough**. A gradient-boosted ensemble would improve AUC and would not
  change a single decision, because the decisions turn on large, structural
  effects rather than on fine gradients.

Numerical care taken:

* Features are standardised using training-set statistics only, and those
  statistics are stored with the model. Standardising with test data included is
  the most common way a benchmark quietly leaks.
* A zero-variance feature gets scale 1.0 rather than dividing by zero.
* The sigmoid is written in its two-branch form so a large negative input cannot
  overflow ``exp``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence


def sigmoid(z: float) -> float:
    """Overflow-safe logistic.

    ``exp(710)`` overflows a float64. During early training a poorly scaled
    feature can easily produce a logit that large, and the naive form turns a
    recoverable optimisation problem into an OverflowError.
    """
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticRegression:
    n_features: int
    l2: float = 1e-3
    learning_rate: float = 0.15
    epochs: int = 24
    batch_size: int = 64
    seed: int = 0

    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    fitted: bool = False
    n_train: int = 0
    base_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0] * self.n_features
        if not self.mean:
            self.mean = [0.0] * self.n_features
        if not self.scale:
            self.scale = [1.0] * self.n_features

    # -- fitting ------------------------------------------------------------

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "LogisticRegression":
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if not X:
            # An action that exploration never sampled. Leave the model at the
            # base rate rather than pretending to have learned something.
            self.fitted = False
            return self

        n = len(X)
        self.n_train = n
        self.base_rate = sum(y) / n

        self._compute_standardisation(X)
        Z = [self._standardise(row) for row in X]

        # A degenerate target (all successes or all failures) has no gradient
        # signal. Fall back to predicting the base rate through the bias term.
        positives = sum(y)
        if positives == 0 or positives == n:
            self.weights = [0.0] * self.n_features
            eps = 1.0 / (n + 2.0)
            p = min(1.0 - eps, max(eps, self.base_rate))
            self.bias = math.log(p / (1.0 - p))
            self.fitted = True
            return self

        rng = random.Random(self.seed)
        indices = list(range(n))
        self.weights = [0.0] * self.n_features
        self.bias = 0.0

        for epoch in range(self.epochs):
            rng.shuffle(indices)
            # Decay the step size so late epochs refine rather than oscillate.
            lr = self.learning_rate / (1.0 + 0.35 * epoch)
            for start in range(0, n, self.batch_size):
                batch = indices[start : start + self.batch_size]
                if not batch:
                    continue
                grad_w = [0.0] * self.n_features
                grad_b = 0.0
                for i in batch:
                    row = Z[i]
                    z = self.bias
                    for j, value in enumerate(row):
                        z += self.weights[j] * value
                    error = sigmoid(z) - y[i]
                    grad_b += error
                    for j, value in enumerate(row):
                        grad_w[j] += error * value
                inv = 1.0 / len(batch)
                self.bias -= lr * grad_b * inv
                for j in range(self.n_features):
                    # L2 applies to weights only. Penalising the bias would drag
                    # predictions toward 0.5 rather than toward the base rate.
                    self.weights[j] -= lr * (grad_w[j] * inv + self.l2 * self.weights[j])

        self.fitted = True
        return self

    def _compute_standardisation(self, X: Sequence[Sequence[float]]) -> None:
        n = len(X)
        self.mean = [0.0] * self.n_features
        self.scale = [1.0] * self.n_features
        for j in range(self.n_features):
            total = 0.0
            for row in X:
                total += row[j]
            mu = total / n
            var = 0.0
            for row in X:
                diff = row[j] - mu
                var += diff * diff
            sd = math.sqrt(var / n) if n else 0.0
            self.mean[j] = mu
            # A constant feature carries no information. Scale 1.0 keeps it at
            # zero after centring instead of producing inf.
            self.scale[j] = sd if sd > 1e-9 else 1.0

    def _standardise(self, row: Sequence[float]) -> list[float]:
        return [(row[j] - self.mean[j]) / self.scale[j] for j in range(self.n_features)]

    # -- prediction ---------------------------------------------------------

    def predict_one(self, row: Sequence[float]) -> float:
        if len(row) != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {len(row)}")
        if not self.fitted:
            return self.base_rate
        z = self.bias
        for j in range(self.n_features):
            z += self.weights[j] * ((row[j] - self.mean[j]) / self.scale[j])
        return sigmoid(z)

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_one(row) for row in X]

    def top_coefficients(self, names: Sequence[str], k: int = 12) -> list[tuple[str, float]]:
        """Largest absolute standardised coefficients.

        Comparable across features precisely because inputs were standardised,
        so these read as "effect per standard deviation".
        """
        pairs = list(zip(names, self.weights))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "l2": self.l2,
            "weights": [round(w, 6) for w in self.weights],
            "bias": round(self.bias, 6),
            "mean": [round(m, 6) for m in self.mean],
            "scale": [round(s, 6) for s in self.scale],
            "fitted": self.fitted,
            "n_train": self.n_train,
            "base_rate": round(self.base_rate, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogisticRegression":
        model = cls(n_features=data["n_features"], l2=data.get("l2", 1e-3))
        model.weights = list(data["weights"])
        model.bias = data["bias"]
        model.mean = list(data["mean"])
        model.scale = list(data["scale"])
        model.fitted = data.get("fitted", True)
        model.n_train = data.get("n_train", 0)
        model.base_rate = data.get("base_rate", 0.0)
        return model
