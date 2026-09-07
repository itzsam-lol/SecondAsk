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

    # Online state.
    #
    # ``precision`` is the diagonal of a ridge design matrix: it starts at the
    # prior and accumulates the squared standardised feature values that have
    # actually been observed. It is what makes an uncertainty estimate possible
    # without storing or inverting a 113x113 matrix, which in pure Python would
    # cost more than the rest of the system combined.
    #
    # A diagonal approximation ignores correlation between features and so
    # understates uncertainty where features move together. That is the right
    # direction to be wrong in for a system that spends money on exploration:
    # it explores less than full LinUCB would, never more.
    precision: list[float] = field(default_factory=list)
    n_online: int = 0
    online_lr: float = 0.05

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0] * self.n_features
        if not self.mean:
            self.mean = [0.0] * self.n_features
        if not self.scale:
            self.scale = [1.0] * self.n_features
        if not self.precision:
            self.precision = [1.0] * self.n_features

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
        # Seed the precision diagonal from the batch so an online update does not
        # treat a model fitted on thousands of samples as though it knew nothing.
        for j in range(self.n_features):
            total = 0.0
            for row in Z:
                total += row[j] * row[j]
            self.precision[j] = 1.0 + total
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

    # -- online updating ----------------------------------------------------

    def uncertainty(self, row: Sequence[float]) -> float:
        """How little is known about this region of feature space.

        ``sqrt(sum(z_j^2 / precision_j))``, the diagonal analogue of the LinUCB
        confidence width. Large when the standardised features are unusual
        relative to what has been observed, and it shrinks as evidence
        accumulates, which is the property that makes exploration self-limiting.
        """
        if not self.fitted:
            return 1.0
        total = 0.0
        for j in range(self.n_features):
            z = (row[j] - self.mean[j]) / self.scale[j]
            total += (z * z) / max(1e-9, self.precision[j])
        return math.sqrt(max(0.0, total))

    def predict_optimistic(self, row: Sequence[float], alpha: float) -> float:
        """Prediction with an optimism bonus applied in logit space.

        Adding the bonus to the probability directly would be wrong: it would
        add the same absolute amount at p=0.02 and p=0.9, and near the ceiling it
        would push predictions past 1. In logit space the bonus is a shift in
        evidence, which is what an uncertainty width actually is.
        """
        if not self.fitted:
            return self.base_rate
        if alpha <= 0.0:
            return self.predict_one(row)
        z = self.bias
        for j in range(self.n_features):
            z += self.weights[j] * ((row[j] - self.mean[j]) / self.scale[j])
        return sigmoid(z + alpha * self.uncertainty(row))

    def partial_fit(self, row: Sequence[float], label: int, *, lr: float | None = None) -> float:
        """One SGD step on a single observed outcome. Returns the residual.

        Standardisation statistics are deliberately **frozen** at their batch
        values rather than updated online. Letting the mean and scale drift while
        the weights are expressed in terms of them silently rescales every
        existing coefficient, which shows up as a model that slowly forgets
        things nobody changed. Recalibrating those belongs in a retrain.

        The learning rate decays as ``1 / (1 + n_online / 500)``, so early
        feedback moves the model and later feedback refines it. Without decay a
        long-running process oscillates around the optimum forever.
        """
        if len(row) != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {len(row)}")
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")

        standardised = [(row[j] - self.mean[j]) / self.scale[j] for j in range(self.n_features)]
        z = self.bias
        for j in range(self.n_features):
            z += self.weights[j] * standardised[j]
        error = sigmoid(z) - label

        step = (lr if lr is not None else self.online_lr) / (1.0 + self.n_online / 500.0)
        self.bias -= step * error
        for j in range(self.n_features):
            self.weights[j] -= step * (error * standardised[j] + self.l2 * self.weights[j])
            self.precision[j] += standardised[j] * standardised[j]

        self.n_online += 1
        self.fitted = True
        return error

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
            "precision": [round(p, 4) for p in self.precision],
            "n_online": self.n_online,
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
        stored_precision = data.get("precision")
        if isinstance(stored_precision, list) and len(stored_precision) == model.n_features:
            model.precision = list(stored_precision)
        model.n_online = data.get("n_online", 0)
        return model
