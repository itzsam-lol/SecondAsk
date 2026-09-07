"""The underwriter.

One calibrated logistic regression per action kind, answering a single question:

    given what we can observe, what is the probability that *this* action, taken
    at *this* time, recovers the money?

Fitting a separate model per action rather than one model with the action as a
feature is deliberate. The effects that matter are almost entirely interactions
between the action and the state: a retry is worthless on a dead card and
excellent after an outage closes, and a linear model with an action one-hot
cannot express that at all. Per-action models get every interaction for free and
stay individually readable.

Training discipline, which is the part that makes the reported numbers mean
something:

* Training worlds and evaluation worlds use **disjoint seeds**. ``assert_disjoint``
  enforces it rather than trusting a convention.
* Only observable features are used. Latent state is unreachable from here.
* The simulator's parameters are never read by this module. The model learns the
  same way it would in production, from outcomes.

An action the exploration policy never sampled falls back to its base rate. That
is honest: the correct output for "no evidence" is the prior, not a confident
guess.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from ..world.downtime import DowntimeFeed
from ..world.entities import ActionKind, Customer, RiskItem
from . import features as F
from .logreg import LogisticRegression

MODELABLE_ACTIONS: tuple[ActionKind, ...] = (
    ActionKind.SILENT_RETRY,
    ActionKind.MANDATE_DEBIT,
    ActionKind.PAYMENT_LINK_SMS,
    ActionKind.PAYMENT_LINK_WHATSAPP,
    ActionKind.PAYMENT_LINK_EMAIL,
    ActionKind.UPDATE_INSTRUMENT,
    ActionKind.INCENTIVE_OFFER,
    ActionKind.VOICE_CALL,
    ActionKind.HUMAN_ESCALATION,
)


class SeedLeakage(AssertionError):
    """Raised when training and evaluation seeds overlap."""


def assert_disjoint(train_seeds: Sequence[int], eval_seeds: Sequence[int]) -> None:
    overlap = sorted(set(train_seeds) & set(eval_seeds))
    if overlap:
        raise SeedLeakage(
            f"training and evaluation seeds overlap on {overlap}; "
            "every reported number would be measured on data the model was fitted to"
        )


@dataclass
class Underwriter:
    models: dict[str, LogisticRegression] = field(default_factory=dict)
    train_seeds: list[int] = field(default_factory=list)
    n_samples: int = 0
    version: str = "2"
    n_online: int = 0

    # -- prediction ---------------------------------------------------------

    def p_recover(
        self,
        item: RiskItem,
        customer: Customer,
        now: datetime,
        downtime: DowntimeFeed,
        bank: str,
        action: ActionKind,
        *,
        optimism: float = 0.0,
    ) -> float:
        model = self.models.get(action.value)
        if model is None or not model.fitted:
            # No evidence means zero, not a small positive.
            #
            # The clamp below has a floor, and applying it here would turn an
            # action the exploration policy never sampled into one worth
            # 0.0005 of the balance. On a one lakh rupee item that is fifty
            # rupees of expected value for an action that costs nothing, so the
            # planner would propose it, the policy engine would refuse it, and
            # the proposal budget for that visit would be spent on an action
            # that cannot ever happen.
            return 0.0
        vector = F.extract(item, customer, now, downtime, bank)
        p = model.predict_optimistic(vector, optimism) if optimism > 0 else model.predict_one(vector)
        # Clamp away from the endpoints. A probability of exactly 1.0 would let
        # a single action dominate every budget comparison on the strength of
        # what is really just an unpenalised extrapolation.
        return min(0.97, max(0.0005, p))

    def p_recover_batch(
        self,
        item: RiskItem,
        customer: Customer,
        times: Sequence[datetime],
        downtime: DowntimeFeed,
        bank: str,
        action: ActionKind,
        *,
        optimism: float = 0.0,
    ) -> list[float]:
        model = self.models.get(action.value)
        if model is None or not model.fitted:
            return [0.0] * len(times)
        out = []
        for now in times:
            vector = F.extract(item, customer, now, downtime, bank)
            p = model.predict_optimistic(vector, optimism) if optimism > 0 else model.predict_one(vector)
            out.append(min(0.97, max(0.0005, p)))
        return out

    # -- online learning ----------------------------------------------------

    def observe(
        self,
        item: RiskItem,
        customer: Customer,
        now: datetime,
        downtime: DowntimeFeed,
        bank: str,
        action: ActionKind,
        success: bool,
    ) -> bool:
        """Fold one observed outcome into the model. Returns whether it applied.

        This is what makes the agent adaptive without a retraining run: the
        features are recomputed exactly as they were at decision time, and the
        realised outcome is used for a single gradient step.

        Only actions that have a fitted model are updated. An action the
        exploration policy never sampled stays at zero rather than being
        bootstrapped from a handful of live observations, because a model built
        from three successes would be confidently wrong and would then be used
        to price real money.

        Updating is deterministic given the sequence of outcomes, so a run with
        online learning enabled still reproduces exactly from its seed. It does
        mean the model at the end of a run differs from the model at the start,
        which is why it is opt-in and why the benchmark leaves it off.
        """
        model = self.models.get(action.value)
        if model is None or not model.fitted:
            return False
        vector = F.extract(item, customer, now, downtime, bank)
        model.partial_fit(vector, 1 if success else 0)
        self.n_online += 1
        return True

    def online_report(self) -> dict[str, Any]:
        return {
            "total_online_updates": self.n_online,
            "per_action": {
                k: {"n_online": m.n_online, "n_train": m.n_train}
                for k, m in sorted(self.models.items())
                if m.n_online
            },
        }

    # -- training -----------------------------------------------------------

    def fit(self, samples: dict[str, tuple[list[list[float]], list[int]]], *, seed: int = 0) -> "Underwriter":
        total = 0
        for action_value, (X, y) in samples.items():
            model = LogisticRegression(n_features=F.N_FEATURES, seed=seed)
            model.fit(X, y)
            self.models[action_value] = model
            total += len(X)
        self.n_samples = total
        return self

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": self.version, "train_seeds": self.train_seeds, "n_samples": self.n_samples}
        per_action = {}
        for action_value, model in sorted(self.models.items()):
            per_action[action_value] = {
                "n_train": model.n_train,
                "base_rate": round(model.base_rate, 4),
                "fitted": model.fitted,
                "top_coefficients": [
                    [name, round(weight, 4)] for name, weight in model.top_coefficients(F.FEATURE_NAMES, 8)
                ],
            }
        out["actions"] = per_action
        return out

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "train_seeds": self.train_seeds,
            "n_samples": self.n_samples,
            "n_online": self.n_online,
            "feature_names": F.FEATURE_NAMES,
            "models": {k: m.to_dict() for k, m in self.models.items()},
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Underwriter":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        stored = data.get("feature_names")
        # A model fitted against a different feature layout would silently
        # produce nonsense, since position is all a linear model knows.
        if stored is not None and stored != F.FEATURE_NAMES:
            raise ValueError(
                "stored model was fitted with a different feature set; retrain with "
                "'python -m secondask train'"
            )
        underwriter = cls(
            train_seeds=data.get("train_seeds", []),
            n_samples=data.get("n_samples", 0),
            version=data.get("version", "1"),
            n_online=data.get("n_online", 0),
        )
        underwriter.models = {k: LogisticRegression.from_dict(v) for k, v in data["models"].items()}
        return underwriter
