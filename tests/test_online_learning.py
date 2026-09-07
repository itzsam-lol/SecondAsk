"""Online learning and the contextual bandit.

Three things have to hold at once, and the third is the one that is easy to lose:

1. the model actually adapts to new evidence,
2. uncertainty shrinks where evidence accumulates, so exploration is
   self-limiting rather than permanent,
3. **runs stay reproducible**. Online updates mutate the model mid-run, which is
   a closed loop: the model shapes the actions, the actions shape the outcomes,
   the outcomes shape the model. That is still deterministic given a seed, and
   it has to be, or the audit trail stops being replayable.
"""

from __future__ import annotations

import math
import random
import unittest
from datetime import datetime, timedelta

from secondask.underwrite import features as F
from secondask.underwrite.logreg import LogisticRegression, sigmoid
from secondask.underwrite.model import Underwriter
from secondask.world.entities import ActionKind
from secondask.world.generator import generate_world


def separable(n: int = 600, flip: bool = False) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(7)
    X = [[float(i % 2), rng.random()] for i in range(n)]
    y = [(1 - i % 2) if flip else (i % 2) for i in range(n)]
    return X, y


class PartialFitTest(unittest.TestCase):
    def fitted(self) -> LogisticRegression:
        X, y = separable()
        return LogisticRegression(n_features=2, epochs=30).fit(X, y)

    def test_a_single_step_moves_in_the_right_direction(self):
        model = self.fitted()
        before = model.predict_one([1.0, 0.5])
        for _ in range(50):
            model.partial_fit([1.0, 0.5], 0)
        self.assertLess(model.predict_one([1.0, 0.5]), before)

    def test_it_adapts_to_a_reversed_relationship(self):
        model = self.fitted()
        self.assertGreater(model.predict_one([1.0, 0.5]), 0.8)
        rng = random.Random(3)
        for _ in range(400):
            model.partial_fit([1.0, rng.random()], 0)
        self.assertLess(model.predict_one([1.0, 0.5]), 0.2)

    def test_the_learning_rate_decays(self):
        """Otherwise a long-running process oscillates around the optimum."""
        model = self.fitted()
        first = abs(model.partial_fit([1.0, 0.5], 0))
        for _ in range(2000):
            model.partial_fit([1.0, 0.5], 1)
        moved_late = model.weights[0]
        for _ in range(10):
            model.partial_fit([1.0, 0.5], 0)
        self.assertLess(abs(model.weights[0] - moved_late), abs(first))

    def test_standardisation_is_frozen(self):
        """Drifting mean/scale silently rescales every existing coefficient."""
        model = self.fitted()
        mean_before = list(model.mean)
        scale_before = list(model.scale)
        for _ in range(200):
            model.partial_fit([50.0, 50.0], 1)
        self.assertEqual(model.mean, mean_before)
        self.assertEqual(model.scale, scale_before)

    def test_bad_input_is_rejected(self):
        model = self.fitted()
        with self.assertRaises(ValueError):
            model.partial_fit([1.0], 1)
        with self.assertRaises(ValueError):
            model.partial_fit([1.0, 0.5], 2)
        with self.assertRaises(ValueError):
            model.partial_fit([1.0, 0.5], None)  # type: ignore[arg-type]

    def test_weights_stay_finite_under_heavy_updating(self):
        model = self.fitted()
        rng = random.Random(11)
        for _ in range(5000):
            model.partial_fit([rng.uniform(-8, 8), rng.uniform(-8, 8)], rng.randint(0, 1))
        self.assertTrue(all(math.isfinite(w) for w in model.weights))
        self.assertTrue(math.isfinite(model.bias))
        self.assertTrue(0.0 <= model.predict_one([1.0, 0.5]) <= 1.0)


class UncertaintyTest(unittest.TestCase):
    def fitted(self) -> LogisticRegression:
        X, y = separable()
        return LogisticRegression(n_features=2, epochs=30).fit(X, y)

    def test_novel_regions_are_more_uncertain(self):
        model = self.fitted()
        self.assertGreater(model.uncertainty([9.0, 9.0]), model.uncertainty([1.0, 0.5]) * 5)

    def test_uncertainty_shrinks_with_evidence(self):
        """Self-limiting exploration. If it did not shrink, it would explore forever."""
        model = self.fitted()
        point = [4.0, 4.0]
        before = model.uncertainty(point)
        for _ in range(300):
            model.partial_fit(point, 1)
        self.assertLess(model.uncertainty(point), before)

    def test_optimism_raises_the_estimate_and_stays_a_probability(self):
        model = self.fitted()
        plain = model.predict_one([1.0, 0.5])
        optimistic = model.predict_optimistic([1.0, 0.5], alpha=2.0)
        self.assertGreaterEqual(optimistic, plain)
        self.assertLessEqual(optimistic, 1.0)

    def test_zero_alpha_is_the_plain_prediction(self):
        model = self.fitted()
        self.assertAlmostEqual(model.predict_optimistic([1.0, 0.5], 0.0), model.predict_one([1.0, 0.5]))

    def test_bonus_is_applied_in_logit_space(self):
        """Added to the probability directly it would exceed 1 near the ceiling."""
        model = self.fitted()
        for alpha in (1.0, 5.0, 50.0):
            value = model.predict_optimistic([1.0, 0.5], alpha)
            self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(value, 0.0)

    def test_an_unfitted_model_is_maximally_uncertain(self):
        self.assertEqual(LogisticRegression(n_features=2).uncertainty([1.0, 1.0]), 1.0)


class PersistenceTest(unittest.TestCase):
    def test_online_state_survives_a_round_trip(self):
        X, y = separable()
        model = LogisticRegression(n_features=2, epochs=20).fit(X, y)
        for _ in range(120):
            model.partial_fit([1.0, 0.3], 0)
        restored = LogisticRegression.from_dict(model.to_dict())
        self.assertEqual(restored.n_online, model.n_online)
        self.assertAlmostEqual(restored.predict_one([1.0, 0.3]), model.predict_one([1.0, 0.3]), places=5)
        self.assertAlmostEqual(restored.uncertainty([1.0, 0.3]), model.uncertainty([1.0, 0.3]), places=4)

    def test_precision_is_seeded_from_the_batch(self):
        """A model fitted on thousands of rows must not look uninformed."""
        X, y = separable(800)
        model = LogisticRegression(n_features=2, epochs=10).fit(X, y)
        self.assertTrue(all(p > 1.0 for p in model.precision))


class UnderwriterOnlineTest(unittest.TestCase):
    def setUp(self):
        import os

        path = os.path.join("models", "underwriter.json")
        if not os.path.exists(path):
            raise unittest.SkipTest("no fitted model on disk; run 'python -m secondask train'")
        self.underwriter = Underwriter.load(path)
        self.world = generate_world(seed=33, n_items=40, horizon_days=14)
        self.when = self.world.start + timedelta(days=2)

    def observe(self, action, success, times=1):
        item = self.world.items[0]
        customer = self.world.customers[item.customer_id]
        applied = False
        for _ in range(times):
            applied = self.underwriter.observe(
                item, customer, self.when, self.world.downtime, self.world.bank_of(item), action, success
            )
        return item, customer, applied

    def test_observing_moves_the_prediction(self):
        action = ActionKind.PAYMENT_LINK_SMS
        item, customer, _ = self.observe(action, False, 0)
        before = self.underwriter.p_recover(
            item, customer, self.when, self.world.downtime, self.world.bank_of(item), action
        )
        self.observe(action, True, 250)
        after = self.underwriter.p_recover(
            item, customer, self.when, self.world.downtime, self.world.bank_of(item), action
        )
        self.assertGreater(after, before)

    def test_unfitted_actions_are_not_bootstrapped(self):
        """SILENT_RETRY has no training data. Live outcomes must not invent one."""
        item, customer, applied = self.observe(ActionKind.SILENT_RETRY, True, 5)
        self.assertFalse(applied)
        self.assertEqual(
            self.underwriter.p_recover(
                item, customer, self.when, self.world.downtime,
                self.world.bank_of(item), ActionKind.SILENT_RETRY,
            ),
            0.0,
        )

    def test_predictions_stay_bounded_after_updates(self):
        action = ActionKind.PAYMENT_LINK_WHATSAPP
        self.observe(action, True, 500)
        item = self.world.items[0]
        p = self.underwriter.p_recover(
            item, self.world.customers[item.customer_id], self.when,
            self.world.downtime, self.world.bank_of(item), action,
        )
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 0.97)

    def test_online_report_counts_updates(self):
        self.observe(ActionKind.PAYMENT_LINK_SMS, True, 10)
        report = self.underwriter.online_report()
        self.assertGreaterEqual(report["total_online_updates"], 10)


class DeterminismTest(unittest.TestCase):
    """Online learning must not cost reproducibility."""

    def test_the_same_sequence_gives_the_same_model(self):
        outcomes = [(i % 3 == 0) for i in range(300)]

        def train_one():
            X, y = separable()
            model = LogisticRegression(n_features=2, epochs=20).fit(X, y)
            for i, success in enumerate(outcomes):
                model.partial_fit([float(i % 2), (i % 7) / 7.0], 1 if success else 0)
            return model

        a, b = train_one(), train_one()
        self.assertEqual(a.weights, b.weights)
        self.assertEqual(a.bias, b.bias)
        self.assertEqual(a.precision, b.precision)

    def test_feature_extraction_is_stable(self):
        """The online path recomputes features at observation time."""
        world = generate_world(seed=44, n_items=20, horizon_days=14)
        item = world.items[0]
        when = world.start + timedelta(days=1)
        first = F.extract(item, world.customers[item.customer_id], when, world.downtime, world.bank_of(item))
        second = F.extract(item, world.customers[item.customer_id], when, world.downtime, world.bank_of(item))
        self.assertEqual(first, second)


class FeatureInteractionTest(unittest.TestCase):
    def test_crosses_are_present_and_one_hot(self):
        world = generate_world(seed=55, n_items=30, horizon_days=14)
        item = world.items[0]
        vector = F.extract(
            item, world.customers[item.customer_id], world.start + timedelta(days=1),
            world.downtime, world.bank_of(item),
        )
        names = F.FEATURE_NAMES
        for prefix, expected_hot in (("x:", 1), ("family=", 1), ("srcclass=", 1), ("tod=", 1)):
            block = [v for n, v in zip(names, vector) if n.startswith(prefix)]
            self.assertTrue(block, f"no features with prefix {prefix}")
            if prefix != "x:":
                self.assertEqual(sum(block), expected_hot, f"{prefix} is not one-hot")
        # Three separate crosses, each contributing exactly one hot cell.
        crosses = [v for n, v in zip(names, vector) if n.startswith("x:")]
        self.assertEqual(sum(crosses), 3.0)

    def test_method_family_partitions_the_methods(self):
        from secondask.world.entities import Method

        for method in Method:
            self.assertIn(F.method_family(method), F.METHOD_FAMILIES)

    def test_time_buckets_align_with_the_contact_window(self):
        """night is exactly the forbidden span, so the bucket carries legality."""
        self.assertEqual(F.time_bucket(7.99), "night")
        self.assertEqual(F.time_bucket(8.0), "morning")
        self.assertEqual(F.time_bucket(18.99), "evening")
        self.assertEqual(F.time_bucket(19.0), "night")

    def test_names_and_vector_stay_aligned(self):
        world = generate_world(seed=66, n_items=25, horizon_days=14)
        for item in world.items:
            vector = F.extract(
                item, world.customers[item.customer_id], world.start + timedelta(hours=30),
                world.downtime, world.bank_of(item),
            )
            self.assertEqual(len(vector), len(F.FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
