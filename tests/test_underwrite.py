"""Underwriter, calibration and planner tests."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

from secondask.underwrite import calibration
from secondask.underwrite.logreg import LogisticRegression, sigmoid
from secondask.underwrite.model import Underwriter
from secondask.underwrite.planner import (
    ANNOYANCE_BUDGET,
    ANNOYANCE_BUDGET_CAP,
    Candidate,
    annoyance_budget_for,
    available_actions,
    candidate_times,
    goodwill_cost,
    plan,
)
from secondask.underwrite import features as F
from secondask.world.entities import ActionKind, Channel, Customer, ErrorReason, Method
from secondask.world.generator import generate_world


class SigmoidTest(unittest.TestCase):
    def test_does_not_overflow(self):
        """The naive form raises OverflowError past about -710."""
        self.assertAlmostEqual(sigmoid(-10000.0), 0.0, places=10)
        self.assertAlmostEqual(sigmoid(10000.0), 1.0, places=10)
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_is_monotonic(self):
        values = [sigmoid(z) for z in range(-20, 21)]
        self.assertEqual(values, sorted(values))


class LogisticRegressionTest(unittest.TestCase):
    def test_learns_a_separable_signal(self):
        X = [[float(i % 2), 1.0] for i in range(400)]
        y = [i % 2 for i in range(400)]
        model = LogisticRegression(n_features=2, epochs=40).fit(X, y)
        self.assertGreater(model.predict_one([1.0, 1.0]), 0.8)
        self.assertLess(model.predict_one([0.0, 1.0]), 0.2)

    def test_constant_feature_does_not_divide_by_zero(self):
        X = [[1.0, float(i % 2)] for i in range(100)]
        y = [i % 2 for i in range(100)]
        model = LogisticRegression(n_features=2).fit(X, y)
        self.assertTrue(all(math.isfinite(w) for w in model.weights))
        self.assertTrue(math.isfinite(model.predict_one([1.0, 1.0])))

    def test_degenerate_labels_fall_back_to_the_base_rate(self):
        X = [[float(i), 1.0] for i in range(50)]
        model = LogisticRegression(n_features=2).fit(X, [0] * 50)
        self.assertLess(model.predict_one([10.0, 1.0]), 0.1)
        model = LogisticRegression(n_features=2).fit(X, [1] * 50)
        self.assertGreater(model.predict_one([10.0, 1.0]), 0.9)

    def test_empty_training_set_is_not_fitted(self):
        model = LogisticRegression(n_features=3).fit([], [])
        self.assertFalse(model.fitted)
        self.assertEqual(model.predict_one([0.0, 0.0, 0.0]), 0.0)

    def test_wrong_feature_count_raises(self):
        model = LogisticRegression(n_features=3)
        with self.assertRaises(ValueError):
            model.predict_one([1.0, 2.0])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            LogisticRegression(n_features=1).fit([[1.0], [2.0]], [1])

    def test_round_trip_through_json(self):
        X = [[float(i % 3), float(i % 5)] for i in range(200)]
        y = [1 if (i % 3) else 0 for i in range(200)]
        model = LogisticRegression(n_features=2).fit(X, y)
        restored = LogisticRegression.from_dict(model.to_dict())
        for row in X[:20]:
            self.assertAlmostEqual(model.predict_one(row), restored.predict_one(row), places=5)


class CalibrationTest(unittest.TestCase):
    def test_perfect_predictions_score_zero_brier(self):
        report = calibration.evaluate([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0])
        self.assertAlmostEqual(report.brier, 0.0)
        self.assertAlmostEqual(report.auc, 1.0)

    def test_auc_of_a_reversed_ranking(self):
        self.assertAlmostEqual(calibration.auc_score([0.9, 0.1], [0, 1]), 0.0)

    def test_auc_with_ties_is_half(self):
        self.assertAlmostEqual(calibration.auc_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.5)

    def test_auc_is_defined_when_one_class_is_absent(self):
        """Degenerate labels are a fact about the data, not a crash."""
        self.assertEqual(calibration.auc_score([0.2, 0.8], [1, 1]), 0.5)
        self.assertEqual(calibration.auc_score([0.2, 0.8], [0, 0]), 0.5)

    def test_probability_of_one_lands_in_the_last_bin(self):
        report = calibration.evaluate([1.0] * 10, [1] * 10, n_bins=10)
        self.assertEqual(sum(b.count for b in report.bins), 10)

    def test_empty_input_does_not_raise(self):
        self.assertEqual(calibration.evaluate([], []).n, 0)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            calibration.evaluate([0.5], [1, 0])

    def test_a_well_calibrated_model_has_low_ece(self):
        import random

        rng = random.Random(4)
        predictions, outcomes = [], []
        for _ in range(20000):
            p = rng.random()
            predictions.append(p)
            outcomes.append(1 if rng.random() < p else 0)
        report = calibration.evaluate(predictions, outcomes)
        self.assertLess(report.ece, 0.02)


class FeatureTest(unittest.TestCase):
    def test_vector_length_matches_the_declared_names(self):
        world = generate_world(seed=5, n_items=30, horizon_days=14)
        item = world.items[0]
        vector = F.extract(
            item, world.customers[item.customer_id], world.start + timedelta(days=1),
            world.downtime, world.bank_of(item),
        )
        self.assertEqual(len(vector), len(F.FEATURE_NAMES))
        self.assertEqual(len(vector), F.N_FEATURES)

    def test_all_values_are_finite(self):
        world = generate_world(seed=6, n_items=100, horizon_days=14)
        when = world.start + timedelta(days=2)
        for item in world.items:
            vector = F.extract(item, world.customers[item.customer_id], when, world.downtime, world.bank_of(item))
            for name, value in zip(F.FEATURE_NAMES, vector):
                self.assertTrue(math.isfinite(value), f"{name} is {value}")

    def test_unknown_reason_does_not_raise(self):
        self.assertEqual(F.reason_class(ErrorReason.UNKNOWN), "other")

    def test_zero_amount_is_safe(self):
        world = generate_world(seed=7, n_items=10, horizon_days=14)
        item = world.items[0]
        item.recovered_paise = item.amount_paise  # outstanding becomes zero
        vector = F.extract(
            item, world.customers[item.customer_id], world.start + timedelta(days=1),
            world.downtime, world.bank_of(item),
        )
        self.assertTrue(all(math.isfinite(v) for v in vector))


class BudgetTest(unittest.TestCase):
    def _item(self, world, paise):
        item = world.items[0]
        item.amount_paise = paise
        item.recovered_paise = 0
        return item

    def test_budget_grows_with_the_amount(self):
        world = generate_world(seed=8, n_items=5, horizon_days=14)
        small = annoyance_budget_for(self._item(world, 20000))       # Rs 200
        medium = annoyance_budget_for(self._item(world, 500000))     # Rs 5,000
        large = annoyance_budget_for(self._item(world, 15000000))    # Rs 1.5 lakh
        self.assertLess(small, medium)
        self.assertLess(medium, large)

    def test_budget_is_hard_capped(self):
        """No amount of money buys unlimited contact."""
        world = generate_world(seed=8, n_items=5, horizon_days=14)
        enormous = annoyance_budget_for(self._item(world, 10_000_000_000))
        self.assertLessEqual(enormous, ANNOYANCE_BUDGET_CAP)

    def test_small_items_keep_the_base_budget(self):
        world = generate_world(seed=8, n_items=5, horizon_days=14)
        self.assertAlmostEqual(annoyance_budget_for(self._item(world, 5000)), ANNOYANCE_BUDGET, places=6)

    def test_goodwill_escalates_with_annoyance(self):
        calm = Customer(customer_id="c", annoyance=0.0)
        irritated = Customer(customer_id="c", annoyance=3.0)
        self.assertLess(
            goodwill_cost(ActionKind.PAYMENT_LINK_SMS, calm),
            goodwill_cost(ActionKind.PAYMENT_LINK_SMS, irritated),
        )

    def test_non_contact_actions_cost_no_goodwill(self):
        customer = Customer(customer_id="c", annoyance=2.0)
        self.assertEqual(goodwill_cost(ActionKind.MANDATE_DEBIT, customer), 0)


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.world = generate_world(seed=9, n_items=60, horizon_days=21)
        self.underwriter = Underwriter()

    def test_settled_items_get_no_plan(self):
        item = self.world.items[0]
        item.recovered_paise = item.amount_paise
        result = plan(
            self.underwriter, item, self.world.customers[item.customer_id],
            self.world.start, self.world.downtime, self.world.bank_of(item), self.world.end,
        )
        self.assertIsNone(result.best)
        self.assertIn("nothing outstanding", result.reason)

    def test_an_untrained_underwriter_proposes_nothing(self):
        """No evidence must mean no action, not a confident guess."""
        item = self.world.items[0]
        result = plan(
            self.underwriter, item, self.world.customers[item.customer_id],
            self.world.start, self.world.downtime, self.world.bank_of(item), self.world.end,
        )
        self.assertIsNone(result.best)

    def test_mandate_debit_only_offered_on_mandate_rails(self):
        for item in self.world.items:
            customer = self.world.customers[item.customer_id]
            actions = available_actions(item, customer, set())
            if not item.method.is_mandate:
                self.assertNotIn(ActionKind.MANDATE_DEBIT, actions, item.method.value)

    def test_unavailable_channels_are_excluded(self):
        item = self.world.items[0]
        customer = Customer(customer_id="c", has_phone=False, has_email=False, has_whatsapp=False)
        actions = available_actions(item, customer, set())
        for action in actions:
            self.assertNotIn(
                action,
                (ActionKind.PAYMENT_LINK_SMS, ActionKind.PAYMENT_LINK_WHATSAPP,
                 ActionKind.PAYMENT_LINK_EMAIL, ActionKind.VOICE_CALL),
            )

    def test_already_escalated_items_are_excluded(self):
        item = self.world.items[0]
        item.escalated_at = self.world.start
        actions = available_actions(item, self.world.customers[item.customer_id], set())
        self.assertNotIn(ActionKind.HUMAN_ESCALATION, actions)

    def test_candidate_times_are_inside_the_horizon(self):
        item = self.world.items[0]
        times = candidate_times(item, self.world.start, self.world.downtime, self.world.bank_of(item), self.world.end)
        self.assertTrue(times)
        for when in times:
            self.assertGreaterEqual(when, self.world.start)
            self.assertLess(when, self.world.end)
        self.assertEqual(times, sorted(times), "candidate times must be ordered for stable tie-breaks")

    def test_candidate_times_are_unique(self):
        item = self.world.items[0]
        times = candidate_times(item, self.world.start, self.world.downtime, self.world.bank_of(item), self.world.end)
        self.assertEqual(len(times), len(set(times)))


class TrainedPlannerTest(unittest.TestCase):
    """Behaviour of the fitted model, if one is on disk."""

    @classmethod
    def setUpClass(cls):
        import os

        path = os.path.join("models", "underwriter.json")
        if not os.path.exists(path):
            raise unittest.SkipTest("no fitted model on disk; run 'python -m secondask train'")
        cls.underwriter = Underwriter.load(path)

    def test_probabilities_are_bounded(self):
        world = generate_world(seed=31, n_items=80, horizon_days=21)
        when = world.start + timedelta(days=2)
        for item in world.items:
            customer = world.customers[item.customer_id]
            for action in (ActionKind.PAYMENT_LINK_SMS, ActionKind.HUMAN_ESCALATION, ActionKind.VOICE_CALL):
                p = self.underwriter.p_recover(item, customer, when, world.downtime, world.bank_of(item), action)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_silent_retry_on_a_one_off_rail_is_worthless(self):
        """No Indian one-off rail carries a standing authorisation."""
        world = generate_world(seed=32, n_items=60, horizon_days=21)
        when = world.start + timedelta(days=1)
        for item in world.items:
            if item.method.is_mandate:
                continue
            p = self.underwriter.p_recover(
                item, world.customers[item.customer_id], when, world.downtime,
                world.bank_of(item), ActionKind.SILENT_RETRY,
            )
            self.assertEqual(p, 0.0)


if __name__ == "__main__":
    unittest.main()
