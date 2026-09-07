"""Integrity tests.

The properties in this file are the ones whose failure would invalidate every
number the project reports, and which would fail silently if nothing checked
them. They are worth more than the unit tests.

* **No latent leakage.** A simulator-trained model that peeks at hidden state
  produces excellent benchmark numbers and is worth nothing. The check is
  destructive: corrupt every latent field and assert the feature vectors are
  byte-identical.
* **Determinism.** Same seed, same ledger chain head. Without this, "reproduce
  the numbers" is not a claim anybody can act on.
* **Common random numbers.** Two agents taking the same action at the same time
  must observe the same draw, or agent comparison is measuring luck.
* **Idempotency.** A retried call after a timeout must not create a second
  payment link.
* **Train/test separation.** Enforced, not documented.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from secondask.eval.harness import CONFIGS, run_agent
from secondask.execute.circuit import BreakerState, CircuitBreaker, CircuitOpen, backoff_delay
from secondask.execute.razorpay_client import RazorpayClient, RazorpayError
from secondask.underwrite import features as F
from secondask.underwrite.model import SeedLeakage, assert_disjoint
from secondask.world.entities import ActionKind, Blocker
from secondask.world.generator import generate_world
from secondask.world.outcomes import execute_counterfactual


class NoLatentLeakageTest(unittest.TestCase):
    def test_features_ignore_latent_state(self):
        """Corrupt every hidden field; the observable features must not move."""
        world = generate_world(seed=41, n_items=120, horizon_days=14)
        now = world.start + timedelta(days=3)

        before = [
            F.extract(item, world.customers[item.customer_id], now, world.downtime, world.bank_of(item))
            for item in world.items
        ]

        for item in world.items:
            item._blocker = Blocker.UNREACHABLE
            item._resolves_at = None
        for customer in world.customers.values():
            customer._responsiveness = 0.0
            customer._annoyance_tolerance = 0.0
            customer._payday_day = 0

        after = [
            F.extract(item, world.customers[item.customer_id], now, world.downtime, world.bank_of(item))
            for item in world.items
        ]
        self.assertEqual(before, after, "a feature is reading latent simulator state")

    def test_observable_projection_is_an_allow_list(self):
        world = generate_world(seed=42, n_items=10, horizon_days=7)
        observable = world.items[0].observable()
        for key in observable:
            self.assertFalse(key.startswith("_"), f"latent field {key} escaped into observable()")
        self.assertNotIn("_blocker", observable)
        self.assertNotIn("_resolves_at", observable)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_gives_the_same_world(self):
        a = generate_world(seed=99, n_items=200, horizon_days=14)
        b = generate_world(seed=99, n_items=200, horizon_days=14)
        self.assertEqual(a.total_at_risk_paise, b.total_at_risk_paise)
        self.assertEqual(
            [(i.item_id, i.amount_paise, i.method, i.error_reason) for i in a.items],
            [(i.item_id, i.amount_paise, i.method, i.error_reason) for i in b.items],
        )

    def test_same_seed_gives_the_same_ledger_chain(self):
        """The strongest reproducibility claim available: identical hash head."""
        heads = []
        for _ in range(2):
            world = generate_world(seed=17, n_items=150, horizon_days=14)
            result = run_agent(CONFIGS["b1_fixed_schedule"], world)
            heads.append(result.ledger_head)
        self.assertEqual(heads[0], heads[1])
        self.assertNotEqual(heads[0], "0" * 64)

    def test_secondask_is_reproducible(self):
        heads = []
        for _ in range(2):
            world = generate_world(seed=23, n_items=120, horizon_days=14)
            result = run_agent(CONFIGS["secondask"], world)
            heads.append((result.ledger_head, result.recovered_paise, result.messages_sent))
        self.assertEqual(heads[0], heads[1])


class CommonRandomNumbersTest(unittest.TestCase):
    def test_same_action_same_hour_gives_the_same_draw(self):
        """Otherwise a difference between agents is luck, not policy."""
        world_a = generate_world(seed=55, n_items=60, horizon_days=14)
        world_b = generate_world(seed=55, n_items=60, horizon_days=14)
        when = world_a.start + timedelta(days=2, hours=5)

        for item_a, item_b in zip(world_a.items, world_b.items):
            out_a = execute_counterfactual(
                world_a, item_a, world_a.customers[item_a.customer_id], ActionKind.PAYMENT_LINK_SMS, when
            )
            out_b = execute_counterfactual(
                world_b, item_b, world_b.customers[item_b.customer_id], ActionKind.PAYMENT_LINK_SMS, when
            )
            self.assertEqual(out_a.success, out_b.success, item_a.item_id)

    def test_retries_do_not_compound(self):
        """A retry asks about the account, not about a person.

        Repeating it while nothing has changed must give the same answer, or
        persistence pays and timing stops mattering.
        """
        world = generate_world(seed=61, n_items=200, horizon_days=14)
        mandates = [i for i in world.items if i.method.is_mandate]
        self.assertTrue(mandates, "no mandate items generated")
        item = mandates[0]
        customer = world.customers[item.customer_id]
        when = item.failed_at + timedelta(hours=2)
        outcomes = {
            execute_counterfactual(world, item, customer, ActionKind.MANDATE_DEBIT, when + timedelta(minutes=m)).success
            for m in (0, 5, 30, 55)
        }
        self.assertEqual(len(outcomes), 1, "retrying within the same state changed the outcome")


class DeadInstrumentTest(unittest.TestCase):
    def test_retrying_a_dead_instrument_never_works(self):
        """A hard zero, not a low probability. This is the load-bearing fact."""
        world = generate_world(seed=71, n_items=600, horizon_days=21)
        dead = [i for i in world.items if i._blocker == Blocker.INSTRUMENT_DEAD and i.method.is_mandate]
        self.assertTrue(dead, "no dead-instrument mandate items generated")
        for item in dead[:40]:
            customer = world.customers[item.customer_id]
            for days in range(0, 14, 2):
                when = item.failed_at + timedelta(days=days)
                if when >= world.end:
                    break
                outcome = execute_counterfactual(world, item, customer, ActionKind.MANDATE_DEBIT, when)
                self.assertFalse(outcome.success, "a retry recovered a dead instrument")


class IdempotencyTest(unittest.TestCase):
    def now(self):
        return datetime(2026, 3, 10, 12, tzinfo=timezone.utc)

    def test_replay_returns_the_same_link(self):
        """A timeout on a request that actually succeeded must not create two."""
        client = RazorpayClient(mode="mock", seed=1, failure_rate=0.0, rate_limit_rate=0.0)
        first = client.create_payment_link(
            amount_paise=45000, description="x", reference_id="r", idempotency_key="SAME", now=self.now()
        )
        second = client.create_payment_link(
            amount_paise=45000, description="x", reference_id="r", idempotency_key="SAME", now=self.now()
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(client.stats.idempotent_hits, 1)

    def test_different_keys_create_different_links(self):
        client = RazorpayClient(mode="mock", seed=1, failure_rate=0.0, rate_limit_rate=0.0)
        a = client.create_payment_link(amount_paise=1, description="x", reference_id="r", idempotency_key="A", now=self.now())
        b = client.create_payment_link(amount_paise=1, description="x", reference_id="r", idempotency_key="B", now=self.now())
        self.assertNotEqual(a["id"], b["id"])

    def test_non_positive_amount_is_refused(self):
        client = RazorpayClient(mode="mock", seed=1)
        with self.assertRaises(RazorpayError):
            client.create_payment_link(
                amount_paise=0, description="x", reference_id="r", idempotency_key="Z", now=self.now()
            )

    def test_live_mode_refuses_a_non_test_key(self):
        import os

        saved = {k: os.environ.get(k) for k in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET")}
        os.environ["RAZORPAY_KEY_ID"] = "rzp_live_ABCDEFGH"
        os.environ["RAZORPAY_KEY_SECRET"] = "secret"
        try:
            with self.assertRaises(RazorpayError) as caught:
                RazorpayClient(mode="live_test")
            self.assertIn("rzp_test_", str(caught.exception))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class CircuitBreakerTest(unittest.TestCase):
    def test_opens_after_the_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_after_seconds=60)
        now = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        for _ in range(3):
            breaker.before_call(now)
            breaker.on_failure(now)
        self.assertEqual(breaker.state, BreakerState.OPEN)
        with self.assertRaises(CircuitOpen):
            breaker.before_call(now)

    def test_half_open_then_closes_on_success(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_after_seconds=60)
        now = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        for _ in range(2):
            breaker.before_call(now)
            breaker.on_failure(now)
        later = now + timedelta(seconds=61)
        breaker.before_call(later)
        self.assertEqual(breaker.state, BreakerState.HALF_OPEN)
        breaker.on_success()
        self.assertEqual(breaker.state, BreakerState.CLOSED)

    def test_a_failed_probe_reopens_immediately(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_after_seconds=60)
        now = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)
        for _ in range(2):
            breaker.before_call(now)
            breaker.on_failure(now)
        later = now + timedelta(seconds=61)
        breaker.before_call(later)
        breaker.on_failure(later)
        self.assertEqual(breaker.state, BreakerState.OPEN)
        self.assertEqual(breaker.trips, 2)

    def test_backoff_is_bounded_and_deterministic(self):
        self.assertEqual(backoff_delay(1, "k"), backoff_delay(1, "k"))
        for attempt in range(1, 40):
            delay = backoff_delay(attempt, "k", cap_seconds=30.0)
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, 30.0)


class SeedSeparationTest(unittest.TestCase):
    def test_overlap_raises(self):
        with self.assertRaises(SeedLeakage):
            assert_disjoint((1, 2, 3), (3, 4))

    def test_disjoint_passes(self):
        assert_disjoint((1, 2), (3, 4))

    def test_shipped_model_was_not_trained_on_an_eval_seed(self):
        import os

        path = os.path.join("models", "underwriter.json")
        if not os.path.exists(path):
            self.skipTest("no fitted model on disk")
        from secondask.underwrite.model import Underwriter

        underwriter = Underwriter.load(path)
        assert_disjoint(underwriter.train_seeds, (7, 11, 13))


class OverpaymentTest(unittest.TestCase):
    def test_recovered_never_exceeds_the_amount(self):
        world = generate_world(seed=88, n_items=400, horizon_days=21)
        run_agent(CONFIGS["secondask"], world)
        for item in world.items:
            self.assertLessEqual(item.recovered_paise, item.amount_paise, item.item_id)
            self.assertGreaterEqual(item.outstanding_paise, 0)


if __name__ == "__main__":
    unittest.main()
