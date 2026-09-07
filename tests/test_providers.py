"""Provider selection, the Gemini backend, and the confidence intervals.

The Gemini tests deliberately make **no network calls**. A test suite that needs
credentials and a working internet connection is a test suite that gets skipped,
and the parts worth testing here are all local: request shape, response parsing,
error classification, and the fact that a missing key returns None rather than
raising.

The statistics tests are the more important half. An interval that is wrong is
worse than no interval, because it looks authoritative.
"""

from __future__ import annotations

import json
import os
import unittest
from contextlib import contextmanager

from secondask.eval.stats import (
    MIN_SEEDS_FOR_CI,
    bootstrap_ci,
    paired_bootstrap_ci,
    ratio_ci,
    sign_test_p,
)
from secondask.llm import providers
from secondask.llm.gemini_client import GeminiBackend, _first_text, _key_from_env


@contextmanager
def env(**values):
    """Temporarily set or clear environment variables."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


ALL_KEYS = {
    "ANTHROPIC_API_KEY": None,
    "GEMINI_API_KEY": None,
    "GOOGLE_API_KEY": None,
    "GOOGLE_GENAI_API_KEY": None,
}


class ProviderSelectionTest(unittest.TestCase):
    def test_no_credentials_returns_none(self):
        """Running without a key is the normal case and must never raise."""
        with env(**ALL_KEYS):
            self.assertIsNone(providers.build("auto"))
            self.assertFalse(any(providers.available().values()))

    def test_explicit_none_returns_none_even_with_keys(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "x"}):
            self.assertIsNone(providers.build("none"))

    def test_gemini_is_selected_when_only_it_has_a_key(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "test-key"}):
            backend = providers.build("auto")
            self.assertIsNotNone(backend)
            self.assertTrue(backend.name.startswith("gemini"))

    def test_google_api_key_is_accepted_as_an_alias(self):
        with env(**{**ALL_KEYS, "GOOGLE_API_KEY": "test-key"}):
            self.assertTrue(providers.available()["gemini"])
            self.assertIsNotNone(providers.build("gemini"))

    def test_claude_wins_auto_when_both_are_present(self):
        """The README numbers were produced with Claude, so auto keeps that."""
        with env(**{**ALL_KEYS, "ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g"}):
            backend = providers.build("auto")
            self.assertIsNotNone(backend)
            self.assertTrue(backend.name.startswith("claude"))

    def test_explicit_choice_overrides_auto_order(self):
        with env(**{**ALL_KEYS, "ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g"}):
            self.assertTrue(providers.build("gemini").name.startswith("gemini"))

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            providers.build("openai")

    def test_describe_never_leaks_a_key(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "super-secret-value"}):
            text = providers.describe()
            self.assertNotIn("super-secret-value", text)
            self.assertIn("gemini", text)

    def test_available_reports_presence_not_values(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "k"}):
            for value in providers.available().values():
                self.assertIsInstance(value, bool)


class GeminiBackendTest(unittest.TestCase):
    def test_missing_key_raises_on_direct_construction(self):
        with env(**ALL_KEYS):
            with self.assertRaises(RuntimeError):
                GeminiBackend()

    def test_build_backend_returns_none_rather_than_raising(self):
        from secondask.llm.gemini_client import build_backend

        with env(**ALL_KEYS):
            self.assertIsNone(build_backend())

    def test_key_lookup_order(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "first", "GOOGLE_API_KEY": "second"}):
            self.assertEqual(_key_from_env(), "first")
        with env(**{**ALL_KEYS, "GOOGLE_API_KEY": "second"}):
            self.assertEqual(_key_from_env(), "second")

    def test_response_parsing(self):
        good = {"candidates": [{"content": {"parts": [{"text": '{"intents":["opt_out"]}'}]}}]}
        self.assertEqual(_first_text(good), '{"intents":["opt_out"]}')

    def test_response_parsing_survives_every_empty_shape(self):
        """A blocked prompt or a MAX_TOKENS finish yields no parts.

        Returning "" lets the gateway treat it as malformed and fall back, rather
        than raising a KeyError three frames deeper.
        """
        for payload in (
            {},
            {"candidates": []},
            {"candidates": [{}]},
            {"candidates": [{"content": {}}]},
            {"candidates": [{"content": {"parts": []}}]},
            {"candidates": [{"content": {"parts": [{"inlineData": {}}]}}]},
            {"candidates": [None]},
        ):
            self.assertEqual(_first_text(payload), "")

    def test_model_is_configurable(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "k", "SECONDASK_GEMINI_MODEL": "gemini-x"}):
            self.assertEqual(GeminiBackend().model, "gemini-x")

    def test_usage_counters_start_clean(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "k"}):
            usage = GeminiBackend().usage()
            self.assertEqual(usage["calls"], 0)
            self.assertIn("gemini", usage["backend"])

    def test_satisfies_the_backend_protocol(self):
        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "k"}):
            backend = GeminiBackend()
            self.assertTrue(hasattr(backend, "name"))
            self.assertTrue(callable(getattr(backend, "complete_json", None)))

    def test_gateway_accepts_it_without_knowing_the_vendor(self):
        """The swappability claim, asserted rather than described."""
        from secondask.llm.gateway import LLMGateway

        with env(**{**ALL_KEYS, "GEMINI_API_KEY": "k"}):
            gateway = LLMGateway(backend=GeminiBackend())
            self.assertTrue(gateway.backend_name.startswith("gemini"))


class BootstrapTest(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self):
        values = [100, 110, 105, 120, 98, 115, 102, 108]
        interval = bootstrap_ci(values)
        self.assertLessEqual(interval.low, interval.point)
        self.assertLessEqual(interval.point, interval.high)
        self.assertEqual(interval.n, len(values))

    def test_it_is_deterministic(self):
        """An interval that moves between runs of the same data is a rumour."""
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        a, b = bootstrap_ci(values), bootstrap_ci(values)
        self.assertEqual((a.low, a.point, a.high), (b.low, b.point, b.high))

    def test_more_data_gives_a_tighter_interval(self):
        import random

        rng = random.Random(5)
        small = [rng.gauss(100, 15) for _ in range(6)]
        large = [rng.gauss(100, 15) for _ in range(200)]
        self.assertGreater(
            bootstrap_ci(small).high - bootstrap_ci(small).low,
            bootstrap_ci(large).high - bootstrap_ci(large).low,
        )

    def test_degenerate_inputs(self):
        self.assertEqual(bootstrap_ci([]).n, 0)
        one = bootstrap_ci([42.0])
        self.assertEqual((one.point, one.low, one.high), (42.0, 42.0, 42.0))
        self.assertFalse(one.reliable)

    def test_reliability_threshold(self):
        self.assertFalse(bootstrap_ci([1] * (MIN_SEEDS_FOR_CI - 1)).reliable)
        self.assertTrue(bootstrap_ci(list(range(MIN_SEEDS_FOR_CI))).reliable)

    def test_zero_variance_collapses_the_interval(self):
        interval = bootstrap_ci([7.0] * 10)
        self.assertAlmostEqual(interval.low, 7.0)
        self.assertAlmostEqual(interval.high, 7.0)


class PairedTest(unittest.TestCase):
    def test_pairing_is_tighter_than_treating_samples_as_independent(self):
        """The reason shared seeds exist.

        Both agents saw the same worlds, so most of the spread is the batch
        rather than the policy. Pairing removes it; ignoring the pairing gives an
        interval several times too wide.
        """
        treatment = [100, 200, 300, 400, 500, 600, 700]
        control = [90, 190, 290, 390, 490, 590, 690]  # always exactly 10 lower
        paired = paired_bootstrap_ci(treatment, control)
        self.assertAlmostEqual(paired.point, 10.0, places=6)
        self.assertLess(paired.high - paired.low, 1e-6, "a constant delta must have a zero-width interval")

        unpaired_spread = bootstrap_ci(treatment).high - bootstrap_ci(treatment).low
        self.assertGreater(unpaired_spread, paired.high - paired.low)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1, 2, 3], [1, 2])

    def test_excludes_zero_detects_a_real_effect(self):
        clearly_better = paired_bootstrap_ci([10] * 8, [1] * 8)
        self.assertTrue(clearly_better.excludes_zero)
        noise = paired_bootstrap_ci([1, -1, 1, -1, 1, -1], [0] * 6)
        self.assertFalse(noise.excludes_zero)


class RatioTest(unittest.TestCase):
    def test_it_is_the_ratio_of_sums(self):
        """Not the mean of per-seed ratios.

        A seed with a tiny denominator produces an enormous per-seed ratio and
        drags the average somewhere no individual seed went.
        """
        treatment = [10, 100]
        control = [1, 100]
        self.assertAlmostEqual(ratio_ci(treatment, control).point, 110 / 101)

    def test_zero_denominator_does_not_divide_by_zero(self):
        self.assertEqual(ratio_ci([1, 2], [0, 0]).point, 0.0)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            ratio_ci([1], [1, 2])


class SignTestTest(unittest.TestCase):
    def test_all_wins_is_significant(self):
        self.assertLess(sign_test_p([1] * 8), 0.01)

    def test_an_even_split_is_not(self):
        self.assertGreater(sign_test_p([1, -1, 1, -1, 1, -1]), 0.5)

    def test_ties_are_dropped(self):
        self.assertEqual(sign_test_p([0, 0, 0]), 1.0)
        self.assertEqual(sign_test_p([]), 1.0)

    def test_known_value(self):
        """Six of six one-sided is 1/64; two-sided is 2/64."""
        self.assertAlmostEqual(sign_test_p([1] * 6), 2 / 64)


if __name__ == "__main__":
    unittest.main()


class BackendBreakerTest(unittest.TestCase):
    """The gateway stops calling a backend that is consistently failing.

    Found by pointing the system at a real credential that turned out to be for
    a different API. Every message cost two failed round trips before falling
    back, which is invisible at four messages and is two hours of latency plus
    24,000 rejected requests across a 12,000 item batch.
    """

    class AlwaysFails:
        name = "broken"

        def __init__(self):
            self.calls = 0

        def complete_json(self, system, user, *, max_tokens=256):
            self.calls += 1
            raise RuntimeError("HTTP 403: blocked")

    class FailsThenWorks:
        name = "flaky"

        def __init__(self, failures):
            self.remaining = failures
            self.calls = 0

        def complete_json(self, system, user, *, max_tokens=256):
            self.calls += 1
            if self.remaining > 0:
                self.remaining -= 1
                raise RuntimeError("transient")
            return json.dumps({"intents": ["opt_out"]})

    def setUp(self):
        from datetime import datetime, timezone

        self.now = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)

    def gateway(self, backend, **kwargs):
        from secondask.llm.gateway import LLMGateway

        return LLMGateway(backend=backend, max_repairs=1, **kwargs)

    def test_backend_is_cut_off_after_the_threshold(self):
        from secondask.world.entities import ReplyIntent

        backend = self.AlwaysFails()
        gw = self.gateway(backend, failure_threshold=4)
        for _ in range(50):
            parsed = gw.parse_reply("STOP", self.now)
        self.assertTrue(gw.backend_disabled)
        self.assertEqual(gw.stats.backend_disabled_after, 4)
        # Bounded, not 50 parses worth of retries.
        self.assertLessEqual(backend.calls, 6)
        # And still correct.
        self.assertEqual(parsed.intent, ReplyIntent.OPT_OUT)
        self.assertEqual(parsed.source, "stub_backend_down")

    def test_a_transient_failure_does_not_latch(self):
        """Recovering from a blip must not disable the backend for the run."""
        backend = self.FailsThenWorks(failures=2)
        gw = self.gateway(backend, failure_threshold=8)
        for _ in range(6):
            gw.parse_reply("stop", self.now)
        self.assertFalse(gw.backend_disabled)
        self.assertGreater(backend.calls, 2)

    def test_success_resets_the_counter(self):
        backend = self.FailsThenWorks(failures=3)
        gw = self.gateway(backend, failure_threshold=5)
        for _ in range(10):
            gw.parse_reply("stop", self.now)
        self.assertFalse(gw.backend_disabled)

    def test_disabled_backend_still_answers_correctly(self):
        from secondask.world.entities import ReplyIntent

        gw = self.gateway(self.AlwaysFails(), failure_threshold=2)
        for _ in range(10):
            gw.parse_reply("x", self.now)
        self.assertTrue(gw.backend_disabled)
        cases = [
            ("STOP", ReplyIntent.OPT_OUT),
            ("I never ordered this", ReplyIntent.DISPUTE),
            ("I can pay half, Rs 2500, on the 5th", ReplyIntent.PARTIAL_PAYMENT_PROMISE),
        ]
        for text, expected in cases:
            self.assertEqual(gw.parse_reply(text, self.now).intent, expected, text)

    def test_slot_filling_also_honours_the_breaker(self):
        gw = self.gateway(self.AlwaysFails(), failure_threshold=2)
        for _ in range(6):
            gw.parse_reply("x", self.now)
        self.assertTrue(gw.backend_disabled)
        before = gw.backend.calls
        out = gw.fill_slots("RETRY_LINK_EN", ("merchant",), {}, "en", {"merchant": "Acme"})
        self.assertEqual(out, {"merchant": "Acme"})
        self.assertEqual(gw.backend.calls, before, "fill_slots called a disabled backend")

    def test_degradation_is_visible_in_the_stats(self):
        """A quietly degraded run is worse than a loud one."""
        gw = self.gateway(self.AlwaysFails(), failure_threshold=3)
        for _ in range(10):
            gw.parse_reply("x", self.now)
        self.assertEqual(gw.stats.to_dict()["backend_disabled_after"], 3)
