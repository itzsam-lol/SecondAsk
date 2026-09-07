"""Money, time, randomness and the ledger."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from secondask import money
from secondask.clock import (
    IST,
    UTC,
    VirtualClock,
    day_of_month_ist,
    days_in_month,
    days_to_month_end_ist,
    hour_bucket,
    ist_time_of_day,
    iso,
    next_ist_time,
    parse_iso,
    to_ist,
    to_utc,
)
from secondask.ledger import GENESIS, Ledger, canonical_json
from secondask.rng import Stream, crn_uniform


class MoneyTest(unittest.TestCase):
    def test_rupees_to_paise_is_exact(self):
        self.assertEqual(money.rupees(1), 100)
        self.assertEqual(money.rupees("0.07"), 7)
        self.assertEqual(money.rupees(Decimal("1234.56")), 123456)

    def test_float_input_does_not_lose_a_paisa(self):
        """0.07 is not representable in binary; naive int(0.07*100) gives 6."""
        self.assertEqual(money.rupees(0.07), 7)
        self.assertEqual(money.rupees(0.29), 29)
        self.assertEqual(money.rupees(1.005), 101)  # half-up, not banker's

    def test_indian_grouping(self):
        self.assertEqual(money._indian_group(1234567), "12,34,567")
        self.assertEqual(money._indian_group(100), "100")
        self.assertEqual(money._indian_group(1000), "1,000")
        self.assertEqual(money._indian_group(100000), "1,00,000")
        self.assertEqual(money._indian_group(10000000), "1,00,00,000")

    def test_negative_amounts_format(self):
        self.assertTrue(money.fmt(-12345).startswith("-"))

    def test_percentage_of_zero_does_not_raise(self):
        """Empty batches and zero-message runs are real states to report on."""
        self.assertEqual(money.pct(0, 0), "n/a")
        self.assertEqual(money.pct(5, 0), "n/a")
        self.assertEqual(money.safe_div(5, 0), 0.0)

    def test_compact_scales(self):
        self.assertIn("L", money.fmt(money.rupees(250000), compact=True))
        self.assertIn("Cr", money.fmt(money.rupees(30000000), compact=True))


class ClockTest(unittest.TestCase):
    def test_ist_is_a_fixed_offset(self):
        """India has no DST, so the offset must be identical in January and July."""
        january = datetime(2026, 1, 15, 12, tzinfo=UTC)
        july = datetime(2026, 7, 15, 12, tzinfo=UTC)
        self.assertEqual(to_ist(january).utcoffset(), to_ist(july).utcoffset())
        self.assertEqual(to_ist(january).utcoffset(), timedelta(hours=5, minutes=30))

    def test_naive_datetimes_are_rejected(self):
        with self.assertRaises(ValueError):
            to_ist(datetime(2026, 3, 10, 12))
        with self.assertRaises(ValueError):
            to_utc(datetime(2026, 3, 10, 12))

    def test_time_of_day_handles_the_half_hour_offset(self):
        at = datetime(2026, 3, 10, 6, 0, tzinfo=UTC)  # 11:30 IST
        self.assertAlmostEqual(ist_time_of_day(at), 11.5, places=4)

    def test_days_in_month(self):
        self.assertEqual(days_in_month(2026, 2), 28)
        self.assertEqual(days_in_month(2028, 2), 29)  # leap
        self.assertEqual(days_in_month(2026, 4), 30)
        self.assertEqual(days_in_month(2026, 12), 31)

    def test_month_end_across_lengths(self):
        for year, month, day, expected in [
            (2026, 1, 31, 0),
            (2026, 2, 28, 0),
            (2028, 2, 28, 1),
            (2026, 4, 29, 1),
        ]:
            at = to_utc(datetime(year, month, day, 12, tzinfo=IST))
            self.assertEqual(days_to_month_end_ist(at), expected, f"{year}-{month}-{day}")

    def test_next_ist_time_rolls_forward(self):
        at = to_utc(datetime(2026, 3, 10, 20, 0, tzinfo=IST))
        nxt = next_ist_time(at, 8)
        local = to_ist(nxt)
        self.assertEqual((local.day, local.hour), (11, 0 + 8))

    def test_next_ist_time_same_day_when_earlier(self):
        at = to_utc(datetime(2026, 3, 10, 6, 0, tzinfo=IST))
        local = to_ist(next_ist_time(at, 8))
        self.assertEqual((local.day, local.hour), (10, 8))

    def test_clock_never_runs_backwards(self):
        clock = VirtualClock(datetime(2026, 3, 10, 12, tzinfo=UTC))
        clock.advance_to(datetime(2026, 3, 10, 10, tzinfo=UTC))
        self.assertEqual(clock.now, datetime(2026, 3, 10, 12, tzinfo=UTC))

    def test_iso_round_trip_drops_microseconds(self):
        at = datetime(2026, 3, 10, 12, 30, 45, 123456, tzinfo=UTC)
        self.assertEqual(iso(at), "2026-03-10T12:30:45Z")
        self.assertEqual(parse_iso(iso(at)), at.replace(microsecond=0))

    def test_hour_bucket_is_stable_within_an_hour(self):
        base = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        self.assertEqual(hour_bucket(base), hour_bucket(base + timedelta(minutes=59)))
        self.assertNotEqual(hour_bucket(base), hour_bucket(base + timedelta(minutes=61)))


class RngTest(unittest.TestCase):
    def test_crn_is_deterministic_across_calls(self):
        a = crn_uniform(7, "item_1", "payment_link_sms", 12345)
        b = crn_uniform(7, "item_1", "payment_link_sms", 12345)
        self.assertEqual(a, b)

    def test_crn_separates_arguments(self):
        """('ab','c') and ('a','bc') must not collide."""
        self.assertNotEqual(crn_uniform("ab", "c"), crn_uniform("a", "bc"))

    def test_crn_is_in_range(self):
        for i in range(500):
            value = crn_uniform("x", i)
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 1.0)

    def test_substreams_are_independent(self):
        """Adding a draw in one sub-stream must not perturb another."""
        root = Stream(11, "root")
        first = [root.sub("a").uniform() for _ in range(3)]
        root2 = Stream(11, "root")
        _ = [root2.sub("b").uniform() for _ in range(10)]
        second = [root2.sub("a").uniform() for _ in range(3)]
        self.assertEqual(first, second)

    def test_weighted_handles_degenerate_input(self):
        stream = Stream(1, "t")
        self.assertEqual(stream.weighted([("only", 0.0)]), "only")
        with self.assertRaises(ValueError):
            stream.weighted([])

    def test_lognormal_is_clamped(self):
        stream = Stream(3, "t")
        for _ in range(300):
            value = stream.lognormal_int(500, 2.0, low=10, high=1000)
            self.assertGreaterEqual(value, 10)
            self.assertLessEqual(value, 1000)


class LedgerTest(unittest.TestCase):
    def build(self, n=5) -> Ledger:
        ledger = Ledger(run_id="t")
        base = datetime(2026, 3, 10, 12, tzinfo=UTC)
        for i in range(n):
            ledger.append(base + timedelta(minutes=i), "action", f"item_{i}", {"i": i, "amount": 100 * i})
        return ledger

    def test_empty_ledger_head_is_genesis(self):
        self.assertEqual(Ledger().head, GENESIS)

    def test_chain_verifies(self):
        ok, message = self.build().verify()
        self.assertTrue(ok, message)

    def test_editing_a_payload_breaks_the_chain(self):
        ledger = self.build()
        entries = list(ledger.entries)
        # LedgerEntry is frozen, so tamper with the mutable payload dict inside.
        entries[2].payload["amount"] = 999999
        ok, message = ledger.verify()
        self.assertFalse(ok)
        self.assertIn("entry 2", message)
        self.assertIn("modified", message)

    def test_head_commits_to_the_whole_history(self):
        a = self.build()
        b = self.build()
        self.assertEqual(a.head, b.head)
        b.append(datetime(2026, 3, 10, 13, tzinfo=UTC), "action", "extra", {})
        self.assertNotEqual(a.head, b.head)

    def test_entries_view_is_immutable(self):
        ledger = self.build()
        self.assertIsInstance(ledger.entries, tuple)
        before = len(ledger)
        with self.assertRaises(AttributeError):
            ledger.entries.append(None)  # type: ignore[attr-defined]
        self.assertEqual(len(ledger), before)

    def test_canonical_json_is_stable_for_non_ascii(self):
        """Devanagari in a message body must hash identically on every platform."""
        payload = {"body": "नमस्ते", "n": 1}
        first = canonical_json(payload)
        second = canonical_json({"n": 1, "body": "नमस्ते"})
        self.assertEqual(first, second)
        self.assertNotIn("न", first)  # escaped, so the byte stream is ASCII

    def test_round_trip_through_jsonl(self):
        import os
        import tempfile

        ledger = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.jsonl")
            ledger.write_jsonl(path)
            restored = Ledger.read_jsonl(path)
        self.assertEqual(restored.head, ledger.head)
        ok, _ = restored.verify()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
