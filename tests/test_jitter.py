"""Deferral jitter.

The bug this prevents: every action deferred out of the forbidden contact window
returned exactly 08:00:00. Overnight failures accumulate for eleven hours and
then fire in the same second. The gateway sees a spike two orders of magnitude
above steady state, the SMS provider rate-limits, the circuit breaker opens, and
a perfectly compliant system has manufactured its own outage.

What is asserted here is not "there is some randomness". It is the four
properties that make the stagger safe to rely on:

1. spread across the intended window and nowhere else,
2. approximately uniform, so it actually flattens the peak,
3. deterministic, so a replay produces the same schedule,
4. independent of run state, so it does not depend on arrival order or on how
   many items happen to be waiting.
"""

from __future__ import annotations

import unittest
from collections import Counter
from datetime import datetime, timedelta

from secondask.clock import IST, to_ist, to_utc
from secondask.policy import rules
from secondask.policy.rules import DISPATCH_SPREAD_SECONDS, deferred_start

from test_policy import ctx, ist, make_action, make_item


def window_start(day: int, hour: int) -> datetime:
    return to_utc(datetime(2026, 3, day, hour, 0, 0, tzinfo=IST))


class JitterDistributionTest(unittest.TestCase):
    def offsets(self, n: int = 3000, hour: int = 8) -> list[int]:
        at = to_utc(datetime(2026, 3, 10, 23, 0, tzinfo=IST))
        base = window_start(11, hour)
        return [
            int((deferred_start(at, hour, f"item_{i:06d}") - base).total_seconds())
            for i in range(n)
        ]

    def test_every_offset_is_inside_the_window(self):
        for offset in self.offsets():
            self.assertGreaterEqual(offset, 0)
            self.assertLess(offset, DISPATCH_SPREAD_SECONDS)

    def test_distribution_is_approximately_uniform(self):
        """Twelve ten-minute buckets over two hours, each within 25% of even.

        A skewed stagger would still technically spread the load while leaving a
        peak, which is the failure this whole mechanism exists to prevent.
        """
        offsets = self.offsets(6000)
        buckets = Counter(o // 600 for o in offsets)
        self.assertEqual(len(buckets), 12, "expected all twelve ten-minute buckets to be used")
        expected = len(offsets) / 12
        for bucket, count in sorted(buckets.items()):
            self.assertGreater(count, expected * 0.75, f"bucket {bucket} is underfilled: {count}")
            self.assertLess(count, expected * 1.25, f"bucket {bucket} is overfilled: {count}")

    def test_no_second_carries_a_pile_up(self):
        """The original bug, asserted directly: no single second gets them all."""
        offsets = self.offsets(6000)
        worst = Counter(offsets).most_common(1)[0][1]
        self.assertLess(worst, 20, "too many items land on the same second")

    def test_deterministic_across_calls(self):
        at = to_utc(datetime(2026, 3, 10, 23, 0, tzinfo=IST))
        for item_id in ("item_1", "item_99", "abc"):
            self.assertEqual(deferred_start(at, 8, item_id), deferred_start(at, 8, item_id))

    def test_independent_of_the_deferring_moment(self):
        """Two items deferred at different times land at the same relative offset.

        The stagger is a property of the item, not of when it happened to be
        looked at. Otherwise an item that bounced twice would move each time and
        the spread would depend on retry history.
        """
        base = window_start(11, 8)
        for hour in (20, 21, 22, 23):
            at = to_utc(datetime(2026, 3, 10, hour, 0, tzinfo=IST))
            offset = (deferred_start(at, 8, "item_42") - base).total_seconds()
            self.assertEqual(offset, (deferred_start(window_start(10, 23), 8, "item_42") - base).total_seconds())

    def test_different_items_get_different_offsets(self):
        offsets = self.offsets(500)
        self.assertGreater(len(set(offsets)), 400, "offsets are collapsing onto few values")


class JitterThroughTheRulesTest(unittest.TestCase):
    def test_contact_hours_denial_is_staggered_and_legal(self):
        at = ist(hour=23, minute=15)
        seen = set()
        for i in range(200):
            item = make_item(item_id=f"item_{i}")
            verdict = rules.rbi_contact_hours(make_action(at=at, item=item), ctx(item=item, now=at))
            self.assertFalse(verdict.allowed)
            local = to_ist(verdict.retry_at)
            # Still inside the legal window: staggering must never push an action
            # back out of the hours it was deferred into.
            self.assertGreaterEqual(local.hour, rules.CONTACT_START_HOUR)
            self.assertLess(local.hour, rules.CONTACT_END_HOUR)
            seen.add(verdict.retry_at)
        self.assertGreater(len(seen), 150, "denials are not being staggered")

    def test_voice_denial_stays_inside_the_narrower_voice_window(self):
        from secondask.world.entities import ActionKind

        at = ist(hour=22)
        for i in range(120):
            item = make_item(item_id=f"v_{i}")
            action = make_action(ActionKind.VOICE_CALL, at=at, item=item)
            verdict = rules.voice_window(action, ctx(item=item, now=at))
            self.assertFalse(verdict.allowed)
            local = to_ist(verdict.retry_at)
            self.assertGreaterEqual(local.hour, rules.VOICE_START_HOUR)
            self.assertLess(local.hour, rules.VOICE_END_HOUR)


class HolidayRuleTest(unittest.TestCase):
    def test_diwali_blocks_outreach(self):
        from secondask.policy import holidays

        diwali = to_utc(datetime(2026, 11, 8, 11, 0, tzinfo=IST))
        self.assertEqual(holidays.holiday_name(diwali.date()), "Diwali")
        item = make_item()
        verdict = rules.holiday_window(make_action(at=diwali, item=item), ctx(item=item, now=diwali))
        self.assertFalse(verdict.allowed)
        self.assertIn("Diwali", verdict.reason)
        self.assertIsNotNone(verdict.retry_at, "a holiday must queue the item, not drop it")

    def test_national_holidays_resolve_for_any_year(self):
        from secondask.policy import holidays

        for year in (2024, 2030, 2041):
            self.assertEqual(holidays.holiday_name(datetime(year, 1, 26).date()), "Republic Day")
            self.assertEqual(holidays.holiday_name(datetime(year, 8, 15).date()), "Independence Day")
            self.assertEqual(holidays.holiday_name(datetime(year, 10, 2).date()), "Gandhi Jayanti")

    def test_silent_retry_is_exempt(self):
        """Nobody is contacted, so a mandate debit is not delayed by a festival."""
        from secondask.world.entities import ActionKind, Method

        diwali = to_utc(datetime(2026, 11, 8, 11, 0, tzinfo=IST))
        item = make_item(method=Method.NACH)
        action = make_action(ActionKind.MANDATE_DEBIT, at=diwali, item=item)
        self.assertTrue(rules.holiday_window(action, ctx(item=item, now=diwali)).allowed)

    def test_human_escalation_is_exempt(self):
        from secondask.world.entities import ActionKind

        diwali = to_utc(datetime(2026, 11, 8, 11, 0, tzinfo=IST))
        item = make_item()
        action = make_action(ActionKind.HUMAN_ESCALATION, at=diwali, item=item)
        self.assertTrue(rules.holiday_window(action, ctx(item=item, now=diwali)).allowed)

    def test_an_ordinary_day_is_allowed(self):
        ordinary = to_utc(datetime(2026, 3, 10, 11, 0, tzinfo=IST))
        item = make_item()
        self.assertTrue(rules.holiday_window(make_action(at=ordinary, item=item), ctx(item=item, now=ordinary)).allowed)

    def test_timezone_is_respected_at_the_date_boundary(self):
        """19:00 UTC on 1 October is already Gandhi Jayanti in IST.

        Checking the UTC date instead of the IST date would miss it, which is
        the kind of bug that only shows up on the holiday itself.
        """
        from datetime import timezone

        late = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)  # 00:30 IST on 2 Oct
        self.assertEqual(to_ist(late).date().day, 2)
        item = make_item()
        verdict = rules.holiday_window(make_action(at=late, item=item), ctx(item=item, now=late))
        self.assertFalse(verdict.allowed)
        self.assertIn("Gandhi Jayanti", verdict.reason)

    def test_calendar_coverage_is_reported(self):
        """A stale festival table must be visible, not silent."""
        from secondask.policy import holidays

        first, last = holidays.coverage()
        self.assertLessEqual(first, last)
        self.assertTrue(holidays.is_covered(2026))
        self.assertFalse(holidays.is_covered(2099))


if __name__ == "__main__":
    unittest.main()
