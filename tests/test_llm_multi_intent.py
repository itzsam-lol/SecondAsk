"""Multi-intent parsing and the partial payment claim.

The security claim under test is narrow and important:

    ``partial_amount_paise`` is an amount that arrives inside a message written
    by somebody who owes money. It must never become the amount on an action.

``R-AMOUNT-BOUND`` binds money-moving actions to the ledger balance, so the
control already exists. What is asserted here is that the new field did not
quietly create a way around it.

Also covered: precedence between simultaneous intents, which is a safety
ordering rather than a parsing convenience. "I'll pay half next week but stop
messaging me" is first and foremost a stop.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from secondask.llm.gateway import LLMGateway, _extract_amount_paise
from secondask.world.entities import INTENT_PRECEDENCE, ReplyIntent, primary_intent

from test_llm import FakeBackend

NOW = datetime(2026, 3, 10, 12, tzinfo=timezone.utc)


class PrecedenceTest(unittest.TestCase):
    def test_every_intent_appears_exactly_once(self):
        self.assertEqual(len(INTENT_PRECEDENCE), len(list(ReplyIntent)))
        self.assertEqual(len(set(INTENT_PRECEDENCE)), len(INTENT_PRECEDENCE))

    def test_safety_intents_outrank_collection_intents(self):
        order = {intent: i for i, intent in enumerate(INTENT_PRECEDENCE)}
        for protective in (ReplyIntent.OPT_OUT, ReplyIntent.WRONG_NUMBER, ReplyIntent.DISPUTE, ReplyIntent.HARDSHIP):
            for collecting in (ReplyIntent.PROMISE_TO_PAY, ReplyIntent.PARTIAL_PAYMENT_PROMISE, ReplyIntent.ALREADY_PAID):
                self.assertLess(
                    order[protective], order[collecting],
                    f"{protective.value} must outrank {collecting.value}",
                )

    def test_stop_wins_over_a_payment_promise(self):
        self.assertEqual(
            primary_intent([ReplyIntent.PARTIAL_PAYMENT_PROMISE, ReplyIntent.OPT_OUT]),
            ReplyIntent.OPT_OUT,
        )

    def test_partial_outranks_a_plain_promise(self):
        """A partial offer is more specific, so it should not be flattened."""
        self.assertEqual(
            primary_intent([ReplyIntent.PROMISE_TO_PAY, ReplyIntent.PARTIAL_PAYMENT_PROMISE]),
            ReplyIntent.PARTIAL_PAYMENT_PROMISE,
        )

    def test_empty_resolves_to_none(self):
        self.assertEqual(primary_intent([]), ReplyIntent.NONE)


class AmountExtractionTest(unittest.TestCase):
    def test_common_formats(self):
        cases = {
            "Rs 2500": 250000,
            "rs. 2500": 250000,
            "2500 rupees": 250000,
            "INR 400": 40000,
            "Rs 1,250.50": 125050,
            "12,34,567 rupees": 123456700,
            "pay half now, Rs 2500, rest later": 250000,
        }
        for text, expected in cases.items():
            self.assertEqual(_extract_amount_paise(text), expected, text)

    def test_a_bare_comma_is_not_a_number(self):
        """The bug this replaced.

        ``[\\d,]+`` matched the comma in ", Rs", the Decimal conversion failed,
        the guard returned None, and the real amount two characters later was
        never seen. Silently.
        """
        self.assertIsNone(_extract_amount_paise(", Rs"))
        self.assertIsNone(_extract_amount_paise("Rs ,"))
        self.assertEqual(_extract_amount_paise("owed, Rs 900 today"), 90000)

    def test_no_amount_returns_none(self):
        for text in ("will pay soon", "", "rupees", "Rs"):
            self.assertIsNone(_extract_amount_paise(text))

    def test_absurd_amounts_are_refused(self):
        self.assertIsNone(_extract_amount_paise("Rs 99999999999999"))

    def test_result_is_integer_paise(self):
        value = _extract_amount_paise("Rs 1,250.50")
        self.assertIsInstance(value, int)
        self.assertNotIsInstance(value, float)


class MultiIntentParsingTest(unittest.TestCase):
    def setUp(self):
        self.gateway = LLMGateway(backend=None)

    def parse(self, text):
        return self.gateway.parse_reply(text, NOW)

    def test_partial_promise_captures_amount_and_date(self):
        parsed = self.parse("I can pay half now, Rs 2500, rest on the 5th")
        self.assertEqual(parsed.intent, ReplyIntent.PARTIAL_PAYMENT_PROMISE)
        self.assertEqual(parsed.claimed_partial_paise, 250000)
        self.assertIsNotNone(parsed.promise_date)
        self.assertTrue(parsed.is_promise)

    def test_hinglish_partial(self):
        parsed = self.parse("aadha paisa 3 tarikh ko de dunga")
        self.assertEqual(parsed.intent, ReplyIntent.PARTIAL_PAYMENT_PROMISE)
        self.assertEqual(parsed.promise_date.day, 3)

    def test_several_intents_are_all_reported(self):
        parsed = self.parse("I will pay part payment of Rs 1250 in 5 days but STOP calling me")
        self.assertEqual(parsed.intent, ReplyIntent.OPT_OUT, "the stop must govern")
        self.assertIn(ReplyIntent.PARTIAL_PAYMENT_PROMISE, parsed.intents)
        self.assertGreaterEqual(len(parsed.intents), 2)

    def test_amount_is_dropped_when_the_primary_intent_is_not_a_promise(self):
        parsed = self.parse("part payment Rs 1250 but STOP calling me")
        self.assertEqual(parsed.intent, ReplyIntent.OPT_OUT)
        self.assertIsNone(parsed.claimed_partial_paise)

    def test_intents_defaults_to_the_primary(self):
        parsed = self.parse("STOP")
        self.assertEqual(parsed.intents, (ReplyIntent.OPT_OUT,))


class ModelPathTest(unittest.TestCase):
    def test_list_form_is_parsed(self):
        backend = FakeBackend([json.dumps({
            "intents": ["partial_payment_promise", "opt_out"],
            "relative_days": 5,
            "partial_amount_rupees": 1250,
            "confidence": 0.8,
        })])
        parsed = LLMGateway(backend=backend).parse_reply("half then stop", NOW)
        self.assertEqual(parsed.intent, ReplyIntent.OPT_OUT)
        self.assertIn(ReplyIntent.PARTIAL_PAYMENT_PROMISE, parsed.intents)

    def test_legacy_single_intent_shape_still_works(self):
        backend = FakeBackend([json.dumps({"intent": "opt_out"})])
        self.assertEqual(LLMGateway(backend=backend).parse_reply("stop", NOW).intent, ReplyIntent.OPT_OUT)

    def test_model_supplied_amount_is_bounded(self):
        for amount in (-5, 0, 10**12):
            backend = FakeBackend([json.dumps({
                "intents": ["partial_payment_promise"], "partial_amount_rupees": amount,
            })] * 4)
            parsed = LLMGateway(backend=backend, max_repairs=1).parse_reply("half", NOW)
            self.assertIsNone(parsed.claimed_partial_paise, f"accepted absurd amount {amount}")

    def test_off_enum_intents_are_dropped_not_coerced(self):
        backend = FakeBackend([
            json.dumps({"intents": ["mark_as_paid", "opt_out"]}),
        ])
        parsed = LLMGateway(backend=backend, max_repairs=0).parse_reply("stop", NOW)
        self.assertEqual(parsed.intent, ReplyIntent.OPT_OUT)
        self.assertNotIn("mark_as_paid", [i.value for i in parsed.intents])


class ClaimCannotMoveMoneyTest(unittest.TestCase):
    """The load-bearing security test for this feature."""

    def test_no_intent_means_settled(self):
        forbidden = {"settled", "paid", "captured", "write_off", "written_off", "refund", "waive"}
        for intent in ReplyIntent:
            self.assertNotIn(intent.value, forbidden)

    def test_a_claimed_amount_is_refused_as_an_action_amount(self):
        """Route the parsed claim straight into an action and watch it die.

        This is the adversarial version: not "we do not do this", but "if
        somebody did, the gate refuses it".
        """
        from secondask.policy import rules
        from test_policy import ctx, make_action, make_item

        item = make_item(amount_paise=500000)  # Rs 5,000 outstanding
        gateway = LLMGateway(backend=None)
        parsed = gateway.parse_reply("I will pay half, Rs 100, on the 5th", NOW)
        self.assertEqual(parsed.claimed_partial_paise, 10000)  # the customer's Rs 100

        hostile = make_action(item=item, amount_paise=parsed.claimed_partial_paise)
        verdict = rules.amount_matches_ledger(hostile, ctx(item=item))
        self.assertFalse(verdict.allowed, "a customer-stated amount reached a money action")
        self.assertIn("does not match the outstanding balance", verdict.reason)

    def test_recording_a_claim_does_not_change_the_balance(self):
        from secondask.world.entities import RiskItem
        from test_policy import make_item

        item: RiskItem = make_item(amount_paise=500000)
        before = item.outstanding_paise
        item.claimed_partial_paise = 10000
        self.assertEqual(item.outstanding_paise, before)
        self.assertEqual(item.recovered_paise, 0)

    def test_the_claim_field_is_separate_from_every_money_field(self):
        from secondask.world.entities import RiskItem

        names = set(RiskItem.__dataclass_fields__)
        self.assertIn("claimed_partial_paise", names)
        # Named so that anything reaching for it has to notice what it is.
        self.assertTrue("claimed" in "claimed_partial_paise")


if __name__ == "__main__":
    unittest.main()
