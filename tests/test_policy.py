"""Policy engine tests.

Heavy on boundary conditions. Every rule that names a time is tested at the exact
instant on both sides of the boundary, because "between 8 AM and 7 PM" is
precisely the phrase that produces an off-by-one which only ever surfaces in a
regulator's sample.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from secondask.clock import IST, to_utc
from secondask.policy import rules
from secondask.policy.engine import PolicyContext, PolicyEngine, ProposedAction, RuleVerdict
from secondask.world.entities import (
    ACTION_CHANNEL,
    ActionKind,
    Channel,
    Customer,
    ErrorReason,
    ErrorSource,
    ErrorStep,
    ItemState,
    Method,
    RiskItem,
)


def ist(year=2026, month=3, day=10, hour=12, minute=0, second=0) -> datetime:
    return to_utc(datetime(year, month, day, hour, minute, second, tzinfo=IST))


def make_item(**kwargs) -> RiskItem:
    defaults = dict(
        item_id="item_1",
        customer_id="cust_1",
        amount_paise=250000,
        method=Method.UPI,
        error_source=ErrorSource.CUSTOMER,
        error_reason=ErrorReason.INSUFFICIENT_FUNDS,
        error_step=ErrorStep.AUTHORIZATION,
        failed_at=ist(hour=9),
    )
    defaults.update(kwargs)
    return RiskItem(**defaults)


def make_customer(**kwargs) -> Customer:
    defaults = dict(customer_id="cust_1", tenure_days=400)
    defaults.update(kwargs)
    return Customer(**defaults)


def make_action(kind=ActionKind.PAYMENT_LINK_SMS, *, at=None, item=None, **kwargs) -> ProposedAction:
    item = item or make_item()
    at = at or ist()
    slots = {}
    template_id = None
    if ACTION_CHANNEL[kind] in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL):
        template_id = "RETRY_LINK_EN"
        slots = {"amount": "2500.00", "merchant": "Acme", "link": "https://rzp.io/i/AB12CD34"}
    defaults = dict(
        item_id=item.item_id,
        customer_id=item.customer_id,
        kind=kind,
        channel=ACTION_CHANNEL[kind],
        scheduled_at=at,
        amount_paise=item.outstanding_paise,
        idempotency_key=f"k:{kind.value}:{at.isoformat()}",
        template_id=template_id,
        slots=slots,
    )
    defaults.update(kwargs)
    return ProposedAction(**defaults)


def ctx(item=None, customer=None, now=None, **kwargs) -> PolicyContext:
    item = item or make_item()
    customer = customer or make_customer()
    return PolicyContext(item=item, customer=customer, now=now or ist(), **kwargs)


def engine() -> PolicyEngine:
    return PolicyEngine(list(rules.DEFAULT_RULES))


class ContactHoursTest(unittest.TestCase):
    """The RBI contact window is half-open: [08:00, 19:00)."""

    def check(self, hour, minute, second, expected_allowed):
        at = ist(hour=hour, minute=minute, second=second)
        verdict = rules.rbi_contact_hours(make_action(at=at), ctx(now=at))
        self.assertEqual(
            verdict.allowed,
            expected_allowed,
            f"{hour:02d}:{minute:02d}:{second:02d} IST expected allowed={expected_allowed}",
        )

    def test_boundaries(self):
        self.check(7, 59, 59, False)   # one second before the window opens
        self.check(8, 0, 0, True)      # the instant it opens, inclusive
        self.check(18, 59, 59, True)   # last permitted second
        self.check(19, 0, 0, False)    # the instant it closes, exclusive
        self.check(19, 0, 1, False)
        self.check(3, 0, 0, False)     # the classic 3 AM automated SMS
        self.check(23, 30, 0, False)

    def test_silent_retry_is_exempt(self):
        """No human is contacted, so the contact restriction does not apply."""
        item = make_item(method=Method.NACH)
        at = ist(hour=3)
        verdict = rules.rbi_contact_hours(make_action(ActionKind.MANDATE_DEBIT, at=at, item=item), ctx(item=item, now=at))
        self.assertTrue(verdict.allowed)

    def test_denial_offers_the_next_legal_instant(self):
        """Deferral lands inside the window, staggered rather than on the hour.

        The exact minute is a jitter derived from the item id (see
        ``deferred_start``), so this asserts the window rather than a timestamp.
        """
        at = ist(hour=22, minute=30)
        verdict = rules.rbi_contact_hours(make_action(at=at), ctx(now=at))
        self.assertFalse(verdict.allowed)
        self.assertIsNotNone(verdict.retry_at)
        local = verdict.retry_at.astimezone(IST)
        self.assertEqual(local.day, 11)  # next morning, not the same one
        self.assertGreaterEqual(local.hour, rules.CONTACT_START_HOUR)
        self.assertLess(local.hour, rules.CONTACT_START_HOUR + 2)

    def test_deferral_across_a_month_boundary(self):
        at = ist(year=2026, month=3, day=31, hour=23)
        verdict = rules.rbi_contact_hours(make_action(at=at), ctx(now=at))
        local = verdict.retry_at.astimezone(IST)
        self.assertEqual((local.year, local.month, local.day, local.hour), (2026, 4, 1, 8))

    def test_deferral_across_a_leap_day(self):
        at = ist(year=2028, month=2, day=28, hour=23)
        verdict = rules.rbi_contact_hours(make_action(at=at), ctx(now=at))
        local = verdict.retry_at.astimezone(IST)
        self.assertEqual((local.month, local.day), (2, 29))


class PreDebitNotificationTest(unittest.TestCase):
    """RBI requires the notice at least 24 hours before a mandate debit."""

    def verdict_for(self, notice_age_hours):
        now = ist(hour=12)
        item = make_item(method=Method.EMANDATE_UPI)
        if notice_age_hours is not None:
            item.prenotified_at = now - timedelta(hours=notice_age_hours)
        action = make_action(ActionKind.MANDATE_DEBIT, at=now, item=item)
        return rules.emandate_prenotification(action, ctx(item=item, now=now))

    def test_no_notice_is_refused(self):
        self.assertFalse(self.verdict_for(None).allowed)

    def test_boundary(self):
        self.assertFalse(self.verdict_for(23.99).allowed)
        self.assertTrue(self.verdict_for(24.0).allowed)
        self.assertTrue(self.verdict_for(24.01).allowed)

    def test_stale_notice_is_refused(self):
        """Our own stricter reading: a notice from three weeks ago informs nobody."""
        self.assertTrue(self.verdict_for(24 * 7).allowed)
        self.assertFalse(self.verdict_for(24 * 7 + 1).allowed)

    def test_too_recent_offers_the_maturity_time(self):
        verdict = self.verdict_for(10)
        self.assertIsNotNone(verdict.retry_at)

    def test_does_not_apply_to_one_off_rails(self):
        now = ist()
        item = make_item(method=Method.UPI)
        action = make_action(ActionKind.MANDATE_DEBIT, at=now, item=item)
        self.assertTrue(rules.emandate_prenotification(action, ctx(item=item, now=now)).allowed)


class SilentRetryRailTest(unittest.TestCase):
    """There is no standing authorisation to re-charge a one-off payment."""

    def test_one_off_rails_cannot_be_retried(self):
        for method in (Method.UPI, Method.CARD, Method.NETBANKING, Method.WALLET):
            item = make_item(method=method)
            action = make_action(ActionKind.SILENT_RETRY, item=item)
            self.assertFalse(
                rules.silent_retry_rail(action, ctx(item=item)).allowed,
                f"{method.value} should not be silently retryable",
            )

    def test_mandate_rails_can(self):
        for method in (Method.EMANDATE_UPI, Method.EMANDATE_CARD, Method.NACH):
            item = make_item(method=method)
            action = make_action(ActionKind.MANDATE_DEBIT, item=item)
            self.assertTrue(rules.silent_retry_rail(action, ctx(item=item)).allowed)


class AmountBindingTest(unittest.TestCase):
    """The single control that stops a model moving a different amount."""

    def test_matching_amount_passes(self):
        item = make_item(amount_paise=250000)
        action = make_action(item=item, amount_paise=250000)
        self.assertTrue(rules.amount_matches_ledger(action, ctx(item=item)).allowed)

    def test_any_other_amount_is_refused(self):
        item = make_item(amount_paise=250000)
        for amount in (1, 100, 249999, 250001, 999999999):
            action = make_action(item=item, amount_paise=amount)
            self.assertFalse(
                rules.amount_matches_ledger(action, ctx(item=item)).allowed,
                f"amount {amount} should not be permitted against a 250000 paise balance",
            )

    def test_non_positive_is_refused(self):
        item = make_item(amount_paise=250000)
        for amount in (0, -1, -250000):
            action = make_action(item=item, amount_paise=amount)
            self.assertFalse(rules.amount_matches_ledger(action, ctx(item=item)).allowed)

    def test_partial_payment_moves_the_target(self):
        item = make_item(amount_paise=250000)
        item.recovered_paise = 100000
        self.assertFalse(rules.amount_matches_ledger(make_action(item=item, amount_paise=250000), ctx(item=item)).allowed)
        self.assertTrue(rules.amount_matches_ledger(make_action(item=item, amount_paise=150000), ctx(item=item)).allowed)


class StoppingRuleTest(unittest.TestCase):
    def test_settled_items_are_left_alone(self):
        item = make_item(amount_paise=250000)
        item.recovered_paise = 250000
        self.assertFalse(rules.stop_when_settled(make_action(item=item), ctx(item=item)).allowed)

    def test_terminal_states_are_left_alone(self):
        for state in (ItemState.RECOVERED, ItemState.WRITTEN_OFF, ItemState.ESCALATED, ItemState.STOPPED):
            item = make_item()
            item.state = state
            self.assertFalse(rules.stop_when_settled(make_action(item=item), ctx(item=item)).allowed)

    def test_dispute_permits_only_escalation(self):
        customer = make_customer(disputed=True)
        for kind in (ActionKind.PAYMENT_LINK_SMS, ActionKind.VOICE_CALL, ActionKind.INCENTIVE_OFFER):
            self.assertFalse(rules.stop_on_dispute(make_action(kind), ctx(customer=customer)).allowed)
        self.assertTrue(
            rules.stop_on_dispute(make_action(ActionKind.HUMAN_ESCALATION), ctx(customer=customer)).allowed
        )

    def test_hardship_suspends_automated_collection(self):
        customer = make_customer(hardship=True)
        self.assertFalse(rules.stop_on_hardship(make_action(), ctx(customer=customer)).allowed)
        self.assertTrue(
            rules.stop_on_hardship(make_action(ActionKind.HUMAN_ESCALATION), ctx(customer=customer)).allowed
        )

    def test_promise_to_pay_is_honoured(self):
        now = ist(hour=12)
        item = make_item()
        item.promise_to_pay_at = now + timedelta(days=3)
        self.assertFalse(rules.honour_promise_to_pay(make_action(at=now, item=item), ctx(item=item, now=now)).allowed)
        after = now + timedelta(days=4)
        self.assertTrue(
            rules.honour_promise_to_pay(make_action(at=after, item=item), ctx(item=item, now=after)).allowed
        )


class OptOutTest(unittest.TestCase):
    def test_opt_out_is_per_customer_not_per_item(self):
        customer = make_customer(opted_out_channels={Channel.SMS})
        self.assertFalse(rules.respect_optout(make_action(ActionKind.PAYMENT_LINK_SMS), ctx(customer=customer)).allowed)
        # A different item for the same customer is equally barred.
        other = make_item(item_id="item_2")
        self.assertFalse(
            rules.respect_optout(make_action(ActionKind.PAYMENT_LINK_SMS, item=other), ctx(item=other, customer=customer)).allowed
        )

    def test_other_channels_remain_open(self):
        customer = make_customer(opted_out_channels={Channel.SMS})
        self.assertTrue(
            rules.respect_optout(make_action(ActionKind.PAYMENT_LINK_WHATSAPP), ctx(customer=customer)).allowed
        )

    def test_unreachable_channel_is_refused(self):
        customer = make_customer(has_phone=False)
        self.assertFalse(
            rules.channel_reachable(make_action(ActionKind.PAYMENT_LINK_SMS), ctx(customer=customer)).allowed
        )


class TemplateRuleTest(unittest.TestCase):
    def test_free_text_is_impossible(self):
        action = make_action()
        action.template_id = None
        self.assertFalse(rules.registered_template(action, ctx()).allowed)

    def test_unregistered_template_fails_closed_through_the_engine(self):
        """An unknown template id raises inside the rule; the engine must deny."""
        action = make_action()
        action.template_id = "NOT_A_REAL_TEMPLATE"
        decision = engine().evaluate(action, ctx())
        self.assertFalse(decision.allowed)
        self.assertIn("R-DLT-TEMPLATE", decision.denied_rules)

    def test_injected_newline_in_a_slot_is_refused(self):
        action = make_action()
        action.slots = dict(action.slots, merchant="Acme\nURGENT: pay here instead")
        self.assertFalse(rules.registered_template(action, ctx()).allowed)

    def test_promotional_content_needs_consent(self):
        action = make_action(ActionKind.INCENTIVE_OFFER)
        action.template_id = "INCENTIVE_EN"
        action.slots = {"merchant": "Acme", "discount": "50", "link": "https://rzp.io/i/AB12CD34"}
        self.assertFalse(rules.promotional_consent(action, ctx(consent_promotional=False)).allowed)
        self.assertTrue(rules.promotional_consent(action, ctx(consent_promotional=True)).allowed)

    def test_dnd_bars_promotional_even_with_consent(self):
        action = make_action(ActionKind.INCENTIVE_OFFER)
        action.template_id = "INCENTIVE_EN"
        action.slots = {"merchant": "Acme", "discount": "50", "link": "https://rzp.io/i/AB12CD34"}
        self.assertFalse(
            rules.promotional_consent(action, ctx(consent_promotional=True, dnd_registered=True)).allowed
        )


class IdempotencyTest(unittest.TestCase):
    def test_replayed_key_is_refused(self):
        action = make_action()
        used = frozenset({action.idempotency_key})
        self.assertFalse(rules.idempotency(action, ctx(used_idempotency_keys=used)).allowed)

    def test_missing_key_is_refused(self):
        action = make_action(idempotency_key="")
        self.assertFalse(rules.idempotency(action, ctx()).allowed)


class EscalationCapacityTest(unittest.TestCase):
    def test_capacity_is_enforced_and_queues(self):
        action = make_action(ActionKind.HUMAN_ESCALATION)
        allowed = rules.escalation_capacity(action, ctx(escalations_today=2, escalation_daily_cap=3))
        self.assertTrue(allowed.allowed)
        refused = rules.escalation_capacity(action, ctx(escalations_today=3, escalation_daily_cap=3))
        self.assertFalse(refused.allowed)
        self.assertIsNotNone(refused.retry_at, "a capacity denial must queue, not drop")

    def test_escalating_twice_is_refused(self):
        item = make_item()
        item.escalated_at = ist()
        action = make_action(ActionKind.HUMAN_ESCALATION, item=item)
        self.assertFalse(rules.escalate_once(action, ctx(item=item)).allowed)


class FailClosedTest(unittest.TestCase):
    def test_a_raising_rule_denies(self):
        def exploding_rule(action, context):
            raise RuntimeError("something went wrong deep inside")

        exploding_rule.rule_id = "R-BOOM"
        decision = PolicyEngine([exploding_rule]).evaluate(make_action(), ctx())
        self.assertFalse(decision.allowed, "a rule that raises must deny, never allow")
        self.assertIn("R-BOOM", decision.denied_rules)
        self.assertIn("failing closed", decision.denials[0].reason)

    def test_all_denials_are_recorded_not_just_the_first(self):
        """The audit trail should say every reason, not the first one hit."""
        customer = make_customer(disputed=True, opted_out_channels={Channel.SMS})
        at = ist(hour=23)
        decision = engine().evaluate(make_action(at=at), ctx(customer=customer, now=at))
        self.assertFalse(decision.allowed)
        self.assertGreaterEqual(len(decision.denials), 3)
        self.assertIn("R-RBI-HOURS", decision.denied_rules)
        self.assertIn("R-OPTOUT", decision.denied_rules)
        self.assertIn("R-STOP-DISPUTE", decision.denied_rules)


class DeferralSemanticsTest(unittest.TestCase):
    def test_a_purely_temporal_denial_offers_a_time(self):
        at = ist(hour=23)
        decision = engine().evaluate(make_action(at=at), ctx(now=at))
        self.assertFalse(decision.allowed)
        self.assertIsNotNone(decision.earliest_allowed_at)

    def test_a_structural_denial_offers_no_time(self):
        """Rescheduling cannot fix an opt-out, so suggesting a time would mislead."""
        customer = make_customer(opted_out_channels={Channel.SMS})
        decision = engine().evaluate(make_action(), ctx(customer=customer))
        self.assertFalse(decision.allowed)
        self.assertIsNone(decision.earliest_allowed_at)


class DisabledGateTest(unittest.TestCase):
    def test_ablation_allows_but_still_records(self):
        """The no-policy ablation must measure violations, not hide them."""
        at = ist(hour=3)
        decision = PolicyEngine(list(rules.DEFAULT_RULES), enabled=False).evaluate(
            make_action(at=at), ctx(now=at)
        )
        self.assertTrue(decision.allowed)
        self.assertIn("R-RBI-HOURS", decision.denied_rules)


if __name__ == "__main__":
    unittest.main()
