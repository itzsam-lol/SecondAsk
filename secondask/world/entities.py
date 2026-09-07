"""Domain types.

The central modelling decision of this project lives here: the separation
between what the agent can **observe** and the latent **blocker** that actually
determines whether money can be recovered.

Observable (comes off a real Razorpay webhook):
    method, amount, error_source, error_reason, error_step, timestamps,
    bank downtime feed, the customer's own contact history.

Latent (the simulator knows it; the agent never sees it):
    ``Blocker``, the true reason the money did not move.

These are deliberately *aliased*: a single observable error signature maps to a
distribution over blockers. ``error_source=bank, error_reason=payment_failed``
is usually transient infrastructure but is sometimes a dead instrument, and the
agent cannot tell which from the failure alone. That aliasing is the entire
reason this is an inference problem rather than a lookup table, and it is why a
fixed retry schedule loses: it cannot distinguish "retry in an hour and it will
work" from "retry a hundred times and it will never work".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Method(str, Enum):
    """Payment instrument. Values match Razorpay's ``method`` field."""

    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMANDATE_UPI = "emandate_upi"      # UPI Autopay
    EMANDATE_CARD = "emandate_card"    # card-on-file recurring
    NACH = "nach"                      # bank mandate
    INVOICE = "invoice"                # B2B receivable, no instrument on file

    @property
    def is_mandate(self) -> bool:
        """Mandate rails are subject to the RBI 24-hour pre-debit notification."""
        return self in (Method.EMANDATE_UPI, Method.EMANDATE_CARD, Method.NACH)

    @property
    def supports_silent_retry(self) -> bool:
        """Can we re-attempt without the customer being present?

        Only mandates can. A one-off UPI or card payment needs the payer to
        authenticate, so "retrying" it really means sending them a payment link,
        which is a *contact* and must be priced and gated as one. Conflating the
        two is the single most common modelling error in dunning systems.
        """
        return self.is_mandate


class ErrorSource(str, Enum):
    """Razorpay's ``error_source``. This is the highest-signal free feature we get.

    It answers a question a retry schedule cannot: whose fault was it? A
    ``GATEWAY`` failure is somebody else's outage and costs nothing to retry. A
    ``CUSTOMER`` failure means the customer must do something, and retrying
    without telling them is pure waste.
    """

    CUSTOMER = "customer"
    BUSINESS = "business"
    BANK = "bank"
    GATEWAY = "gateway"
    INTERNAL = "internal"
    NA = "NA"          # Razorpay really does emit this
    UNKNOWN = "unknown"  # our own bucket for absent/garbled values


class ErrorReason(str, Enum):
    """Failure reasons, named after Razorpay's error vocabulary."""

    PAYMENT_FAILED = "payment_failed"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_EXPIRED = "card_expired"
    CARD_DECLINED = "card_declined"
    CARD_DISABLED_ONLINE = "card_disabled_for_online_payments"
    INVALID_OTP = "invalid_otp"
    AUTHENTICATION_FAILED = "authentication_failed"
    PAYMENT_TIMED_OUT = "payment_timed_out"
    GATEWAY_TECHNICAL_ERROR = "gateway_technical_error"
    ISSUER_DOWN = "issuer_down"
    MANDATE_REVOKED = "mandate_revoked"
    MANDATE_NOT_FOUND = "mandate_not_found"
    ACCOUNT_CLOSED = "account_closed"
    ACCOUNT_FROZEN = "account_frozen"
    LIMIT_EXCEEDED = "limit_exceeded"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    INVOICE_OVERDUE = "invoice_overdue"
    UNKNOWN = "unknown"


class ErrorStep(str, Enum):
    INITIATION = "payment_initiation"
    AUTHENTICATION = "payment_authentication"
    AUTHORIZATION = "payment_authorization"
    RESPONSE = "payment_response"
    NA = "NA"


class Blocker(str, Enum):
    """LATENT. The true reason the money did not move.

    The agent never reads this field. It exists so that the simulator can answer
    the counterfactual question "would this action, at this time, have worked?",
    which is what makes offline measurement of rupees recovered possible at all.

    Each blocker responds to a genuinely different intervention, which is the
    point:
    """

    TRANSIENT_INFRA = "transient_infra"
    """Bank or gateway was degraded. Resolves on its own. A silent retry after
    resolution works. Contacting the customer is unnecessary and burns goodwill
    for nothing."""

    LIQUIDITY = "liquidity"
    """No money in the account. Retrying before payday always fails; retrying
    after payday usually works. Timing dominates; attempt count is irrelevant."""

    INSTRUMENT_DEAD = "instrument_dead"
    """Card expired, account closed, mandate revoked. Retries can *never*
    succeed. Only a customer action fixes it, so contact is mandatory and
    silent retries are 100% waste."""

    AUTH_FRICTION = "auth_friction"
    """Wrong OTP, 3DS drop-off, session timeout. The intent was there. A prompt
    nudge with a fresh payment link works well, and the window decays fast."""

    INTENT_LOST = "intent_lost"
    """Customer changed their mind. Only persuasion works; base rate is low and
    decays. Incentives help, repetition does not."""

    DISPUTE = "dispute"
    """Customer believes the charge is wrong or already paid. Automated contact
    actively harms: it escalates to a complaint. Must go to a human."""

    UNREACHABLE = "unreachable"
    """Wrong contact details or a hard opt-out. Nothing works. Recognising these
    early is worth real money because every rupee spent contacting them is
    certain loss."""


class ActionKind(str, Enum):
    """Everything the agent is allowed to propose."""

    WAIT = "wait"
    SILENT_RETRY = "silent_retry"
    PRENOTIFY = "prenotify"                      # RBI 24h pre-debit notice
    MANDATE_DEBIT = "mandate_debit"
    PAYMENT_LINK_SMS = "payment_link_sms"
    PAYMENT_LINK_WHATSAPP = "payment_link_whatsapp"
    PAYMENT_LINK_EMAIL = "payment_link_email"
    UPDATE_INSTRUMENT = "update_instrument"      # ask them to fix the card
    INCENTIVE_OFFER = "incentive_offer"
    VOICE_CALL = "voice_call"
    HUMAN_ESCALATION = "human_escalation"
    STOP = "stop"

    @property
    def is_contact(self) -> bool:
        """Does this reach the customer?

        Drives the RBI contact-window rule, frequency caps and the annoyance
        model. ``PRENOTIFY`` counts: it is a message to a human being, and the
        RBI contact-hour restriction covers SMS and instant messaging, not just
        voice.
        """
        return self in _CONTACT_ACTIONS

    @property
    def moves_money(self) -> bool:
        return self in (ActionKind.SILENT_RETRY, ActionKind.MANDATE_DEBIT)


_CONTACT_ACTIONS = frozenset(
    {
        ActionKind.PRENOTIFY,
        ActionKind.PAYMENT_LINK_SMS,
        ActionKind.PAYMENT_LINK_WHATSAPP,
        ActionKind.PAYMENT_LINK_EMAIL,
        ActionKind.UPDATE_INSTRUMENT,
        ActionKind.INCENTIVE_OFFER,
        ActionKind.VOICE_CALL,
    }
)


class Channel(str, Enum):
    NONE = "none"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    VOICE = "voice"
    HUMAN = "human"


ACTION_CHANNEL: dict[ActionKind, Channel] = {
    ActionKind.WAIT: Channel.NONE,
    ActionKind.SILENT_RETRY: Channel.NONE,
    ActionKind.MANDATE_DEBIT: Channel.NONE,
    ActionKind.PRENOTIFY: Channel.SMS,
    ActionKind.PAYMENT_LINK_SMS: Channel.SMS,
    ActionKind.PAYMENT_LINK_WHATSAPP: Channel.WHATSAPP,
    ActionKind.PAYMENT_LINK_EMAIL: Channel.EMAIL,
    ActionKind.UPDATE_INSTRUMENT: Channel.SMS,
    ActionKind.INCENTIVE_OFFER: Channel.WHATSAPP,
    ActionKind.VOICE_CALL: Channel.VOICE,
    ActionKind.HUMAN_ESCALATION: Channel.HUMAN,
    ActionKind.STOP: Channel.NONE,
}

# Cost per action in paise. Sources are documented in COMPLIANCE.md; these are
# ordinary Indian CPaaS list prices rounded to the nearest paisa. They are small
# numbers, which is exactly why naive systems ignore them, and why blasting
# 20,000 messages to recover the same money is a real, quantifiable loss.
ACTION_COST_PAISE: dict[ActionKind, int] = {
    ActionKind.WAIT: 0,
    ActionKind.SILENT_RETRY: 0,
    ActionKind.MANDATE_DEBIT: 0,
    ActionKind.PRENOTIFY: 18,        # DLT transactional SMS
    ActionKind.PAYMENT_LINK_SMS: 18,
    ActionKind.PAYMENT_LINK_WHATSAPP: 88,   # WhatsApp utility conversation
    ActionKind.PAYMENT_LINK_EMAIL: 2,
    ActionKind.UPDATE_INSTRUMENT: 18,
    ActionKind.INCENTIVE_OFFER: 88,
    ActionKind.VOICE_CALL: 4500,     # ~90s of agent-assisted outbound voice
    # A fully loaded ops hour is around 250 rupees and a recovery case takes
    # roughly 20 minutes, so ~85 rupees of direct cost plus overhead. The real
    # constraint on escalation is not this price though, it is capacity: see
    # R-ESCALATION-CAPACITY, which caps how many cases a team can take per day.
    ActionKind.HUMAN_ESCALATION: 15000,
    ActionKind.STOP: 0,
}


class ItemState(str, Enum):
    AT_RISK = "at_risk"
    IN_RECOVERY = "in_recovery"
    RECOVERED = "recovered"
    PARTIALLY_RECOVERED = "partially_recovered"
    WRITTEN_OFF = "written_off"
    ESCALATED = "escalated"
    STOPPED = "stopped"

    @property
    def is_terminal(self) -> bool:
        return self in (
            ItemState.RECOVERED,
            ItemState.WRITTEN_OFF,
            ItemState.ESCALATED,
            ItemState.STOPPED,
        )


class ReplyIntent(str, Enum):
    """The closed set the reply parser may emit.

    This enum is a security control, not a convenience. The parser's output is
    constrained to these values, so no amount of instruction-shaped text inside
    a customer message can cause the system to invent a new action. Note what is
    absent: there is no ``MARK_AS_PAID``. Settlement is only ever observed from
    a payment event, never asserted by a message.
    """

    NONE = "none"
    PROMISE_TO_PAY = "promise_to_pay"
    ALREADY_PAID = "already_paid"        # a *claim*, triggers verification only
    DISPUTE = "dispute"
    OPT_OUT = "opt_out"
    WRONG_NUMBER = "wrong_number"
    HARDSHIP = "hardship"
    NEEDS_HELP = "needs_help"
    UNINTELLIGIBLE = "unintelligible"


@dataclass
class Customer:
    """A payer.

    ``annoyance`` and opt-out state live on the *customer*, not the item. A
    customer with three failed payments must not receive three independent
    dunning sequences. Most naive implementations key everything by item and
    silently triple their contact volume for exactly the customers who are
    already having the worst experience.
    """

    customer_id: str
    has_phone: bool = True
    has_email: bool = True
    has_whatsapp: bool = True
    language: str = "en"
    tenure_days: int = 180
    prior_recoveries: int = 0
    prior_ignores: int = 0

    # Mutable state during a run
    annoyance: float = 0.0
    opted_out_channels: set[Channel] = field(default_factory=set)
    disputed: bool = False
    hardship: bool = False
    complained: bool = False
    churned: bool = False

    # Latent traits (simulator only)
    _responsiveness: float = 0.5
    _annoyance_tolerance: float = 3.0
    _payday_day: int = 1

    def channel_available(self, channel: Channel) -> bool:
        if channel in self.opted_out_channels:
            return False
        if channel == Channel.SMS or channel == Channel.VOICE:
            return self.has_phone
        if channel == Channel.WHATSAPP:
            return self.has_phone and self.has_whatsapp
        if channel == Channel.EMAIL:
            return self.has_email
        return True

    def to_dict(self) -> dict[str, Any]:
        """Public projection. Latent traits are excluded by construction."""
        return {
            "customer_id": self.customer_id,
            "language": self.language,
            "tenure_days": self.tenure_days,
            "prior_recoveries": self.prior_recoveries,
            "prior_ignores": self.prior_ignores,
            "annoyance": round(self.annoyance, 3),
            "opted_out": sorted(c.value for c in self.opted_out_channels),
            "disputed": self.disputed,
            "hardship": self.hardship,
        }


@dataclass
class RiskItem:
    """One unit of money at risk.

    Observable fields are what a real ``payment.failed`` webhook carries. The
    single underscore-prefixed field is latent and is stripped from every public
    projection.
    """

    item_id: str
    customer_id: str
    amount_paise: int
    method: Method
    error_source: ErrorSource
    error_reason: ErrorReason
    error_step: ErrorStep
    failed_at: datetime
    order_id: str = ""

    # B2B receivables only
    is_msme_supplier: bool = False
    invoice_due_at: Optional[datetime] = None
    agreement_days: int = 45

    # Mutable run state
    state: ItemState = ItemState.AT_RISK
    recovered_paise: int = 0
    attempts: int = 0
    contacts: int = 0
    last_contact_at: Optional[datetime] = None
    last_action_at: Optional[datetime] = None
    prenotified_at: Optional[datetime] = None
    promise_to_pay_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    cost_paise: int = 0

    # LATENT: simulator only, never exposed to any agent
    _blocker: Blocker = Blocker.INTENT_LOST
    _resolves_at: Optional[datetime] = None

    @property
    def outstanding_paise(self) -> int:
        """Never negative. An overpayment is clamped and flagged as an exception
        rather than silently producing a negative receivable."""
        return max(0, self.amount_paise - self.recovered_paise)

    def observable(self) -> dict[str, Any]:
        """Exactly what an agent is allowed to see.

        Written as an explicit allow-list rather than by deleting latent keys
        from ``__dict__``. A deny-list would leak any latent field added later;
        an allow-list fails safe.
        """
        return {
            "item_id": self.item_id,
            "customer_id": self.customer_id,
            "amount_paise": self.amount_paise,
            "outstanding_paise": self.outstanding_paise,
            "method": self.method.value,
            "error_source": self.error_source.value,
            "error_reason": self.error_reason.value,
            "error_step": self.error_step.value,
            "failed_at": self.failed_at,
            "state": self.state.value,
            "attempts": self.attempts,
            "contacts": self.contacts,
            "is_msme_supplier": self.is_msme_supplier,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.observable()
        data.update(
            {
                "recovered_paise": self.recovered_paise,
                "cost_paise": self.cost_paise,
                "order_id": self.order_id,
            }
        )
        return data
