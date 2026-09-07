"""Inbound webhook verification.

A webhook endpoint that accepts unauthenticated JSON is an endpoint where
anybody on the internet can tell your system a payment succeeded. For a recovery
agent that is the whole game: forge one `payment.captured` and the item is marked
settled, collection stops, and the money never arrived.

So verification happens **before** the payload is parsed into anything the rest
of the system will act on, and certainly before anything reaches the ledger.

Three properties that are easy to get wrong and are each a real vulnerability:

**Verify the raw bytes.** The signature covers the exact body Razorpay sent.
Parsing to a dict and re-serialising produces different bytes (key order,
separators, unicode escaping) and the comparison then fails for good payloads
and, worse, tempts somebody to "fix" it by verifying the re-serialised form.
``verify_signature`` takes ``bytes`` and there is no overload that takes a dict.

**Compare in constant time.** ``==`` on a hex digest leaks how many leading
characters matched through timing, which is enough to forge a signature byte by
byte given enough attempts. ``hmac.compare_digest`` does not.

**Replay is not authenticity.** A validly signed payload captured off the wire
stays validly signed forever. Signature checking alone lets an attacker, or a
buggy retry, replay a settlement event. Event ids are therefore deduplicated and
an optional freshness window is enforced.

The failure mode is closed throughout: any missing secret, malformed header,
unparseable body or unknown event type results in rejection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

MAX_BODY_BYTES = 1_048_576  # 1 MiB. Razorpay payloads are a few KB.
DEFAULT_TOLERANCE_SECONDS = 300


class WebhookRejected(Exception):
    """Raised when a webhook fails verification. Never carries the secret."""

    def __init__(self, reason: str, *, code: str = "invalid") -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


def compute_signature(body: bytes, secret: str) -> str:
    """HMAC-SHA256 of the raw body, hex encoded. Razorpay's scheme."""
    if not isinstance(body, (bytes, bytearray)):
        raise TypeError("signature must be computed over raw bytes, not a parsed object")
    return hmac.new(secret.encode("utf-8"), bytes(body), hashlib.sha256).hexdigest()


def verify_signature(body: bytes, signature: str | None, secret: str | None) -> bool:
    """Constant-time signature check. Returns False rather than raising."""
    if not secret or not signature:
        return False
    if not isinstance(body, (bytes, bytearray)):
        return False
    try:
        expected = compute_signature(bytes(body), secret)
    except (TypeError, ValueError):
        return False
    # compare_digest raises on non-ASCII, which a hostile header can contain.
    try:
        return hmac.compare_digest(expected, signature.strip())
    except (TypeError, ValueError):
        return False


def verify_payment_signature(
    order_id: str, payment_id: str, signature: str | None, secret: str | None
) -> bool:
    """Checkout handback verification: HMAC over ``order_id|payment_id``.

    A different scheme from the webhook one, on the same secret. Kept here so
    both live next to their shared caveats rather than one being reinvented
    somewhere else with ``==``.
    """
    if not secret or not signature:
        return False
    payload = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(expected, signature.strip())
    except (TypeError, ValueError):
        return False


# Events this system will act on. An unknown event is rejected rather than
# ignored, so a Razorpay-side addition surfaces as a counter instead of being
# silently dropped.
KNOWN_EVENTS = frozenset({
    "payment.captured",
    "payment.failed",
    "payment.authorized",
    "payment_link.paid",
    "payment_link.expired",
    "subscription.charged",
    "subscription.halted",
    "refund.created",
    "refund.processed",
    "payment.downtime.started",
    "payment.downtime.updated",
    "payment.downtime.resolved",
})

# Only these can move an item to settled. Listed explicitly rather than inferred
# from the event name, because "paid" appearing in a string is not authorisation.
SETTLEMENT_EVENTS = frozenset({"payment.captured", "payment_link.paid", "subscription.charged"})


@dataclass
class VerifiedEvent:
    """A webhook that passed every check. Only this type reaches the runtime."""

    event_id: str
    event: str
    created_at: datetime
    payload: dict[str, Any]
    raw_bytes: int = 0

    @property
    def is_settlement(self) -> bool:
        return self.event in SETTLEMENT_EVENTS

    def amount_paise(self) -> Optional[int]:
        """Amount from the payload, as an integer, or None.

        Returns None rather than guessing on anything that is not already an
        integer. A float here would be a currency bug, and a string that happens
        to parse is not worth the risk given this value can drive settlement.
        """
        entity = self.payload.get("payload", {})
        for key in ("payment", "payment_link", "subscription", "refund"):
            node = entity.get(key, {}).get("entity")
            if isinstance(node, dict):
                amount = node.get("amount")
                if isinstance(amount, bool):
                    return None
                if isinstance(amount, int):
                    return amount
        return None

    def to_dict(self) -> dict[str, Any]:
        from ..clock import iso

        return {
            "event_id": self.event_id,
            "event": self.event,
            "created_at": iso(self.created_at),
            "is_settlement": self.is_settlement,
            "amount_paise": self.amount_paise(),
        }


@dataclass
class WebhookVerifier:
    """Verifies, deduplicates and freshness-checks inbound webhooks."""

    secret: str = field(default_factory=lambda: os.environ.get("RAZORPAY_WEBHOOK_SECRET", ""))
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS
    enforce_freshness: bool = False
    max_body_bytes: int = MAX_BODY_BYTES

    _seen: set[str] = field(default_factory=set, repr=False)
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def _reject(self, code: str, reason: str) -> "WebhookRejected":
        self.rejected[code] = self.rejected.get(code, 0) + 1
        return WebhookRejected(reason, code=code)

    def verify(
        self,
        body: bytes,
        signature: str | None,
        *,
        now: Optional[datetime] = None,
    ) -> VerifiedEvent:
        """Verify and parse. Raises ``WebhookRejected`` on any failure.

        Order matters and is deliberate: size, then signature, then parse. The
        body is never parsed before the signature is confirmed, so a hostile
        payload cannot reach the JSON decoder on an unauthenticated request.
        """
        if not self.secret:
            # Fail closed. An unset secret must not mean "accept everything",
            # which is exactly what a truthiness check on the signature would do.
            raise self._reject("no_secret", "RAZORPAY_WEBHOOK_SECRET is not configured")

        if body is None or not isinstance(body, (bytes, bytearray)):
            raise self._reject("bad_body", "body must be raw bytes")
        if len(body) > self.max_body_bytes:
            raise self._reject("too_large", f"body is {len(body)} bytes, limit is {self.max_body_bytes}")

        if not verify_signature(bytes(body), signature, self.secret):
            raise self._reject("bad_signature", "signature does not match the request body")

        try:
            payload = json.loads(bytes(body).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise self._reject("bad_json", f"body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise self._reject("bad_json", "top level payload must be an object")

        event = payload.get("event")
        if not isinstance(event, str) or event not in KNOWN_EVENTS:
            raise self._reject("unknown_event", f"unrecognised event {event!r}")

        # Razorpay sends x-razorpay-event-id in the header; the body carries a
        # created_at. Fall back to a digest of the body so dedupe still works
        # when the id is absent, rather than treating every replay as new.
        event_id = payload.get("id") or payload.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            event_id = hashlib.sha256(bytes(body)).hexdigest()

        created_raw = payload.get("created_at")
        if isinstance(created_raw, int) and not isinstance(created_raw, bool):
            created_at = datetime.fromtimestamp(created_raw, tz=timezone.utc)
        else:
            created_at = now or datetime.now(timezone.utc)

        if self.enforce_freshness:
            reference = now or datetime.now(timezone.utc)
            age = abs((reference - created_at).total_seconds())
            if age > self.tolerance_seconds:
                raise self._reject(
                    "stale",
                    f"event is {age:.0f}s from now, tolerance is {self.tolerance_seconds}s",
                )

        if event_id in self._seen:
            raise self._reject("replay", f"event {event_id} has already been processed")
        self._seen.add(event_id)

        self.accepted += 1
        return VerifiedEvent(
            event_id=event_id,
            event=event,
            created_at=created_at,
            payload=payload,
            raw_bytes=len(body),
        )

    def stats(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
            "rejected_total": sum(self.rejected.values()),
            "distinct_events_seen": len(self._seen),
            "secret_configured": bool(self.secret),
        }
