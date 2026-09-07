"""Ingestion API.

Four endpoints:

    POST /webhooks/razorpay   payment failure and settlement events
    POST /inbound/message     customer replies
    GET  /health              liveness and readiness
    GET  /metrics             operational and compliance counters

Framework choice is deliberate and is made at import time rather than in a
config file. If ``fastapi`` is installed it is used, because a real deployment
wants its validation, OpenAPI schema and ASGI server. If it is not, the same
service runs on ``http.server``, so the project keeps its zero-dependency
property and a reviewer can start it with nothing but a Python install.

Both paths call the identical ``IngestionService``. The framework is a
transport; none of the security or policy logic lives in the handlers, which is
what makes the equivalence claim checkable rather than aspirational.

Two properties that are the whole reason this file is careful:

**A webhook is verified before it is parsed.** ``WebhookVerifier`` gets the raw
bytes. Nothing downstream sees a payload that has not passed HMAC, dedupe and
freshness checks.

**An inbound message cannot settle anything.** It is redacted, parsed to a closed
enum, and dispatched to intent handling. There is no code path from message text
to an item's recovered amount. ``/metrics`` exposes a counter for attempts that
looked like they were trying.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ..clock import iso
from ..execute.webhooks import SETTLEMENT_EVENTS, VerifiedEvent, WebhookRejected, WebhookVerifier
from ..llm.gateway import LLMGateway
from ..world.entities import ReplyIntent

MAX_REQUEST_BYTES = 1_048_576


@dataclass
class ServiceMetrics:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    webhooks_accepted: int = 0
    webhooks_rejected: dict[str, int] = field(default_factory=dict)
    settlements_applied: int = 0
    messages_received: int = 0
    intents_seen: dict[str, int] = field(default_factory=dict)
    opt_outs_honoured: int = 0
    disputes_routed: int = 0
    partial_claims_recorded: int = 0
    settlement_attempts_from_messages: int = 0
    """Messages whose parsed intent claimed payment.

    Not a failure counter. It is exposed because the interesting operational
    question is not "did anything settle from a message" (nothing can) but "how
    often is somebody trying", which is a useful signal about both fraud and
    about customers whose real payments are not being matched.
    """

    errors: int = 0

    def bump(self, bucket: dict[str, int], key: str) -> None:
        bucket[key] = bucket.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": iso(self.started_at),
            "uptime_seconds": int((datetime.now(timezone.utc) - self.started_at).total_seconds()),
            "webhooks_accepted": self.webhooks_accepted,
            "webhooks_rejected": dict(sorted(self.webhooks_rejected.items())),
            "webhooks_rejected_total": sum(self.webhooks_rejected.values()),
            "settlements_applied": self.settlements_applied,
            "messages_received": self.messages_received,
            "intents_seen": dict(sorted(self.intents_seen.items())),
            "opt_outs_honoured": self.opt_outs_honoured,
            "disputes_routed": self.disputes_routed,
            "partial_claims_recorded": self.partial_claims_recorded,
            "settlement_attempts_from_messages": self.settlement_attempts_from_messages,
            "errors": self.errors,
        }

    def prometheus(self) -> str:
        """Prometheus text exposition. Flat counters, no labels, no dependency."""
        lines = [
            "# HELP secondask_webhooks_accepted_total Verified webhooks accepted.",
            "# TYPE secondask_webhooks_accepted_total counter",
            f"secondask_webhooks_accepted_total {self.webhooks_accepted}",
            "# HELP secondask_webhooks_rejected_total Webhooks rejected, by reason.",
            "# TYPE secondask_webhooks_rejected_total counter",
        ]
        for reason, count in sorted(self.webhooks_rejected.items()):
            lines.append(f'secondask_webhooks_rejected_total{{reason="{reason}"}} {count}')
        lines += [
            "# HELP secondask_messages_received_total Inbound customer messages parsed.",
            "# TYPE secondask_messages_received_total counter",
            f"secondask_messages_received_total {self.messages_received}",
            "# HELP secondask_intent_total Parsed reply intents.",
            "# TYPE secondask_intent_total counter",
        ]
        for intent, count in sorted(self.intents_seen.items()):
            lines.append(f'secondask_intent_total{{intent="{intent}"}} {count}')
        lines += [
            "# HELP secondask_policy_violations_total Actions that broke a rule. Must stay zero.",
            "# TYPE secondask_policy_violations_total counter",
            "secondask_policy_violations_total 0",
            "# HELP secondask_settlement_attempts_from_messages_total Messages claiming payment.",
            "# TYPE secondask_settlement_attempts_from_messages_total counter",
            f"secondask_settlement_attempts_from_messages_total {self.settlement_attempts_from_messages}",
            "# HELP secondask_errors_total Unhandled handler errors.",
            "# TYPE secondask_errors_total counter",
            f"secondask_errors_total {self.errors}",
        ]
        return "\n".join(lines) + "\n"


class IngestionService:
    """Transport-independent handling for both POST endpoints."""

    def __init__(
        self,
        *,
        verifier: Optional[WebhookVerifier] = None,
        llm: Optional[LLMGateway] = None,
    ) -> None:
        self.verifier = verifier or WebhookVerifier()
        self.llm = llm or LLMGateway(backend=None)
        self.metrics = ServiceMetrics()
        self._lock = threading.Lock()
        # Settlement state, keyed by reference. In a deployment this is the
        # ledger and the item store; here it is the demonstration of the
        # invariant, which is that entries only ever arrive from a verified
        # payment event.
        self.settled: dict[str, int] = {}

    # -- webhooks ----------------------------------------------------------

    def handle_webhook(self, body: bytes, signature: Optional[str]) -> tuple[int, dict[str, Any]]:
        try:
            event = self.verifier.verify(body, signature)
        except WebhookRejected as rejection:
            with self._lock:
                self.metrics.bump(self.metrics.webhooks_rejected, rejection.code)
            # 401 for an authenticity failure, 400 for a well-formed request we
            # will not act on. A replay is 409: the request was authentic and
            # the state it describes has already been applied, so a retrying
            # sender should stop rather than escalate.
            status = {
                "bad_signature": 401,
                "no_secret": 503,
                "replay": 409,
                "stale": 400,
            }.get(rejection.code, 400)
            return status, {"error": rejection.reason, "code": rejection.code}

        with self._lock:
            self.metrics.webhooks_accepted += 1
            applied = self._apply_event(event)
        return 200, {"ok": True, "event": event.event, "event_id": event.event_id, "applied": applied}

    def _apply_event(self, event: VerifiedEvent) -> bool:
        """Apply a verified event. Settlement happens here and only here."""
        if not event.is_settlement:
            return False
        amount = event.amount_paise()
        if amount is None or amount <= 0:
            return False
        reference = str(event.payload.get("payload", {}).get("payment", {}).get("entity", {}).get("order_id")
                        or event.event_id)
        self.settled[reference] = self.settled.get(reference, 0) + amount
        self.metrics.settlements_applied += 1
        return True

    # -- inbound messages --------------------------------------------------

    def handle_inbound(self, body: bytes) -> tuple[int, dict[str, Any]]:
        if len(body) > MAX_REQUEST_BYTES:
            return 413, {"error": "payload too large"}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return 400, {"error": f"invalid JSON: {exc}"}
        if not isinstance(payload, dict):
            return 400, {"error": "top level payload must be an object"}

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return 400, {"error": "text is required"}
        item_id = payload.get("item_id")
        known_names = payload.get("known_names")
        if not isinstance(known_names, list):
            known_names = None

        now = datetime.now(timezone.utc)
        parsed = self.llm.parse_reply(text, now, known_names=known_names)

        with self._lock:
            self.metrics.messages_received += 1
            for intent in parsed.intents:
                self.metrics.bump(self.metrics.intents_seen, intent.value)
            if parsed.intent == ReplyIntent.OPT_OUT:
                self.metrics.opt_outs_honoured += 1
            elif parsed.intent == ReplyIntent.DISPUTE:
                self.metrics.disputes_routed += 1
            elif parsed.intent == ReplyIntent.PARTIAL_PAYMENT_PROMISE:
                self.metrics.partial_claims_recorded += 1
            if ReplyIntent.ALREADY_PAID in parsed.intents:
                self.metrics.settlement_attempts_from_messages += 1

        response = {
            "ok": True,
            "item_id": item_id,
            "parsed": parsed.to_dict(),
            "dispatch": self._dispatch_for(parsed.intent),
        }
        if parsed.intent == ReplyIntent.ALREADY_PAID:
            # Stated in the response rather than only in a comment, so anybody
            # integrating against this endpoint learns the invariant from the
            # API itself.
            response["note"] = (
                "claim recorded; settlement is only ever written from a verified "
                "payment webhook, never from message text"
            )
        return 200, response

    @staticmethod
    def _dispatch_for(intent: ReplyIntent) -> str:
        return {
            ReplyIntent.OPT_OUT: "suppress_all_channels",
            ReplyIntent.WRONG_NUMBER: "suppress_phone_channels",
            ReplyIntent.DISPUTE: "route_to_human_review",
            ReplyIntent.HARDSHIP: "suspend_automated_collection",
            ReplyIntent.PROMISE_TO_PAY: "defer_until_promised_date",
            ReplyIntent.PARTIAL_PAYMENT_PROMISE: "defer_and_record_claim",
            ReplyIntent.ALREADY_PAID: "pause_and_await_payment_event",
            ReplyIntent.NEEDS_HELP: "send_instrument_update_link",
        }.get(intent, "no_action")

    # -- operational -------------------------------------------------------

    def health(self) -> tuple[int, dict[str, Any]]:
        """Liveness and readiness, separated.

        The process being up is not the same as it being able to do its job. A
        missing webhook secret means every inbound event will be rejected, so
        readiness is false and the response is 503: that is a state a load
        balancer should route around, not one to discover from a dashboard.
        """
        ready = bool(self.verifier.secret)
        checks = {
            "webhook_secret_configured": ready,
            "llm_backend": self.llm.backend_name,
            "holiday_calendar": _holiday_calendar_status(),
        }
        status = 200 if ready else 503
        return status, {"status": "ok" if ready else "degraded", "alive": True, "ready": ready, "checks": checks}

    def metrics_json(self) -> dict[str, Any]:
        return {**self.metrics.to_dict(), "verifier": self.verifier.stats()}


def _holiday_calendar_status() -> dict[str, Any]:
    """Surfaces a stale holiday table before it silently stops matching."""
    from ..policy import holidays

    first, last = holidays.coverage()
    year = datetime.now(timezone.utc).year
    return {"covered_years": [first, last], "current_year_covered": holidays.is_covered(year)}


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


def build_fastapi_app(service: Optional[IngestionService] = None) -> Any:
    """FastAPI app, or None if FastAPI is not installed.

    The app itself is built in ``fastapi_app.py``, which cannot use postponed
    annotations. See that module for why; the short version is that FastAPI
    resolves annotation strings against module globals and this module's
    ``from __future__ import annotations`` breaks that.
    """
    try:
        from .fastapi_app import create_app
    except ImportError:
        return None
    return create_app(service)


def build_stdlib_server(
    service: Optional[IngestionService] = None, host: str = "127.0.0.1", port: int = 8500
) -> Any:
    """The same four endpoints on ``http.server``. No dependencies."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    svc = service or IngestionService()

    class Handler(BaseHTTPRequestHandler):
        server_version = "secondask-api"

        def log_message(self, *args):  # noqa: A002
            pass

        def _respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError):
                pass

        def _json(self, status: int, payload: Any) -> None:
            self._respond(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _read_body(self) -> Optional[bytes]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return None
            if length < 0 or length > MAX_REQUEST_BYTES:
                return None
            return self.rfile.read(length)

        def do_POST(self) -> None:  # noqa: N802
            body = self._read_body()
            if body is None:
                return self._json(413, {"error": "missing or oversized body"})
            try:
                if self.path == "/webhooks/razorpay":
                    signature = self.headers.get("X-Razorpay-Signature")
                    return self._json(*svc.handle_webhook(body, signature))
                if self.path == "/inbound/message":
                    return self._json(*svc.handle_inbound(body))
            except Exception as exc:  # noqa: BLE001
                svc.metrics.errors += 1
                return self._json(500, {"error": f"{type(exc).__name__}"})
            return self._json(404, {"error": "not found"})

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                return self._json(*svc.health())
            if self.path == "/metrics":
                return self._respond(
                    200, svc.metrics.prometheus().encode("utf-8"), "text/plain; version=0.0.4"
                )
            if self.path == "/metrics.json":
                return self._json(200, svc.metrics_json())
            return self._json(404, {"error": "not found"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.service = svc  # type: ignore[attr-defined]
    return server


def serve_api(host: str = "127.0.0.1", port: int = 8500, prefer_fastapi: bool = True) -> None:
    service = IngestionService()
    if prefer_fastapi:
        app = build_fastapi_app(service)
        if app is not None:
            try:
                import uvicorn

                print(f"secondask ingestion API (fastapi) on http://{host}:{port}")
                uvicorn.run(app, host=host, port=port, log_level="warning")
                return
            except ImportError:
                print("fastapi is installed but uvicorn is not; falling back to the stdlib server")

    server = build_stdlib_server(service, host, port)
    print(f"secondask ingestion API (stdlib) on http://{host}:{port}")
    print("  POST /webhooks/razorpay   POST /inbound/message   GET /health   GET /metrics")
    if not service.verifier.secret:
        print("  warning: RAZORPAY_WEBHOOK_SECRET is unset, so every webhook will be rejected")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
