"""Razorpay client.

Two modes, same interface.

``mock`` (default)
    A faithful reimplementation of the request and response shapes, including
    the error envelope, driven by the virtual clock. It deterministically injects
    5xx and 429 responses so the retry path, the backoff and the circuit breaker
    are exercised on every run rather than only when Razorpay happens to be
    unwell. Nobody needs credentials to reproduce the benchmark.

``live_test``
    Real HTTPS calls to ``api.razorpay.com`` with test-mode keys from
    ``RAZORPAY_KEY_ID`` / ``RAZORPAY_KEY_SECRET``. Test mode moves no real money.

Built on ``urllib`` rather than ``requests`` or the Razorpay SDK, so the project
keeps its zero-dependency property.

Idempotency is enforced client side. Razorpay does not expose a universal
idempotency header across these endpoints, so the client keys completed
responses by our own idempotency key and returns the cached response on replay.
That is what stops a socket timeout, where the request actually succeeded, from
creating a second payment link for the same failure. Being explicit about this
matters: it is our guarantee, not the gateway's.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..clock import iso
from ..rng import crn_uniform
from .circuit import CircuitBreaker, CircuitOpen, backoff_delay

API_BASE = "https://api.razorpay.com/v1"


class RazorpayError(RuntimeError):
    """A call failed after the retry budget was exhausted."""

    def __init__(self, message: str, *, status: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    refused_by_breaker: int = 0
    idempotent_hits: int = 0
    injected_5xx: int = 0
    injected_429: int = 0
    total_backoff_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "failures": self.failures,
            "refused_by_breaker": self.refused_by_breaker,
            "idempotent_hits": self.idempotent_hits,
            "injected_5xx": self.injected_5xx,
            "injected_429": self.injected_429,
            "total_backoff_seconds": round(self.total_backoff_seconds, 2),
        }


class RazorpayClient:
    def __init__(
        self,
        mode: str = "mock",
        *,
        seed: int = 0,
        failure_rate: float = 0.06,
        rate_limit_rate: float = 0.02,
        max_attempts: int = 4,
        breaker: Optional[CircuitBreaker] = None,
        timeout: float = 10.0,
    ) -> None:
        if mode not in ("mock", "live_test"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.seed = seed
        self.failure_rate = failure_rate
        self.rate_limit_rate = rate_limit_rate
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.breaker = breaker or CircuitBreaker(name="razorpay")
        self.stats = CallStats()
        self._idempotent: dict[str, dict[str, Any]] = {}
        self._counter = 0

        self._key_id = os.environ.get("RAZORPAY_KEY_ID", "")
        self._key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
        if mode == "live_test":
            if not self._key_id or not self._key_secret:
                raise RazorpayError(
                    "live_test mode needs RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in the environment"
                )
            if not self._key_id.startswith("rzp_test_"):
                # Refusing a live key is a safety control, not a convenience.
                # A live key here would create real payment links against a real
                # merchant account.
                raise RazorpayError(
                    f"refusing to run with key id {self._key_id[:12]}..., only rzp_test_ keys are permitted"
                )

    # -- public API ---------------------------------------------------------

    def create_payment_link(
        self,
        *,
        amount_paise: int,
        description: str,
        reference_id: str,
        idempotency_key: str,
        now: datetime,
        expire_by: Optional[datetime] = None,
        notify_sms: bool = False,
        notify_email: bool = False,
    ) -> dict[str, Any]:
        """Create a payment link for the outstanding amount.

        ``notify_*`` default to False on purpose. Razorpay can send the
        notification itself, but then the send would bypass our policy engine,
        our frequency caps and our contact window. Delivery stays on our side of
        the gate.
        """
        if amount_paise <= 0:
            raise RazorpayError(f"refusing to create a payment link for {amount_paise} paise")
        payload: dict[str, Any] = {
            "amount": amount_paise,
            "currency": "INR",
            "description": description[:255],
            "reference_id": reference_id,
            "notify": {"sms": notify_sms, "email": notify_email},
            "reminder_enable": False,
        }
        if expire_by is not None:
            # Razorpay wants a Unix timestamp, and rejects anything less than
            # 15 minutes out. Clamp rather than let the call fail.
            floor = now + timedelta(minutes=20)
            payload["expire_by"] = int(max(expire_by, floor).timestamp())
        return self._request("POST", "/payment_links", payload, idempotency_key, now)

    def fetch_payment_link(self, link_id: str, *, now: datetime) -> dict[str, Any]:
        return self._request("GET", f"/payment_links/{link_id}", None, f"fetch:{link_id}", now)

    def create_order(
        self, *, amount_paise: int, receipt: str, idempotency_key: str, now: datetime
    ) -> dict[str, Any]:
        payload = {"amount": amount_paise, "currency": "INR", "receipt": receipt[:40]}
        return self._request("POST", "/orders", payload, idempotency_key, now)

    # -- transport ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]],
        idempotency_key: str,
        now: datetime,
    ) -> dict[str, Any]:
        if idempotency_key in self._idempotent:
            self.stats.idempotent_hits += 1
            return self._idempotent[idempotency_key]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.breaker.before_call(now)
            except CircuitOpen as exc:
                self.stats.refused_by_breaker += 1
                raise RazorpayError(str(exc), status=503, retryable=True) from exc

            self.stats.calls += 1
            try:
                if self.mode == "mock":
                    response = self._mock_call(method, path, payload, idempotency_key, attempt, now)
                else:
                    response = self._http_call(method, path, payload)
            except RazorpayError as exc:
                last_error = exc
                self.breaker.on_failure(now)
                self.stats.failures += 1
                if not exc.retryable or attempt == self.max_attempts:
                    raise
                self.stats.retries += 1
                self.stats.total_backoff_seconds += backoff_delay(attempt, idempotency_key)
                continue

            self.breaker.on_success()
            self._idempotent[idempotency_key] = response
            return response

        raise RazorpayError(f"exhausted {self.max_attempts} attempts: {last_error}", retryable=True)

    def _http_call(self, method: str, path: str, payload: Optional[dict[str, Any]]) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        token = base64.b64encode(f"{self._key_id}:{self._key_secret}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", "secondask/0.1")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            # 429 and 5xx are worth retrying. A 4xx is our bug and retrying it
            # just burns the rate limit we were probably already hitting.
            retryable = exc.code == 429 or exc.code >= 500
            raise RazorpayError(f"HTTP {exc.code}: {body[:300]}", status=exc.code, retryable=retryable) from exc
        except urllib.error.URLError as exc:
            raise RazorpayError(f"network error: {exc.reason}", status=0, retryable=True) from exc
        except TimeoutError as exc:
            # A timeout is the dangerous case: the request may well have
            # succeeded. Retrying is safe only because the idempotency cache
            # above collapses the duplicate.
            raise RazorpayError("request timed out", status=0, retryable=True) from exc

    def _mock_call(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]],
        idempotency_key: str,
        attempt: int,
        now: datetime,
    ) -> dict[str, Any]:
        roll = crn_uniform("rzp", self.seed, idempotency_key, attempt, path)
        if roll < self.rate_limit_rate:
            self.stats.injected_429 += 1
            raise RazorpayError(
                json.dumps({"error": {"code": "RATE_LIMIT_ERROR", "description": "Too many requests"}}),
                status=429,
                retryable=True,
            )
        if roll < self.rate_limit_rate + self.failure_rate:
            self.stats.injected_5xx += 1
            raise RazorpayError(
                json.dumps({"error": {"code": "SERVER_ERROR", "description": "The server encountered an error"}}),
                status=502,
                retryable=True,
            )

        self._counter += 1
        suffix = f"{self._counter:08d}"
        if path == "/payment_links" and method == "POST":
            assert payload is not None
            link_id = f"plink_MOCK{suffix}"
            return {
                "id": link_id,
                "entity": "payment_link",
                "status": "created",
                "amount": payload["amount"],
                "amount_paid": 0,
                "currency": "INR",
                "description": payload.get("description", ""),
                "reference_id": payload.get("reference_id", ""),
                "short_url": f"https://rzp.io/i/MOCK{suffix}",
                "created_at": int(now.timestamp()),
                "expire_by": payload.get("expire_by"),
            }
        if path.startswith("/payment_links/") and method == "GET":
            return {
                "id": path.rsplit("/", 1)[-1],
                "entity": "payment_link",
                "status": "created",
                "amount_paid": 0,
            }
        if path == "/orders" and method == "POST":
            assert payload is not None
            return {
                "id": f"order_MOCK{suffix}",
                "entity": "order",
                "amount": payload["amount"],
                "amount_paid": 0,
                "currency": "INR",
                "receipt": payload.get("receipt", ""),
                "status": "created",
                "created_at": int(now.timestamp()),
            }
        raise RazorpayError(f"mock has no handler for {method} {path}", status=404, retryable=False)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "stats": self.stats.to_dict(), "breaker": self.breaker.to_dict()}
