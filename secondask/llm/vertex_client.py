"""Vertex AI backend.

Same Gemini models as ``gemini_client``, reached through Google Cloud rather
than AI Studio. The differences that matter:

* **OAuth, not an API key.** Credentials come from Application Default
  Credentials, so ``gcloud auth application-default login`` is the whole setup
  and there is no long-lived secret sitting in a file for somebody to commit.
* **Project-scoped billing.** Usage lands on a GCP project, which is where a
  cloud credit actually lives.
* **Access tokens expire**, roughly hourly, so they are cached and refreshed on
  a 401 rather than shelled out for on every call.

Two workload-specific choices, both measured rather than assumed:

**Thinking is disabled.** The default flash model spent 51 thinking tokens to
produce 9 output tokens classifying "band karo ye message bhejna". This is a
closed-set classification with a fixed output schema; there is nothing to reason
about, and at thousands of calls per batch that overhead is most of the bill and
most of the latency.

**JSON mode is on, and the repair loop stays.** Even with
``responseMimeType: application/json`` the observed response was wrapped in a
markdown fence. The gateway's extractor handles that, which is precisely why the
gateway validates rather than trusting the provider's guarantee.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .gateway import RateLimited
from .gemini_client import _first_text

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "global"

# Tried in order if the configured model is unavailable in the project.
FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")

# Refresh a little before the nominal hour, so a long batch never trips over an
# expiry mid-call.
TOKEN_TTL_SECONDS = 45 * 60

# Adaptive throttle.
#
# A fresh GCP project has a low default per-minute quota for Gemini, and the
# measured failure mode is not a crash but a 429 storm: fire twenty calls back to
# back and nineteen of them are refused. Retrying harder makes it worse.
#
# So the client paces itself. It starts at a conservative interval, lengthens it
# on every rate limit, and shortens it slowly after sustained success. The result
# is that a long batch settles near whatever the project's real quota is without
# anybody having to look it up.
MIN_INTERVAL_SECONDS = 0.35
MAX_INTERVAL_SECONDS = 6.0
BACKOFF_FACTOR = 1.6
RECOVERY_FACTOR = 0.93
RECOVERY_AFTER = 12


class _Retryable(RuntimeError):
    pass


class _RateLimited(_Retryable):
    """Provider asked us to slow down. Transient, and never a reason to give up."""


class VertexBackend:
    name = "vertex"

    def __init__(
        self,
        *,
        project: str = "",
        location: str = "",
        model: str = "",
        timeout: float = 45.0,
        max_retries: int = 3,
        thinking_budget: int = 0,
    ) -> None:
        self.project = project or _default_project()
        if not self.project:
            raise RuntimeError(
                "no GCP project: set GOOGLE_CLOUD_PROJECT or run 'gcloud config set project PROJECT_ID'"
            )
        self.location = location or os.environ.get("SECONDASK_VERTEX_LOCATION", DEFAULT_LOCATION)
        self.model = model or os.environ.get("SECONDASK_VERTEX_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking_budget = thinking_budget
        self.name = f"vertex:{self.model}"

        self._token = ""
        self._token_at = 0.0
        self._resolved = False

        self._interval = MIN_INTERVAL_SECONDS
        self._last_call_at = 0.0
        self._since_limit = 0
        self.rate_limited = 0

        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0

    # -- auth ---------------------------------------------------------------

    def _access_token(self, force: bool = False) -> str:
        """Cached ADC access token.

        Shelling out to gcloud costs a second or two, which is irrelevant once an
        hour and ruinous on every one of several thousand calls.
        """
        if not force and self._token and (time.time() - self._token_at) < TOKEN_TTL_SECONDS:
            return self._token
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, shell=True, timeout=120,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                "could not obtain an access token; run 'gcloud auth application-default login'. "
                + result.stderr.strip()[:200]
            )
        self._token = result.stdout.strip()
        self._token_at = time.time()
        return self._token

    # -- pacing -------------------------------------------------------------

    def _pace(self) -> None:
        gap = time.time() - self._last_call_at
        if gap < self._interval:
            time.sleep(self._interval - gap)
        self._last_call_at = time.time()

    def _on_rate_limit(self) -> None:
        self.rate_limited += 1
        self._since_limit = 0
        self._interval = min(MAX_INTERVAL_SECONDS, self._interval * BACKOFF_FACTOR)

    def _on_success(self) -> None:
        self._since_limit += 1
        if self._since_limit >= RECOVERY_AFTER:
            self._since_limit = 0
            self._interval = max(MIN_INTERVAL_SECONDS, self._interval * RECOVERY_FACTOR)

    def _endpoint(self, model: str) -> str:
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/v1/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{model}:generateContent"
        )

    # -- transport ----------------------------------------------------------

    def complete_json(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        config: dict[str, Any] = {
            # Zero temperature for reproducibility. Classification and slot
            # filling, not a creative task, and a run that cannot be replayed
            # cannot be audited.
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "candidateCount": 1,
            "responseMimeType": "application/json",
        }
        if self.thinking_budget == 0:
            config["thinkingConfig"] = {"thinkingBudget": 0}
        elif self.thinking_budget > 0:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": config,
        }).encode("utf-8")

        self.calls += 1
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._pace()
            try:
                text = self._post(self.model, body, refresh_token=attempt > 1)
                self._on_success()
                return text
            except _RateLimited as exc:
                last = exc
                self._on_rate_limit()
                if attempt == self.max_retries:
                    raise RateLimited(str(exc)) from exc
                time.sleep(self._interval)
            except _Retryable as exc:
                last = exc
                if attempt == self.max_retries:
                    raise RuntimeError(f"vertex call failed after {attempt} attempts: {exc}") from exc
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
            except RuntimeError as exc:
                if "not found" in str(exc).lower() and not self._resolved:
                    resolved = self._resolve_model(body)
                    if resolved:
                        self.model = resolved
                        self.name = f"vertex:{resolved}"
                        self._resolved = True
                        continue
                raise
        raise RuntimeError(f"vertex call failed: {last}")

    def _post(self, model: str, body: bytes, *, refresh_token: bool = False) -> str:
        request = urllib.request.Request(self._endpoint(model), data=body, method="POST")
        request.add_header("Authorization", "Bearer " + self._access_token(force=refresh_token))
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                data = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 401:
                # Token expired mid-batch. Retryable, and the retry forces a
                # refresh rather than presenting the same dead token again.
                raise _Retryable(f"HTTP 401 (token refresh): {detail}")
            if exc.code == 404:
                raise RuntimeError(f"model {model!r} not found in {self.location}: {detail}")
            if exc.code == 429:
                raise _RateLimited(f"HTTP 429: {detail}")
            if exc.code in (500, 502, 503, 504):
                raise _Retryable(f"HTTP {exc.code}: {detail}")
            raise RuntimeError(f"vertex HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise _Retryable(f"transport error: {exc}")

        usage = data.get("usageMetadata") or {}
        self.prompt_tokens += int(usage.get("promptTokenCount") or 0)
        self.output_tokens += int(usage.get("candidatesTokenCount") or 0)
        self.thinking_tokens += int(usage.get("thoughtsTokenCount") or 0)
        return _first_text(data)

    def _resolve_model(self, body: bytes) -> str:
        for candidate in FALLBACK_MODELS:
            if candidate == self.model:
                continue
            try:
                self._post(candidate, body)
                return candidate
            except Exception:  # noqa: BLE001
                continue
        return ""

    def usage(self) -> dict[str, Any]:
        """Real token counts, from the API rather than estimated.

        ``thinking_tokens`` is reported separately because it is billed as output
        and is invisible in the response text, which makes it the easiest line
        item to be surprised by.
        """
        return {
            "backend": self.name,
            "project": self.project,
            "location": self.location,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "rate_limited": self.rate_limited,
            "settled_interval_seconds": round(self._interval, 3),
        }


def _default_project() -> str:
    for name in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, shell=True, timeout=60,
        )
        value = result.stdout.strip()
        if value and value != "(unset)":
            return value
    except Exception:  # noqa: BLE001
        pass
    return ""


def available() -> bool:
    """Whether Vertex looks usable, without making a network call."""
    if not _default_project():
        return False
    adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc and os.path.exists(adc):
        return True
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
    for candidate in (
        os.path.join(appdata, "gcloud", "application_default_credentials.json"),
        os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
    ):
        if os.path.exists(candidate):
            return True
    return False


def build_backend() -> Optional[VertexBackend]:
    """Return a backend if Vertex looks configured, else None. Never raises."""
    if not available():
        return None
    try:
        return VertexBackend()
    except RuntimeError:
        return None
