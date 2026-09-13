"""Gemini backend.

A second implementation of the ``Backend`` protocol, alongside Claude. Two
reasons it exists, and the first is architectural rather than commercial:

**It makes the "the model is a swappable component" claim concrete.** The whole
design argument is that the language model does three narrow jobs behind a
gateway that validates everything it returns. If that is true, swapping the
provider should be a sixty-line file and change no behaviour anywhere else. This
file is the test of that claim.

**Cost.** The reply parser is a classification workload run thousands of times
per batch. A cheap fast model is the right tool, and having two providers means
the choice is a flag rather than a rewrite.

Built on ``urllib`` like the Claude client, so the zero-dependency line holds.

One Gemini-specific feature is worth using: ``responseMimeType: application/json``
constrains decoding to valid JSON at the sampler. That removes the most common
failure the gateway's repair loop exists to handle. The repair loop still runs,
because "valid JSON" and "JSON matching our schema" are different claims and only
the second one matters, but the first attempt succeeds far more often.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Flash-tier by default. This is a short classification and slot-filling
# workload; a reasoning-tier model costs a great deal more and would not classify
# "band karo ye message bhejna" any better.
DEFAULT_MODEL = "gemini-2.5-flash"

# Tried in order when the configured model returns 404. Model names change
# faster than repositories get updated, and a hard failure on a renamed model
# would make the whole real-LLM path look broken when it is one string out of
# date.
FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")


class GeminiBackend:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        json_mode: bool = True,
    ) -> None:
        self.api_key = api_key or _key_from_env()
        if not self.api_key:
            raise RuntimeError("no Gemini API key: set GEMINI_API_KEY or GOOGLE_API_KEY")
        self.model = model or os.environ.get("SECONDASK_GEMINI_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.name = f"gemini:{self.model}"
        self.calls = 0
        self.input_chars = 0
        self.output_chars = 0
        self._resolved = False

    # -- transport ----------------------------------------------------------

    def complete_json(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                # Zero temperature for reproducibility. This is classification and
                # slot filling, not a creative task, and a run that cannot be
                # replayed cannot be audited.
                "temperature": 0.0,
                "maxOutputTokens": max_tokens,
                "candidateCount": 1,
            },
            # Safety settings are left at their defaults deliberately. Customer
            # collection messages are occasionally angry, and a filter tuned for
            # general chat can refuse to classify "I am not paying, this is
            # robbery". The gateway handles a refusal the same way it handles any
            # malformed response, by falling back to the deterministic parser,
            # so the loop degrades rather than stopping.
        }
        if self.json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        # Thinking off. Measured on a live call: the flash model spent 51
        # thinking tokens to emit 9 output tokens classifying a five-word
        # Hinglish message. This is a closed-set classification against a fixed
        # schema, so there is nothing to reason about, and at thousands of calls
        # per batch that overhead is most of the bill and most of the latency.
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

        body = json.dumps(payload).encode("utf-8")
        self.calls += 1
        self.input_chars += len(system) + len(user)

        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                text = self._post(self.model, body)
                self.output_chars += len(text)
                return text
            except _ModelNotFound as exc:
                # Resolve the model name once, then carry on with whatever works.
                if self._resolved:
                    raise RuntimeError(str(exc)) from exc
                resolved = self._resolve_model(body)
                if resolved is None:
                    raise RuntimeError(
                        f"model {self.model!r} was not found and no fallback responded; "
                        f"set SECONDASK_GEMINI_MODEL to a current model id"
                    ) from exc
                self.model = resolved
                self.name = f"gemini:{resolved}"
                self._resolved = True
                continue
            except _Retryable as exc:
                last = exc
                if attempt == self.max_retries:
                    raise RuntimeError(f"gemini call failed after {attempt} attempts: {exc}") from exc
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"gemini call failed: {exc}") from exc

        raise RuntimeError(f"gemini call failed: {last}")

    def _post(self, model: str, body: bytes) -> str:
        # The key goes in a header rather than the query string. In the URL it
        # would be captured by every proxy log and every exception traceback
        # that prints a failing URL, which is how keys leak.
        url = f"{API_ROOT}/{urllib.parse.quote(model)}:generateContent"
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("x-goog-api-key", self.api_key)
        request.add_header("User-Agent", "secondask/0.1")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                return _first_text(json.loads(handle.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 404:
                raise _ModelNotFound(f"model {model!r} not available: {detail}")
            if exc.code in (429, 500, 502, 503, 504):
                raise _Retryable(f"HTTP {exc.code}: {detail}")
            # 400 and 403 are our problem: a bad request or a bad key. Retrying
            # burns quota and changes nothing.
            raise RuntimeError(f"gemini HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise _Retryable(f"transport error: {exc}")

    def _resolve_model(self, body: bytes) -> str | None:
        for candidate in FALLBACK_MODELS:
            if candidate == self.model:
                continue
            try:
                self._post(candidate, body)
                return candidate
            except (_ModelNotFound, RuntimeError, _Retryable):
                continue
        return None

    def usage(self) -> dict[str, Any]:
        """Rough usage, for a cost sanity check rather than billing.

        Character counts, not tokens: the API returns token counts per call and
        threading them through would mean the gateway carrying provider-specific
        accounting. Roughly four characters per token is close enough to notice
        an order-of-magnitude surprise, which is all this is for.
        """
        return {
            "backend": self.name,
            "calls": self.calls,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "approx_input_tokens": self.input_chars // 4,
            "approx_output_tokens": self.output_chars // 4,
        }


class _Retryable(RuntimeError):
    pass


class _ModelNotFound(RuntimeError):
    pass


def _key_from_env() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _first_text(data: dict[str, Any]) -> str:
    """Extract the text from a generateContent response.

    Defensive about shape. A response can carry zero candidates when the prompt
    is blocked, and a candidate can finish on MAX_TOKENS with no parts. Returning
    "" lets the gateway treat it as malformed and fall back, rather than raising
    a KeyError three frames deeper.
    """
    candidates = data.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    return ""


def build_backend() -> GeminiBackend | None:
    """Return a backend if a key exists, otherwise None. Never raises."""
    if not _key_from_env():
        return None
    try:
        return GeminiBackend()
    except RuntimeError:
        return None
