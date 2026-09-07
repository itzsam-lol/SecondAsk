"""Claude backend, over stdlib urllib.

Kept dependency free deliberately. Adding the SDK would mean a reviewer has to
``pip install`` before they can run anything, and the Messages API is a single
POST.

Used only when ``ANTHROPIC_API_KEY`` is present. Without it the gateway runs its
deterministic path and every headline number in the README is still reproducible,
which is the point of building the fallback first.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5"
API_VERSION = "2023-06-01"


class AnthropicBackend:
    name = "claude"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"claude:{model}"

    def complete_json(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            # Zero temperature for reproducibility. This is a classification and
            # slot-filling workload, not a creative one, and a run that cannot be
            # replayed cannot be audited.
            "temperature": 0.0,
            "messages": [{"role": "user", "content": user}],
        }
        body = json.dumps(payload).encode("utf-8")

        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(API_URL, data=body, method="POST")
            request.add_header("x-api-key", self.api_key)
            request.add_header("anthropic-version", API_VERSION)
            request.add_header("content-type", "application/json")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    data = json.loads(handle.read().decode("utf-8"))
                return _first_text(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                last = RuntimeError(f"anthropic HTTP {exc.code}: {detail}")
                # 429 and 5xx are transient. A 400 means the request is wrong and
                # sending it again will not fix it.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = RuntimeError(f"anthropic transport error: {exc}")
                if attempt == self.max_retries:
                    raise last from exc
            time.sleep(min(4.0, 0.4 * (2 ** (attempt - 1))))

        raise last or RuntimeError("anthropic call failed")


def _first_text(data: dict[str, Any]) -> str:
    """Extract the text from a Messages response.

    Defensive about shape: a stop reason of ``max_tokens`` yields a truncated
    block, and an empty content list is possible. Returning "" lets the gateway
    treat it as malformed and fall back rather than raising a KeyError three
    frames deeper.
    """
    blocks = data.get("content") or []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    return ""


def build_backend() -> AnthropicBackend | None:
    """Return a backend if credentials exist, otherwise None.

    Never raises on a missing key. Absence of a key is the normal case for
    anybody reproducing the benchmark.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    model = os.environ.get("SECONDASK_MODEL", DEFAULT_MODEL)
    try:
        return AnthropicBackend(model=model)
    except RuntimeError:
        return None
