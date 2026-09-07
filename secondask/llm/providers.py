"""Provider selection.

The gateway takes any object satisfying the ``Backend`` protocol. This module
picks one, so that every call site says "give me a model" rather than naming a
vendor.

That indirection is the point of the whole design argument. If the language model
really is three narrow jobs behind a validating gateway, then the provider should
be a lookup and nothing else in the system should know or care. Anything that had
to change to add Gemini would have been a place where the model had leaked out of
its box.

Selection order when nothing is specified: whichever key is present, Claude first
because that is what the numbers in the README were produced with. Absent both,
``None``, and the gateway runs its deterministic path. Nothing here raises on a
missing key, because running without one is the normal case and must stay the
easy path for anybody reproducing the benchmark.
"""

from __future__ import annotations

import os
from typing import Any, Optional

PROVIDERS = ("claude", "gemini")


def available() -> dict[str, bool]:
    """Which providers have credentials, without reading their values."""
    return {
        "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gemini": bool(
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
        ),
    }


def build(provider: str = "auto") -> Optional[Any]:
    """Build a backend, or return None if none is usable.

    ``provider`` is "auto", "claude", "gemini", or "none".
    """
    provider = (provider or "auto").strip().lower()
    if provider in ("none", "off", "stub"):
        return None

    if provider == "claude":
        from .anthropic_client import build_backend

        return build_backend()
    if provider == "gemini":
        from .gemini_client import build_backend

        return build_backend()
    if provider != "auto":
        raise ValueError(f"unknown provider {provider!r}, expected one of: auto, claude, gemini, none")

    have = available()
    if have["claude"]:
        from .anthropic_client import build_backend as claude

        backend = claude()
        if backend is not None:
            return backend
    if have["gemini"]:
        from .gemini_client import build_backend as gemini

        return gemini()
    return None


def describe() -> str:
    have = available()
    ready = [name for name, ok in have.items() if ok]
    if not ready:
        return "no provider credentials found; the deterministic parser will be used"
    return "credentials found for: " + ", ".join(ready)
