"""PII redaction at the model boundary.

Under the DPDP Act 2023, personal data may be processed only for the purpose the
data principal consented to. Shipping a customer's phone number to a third party
inference API in order to decide whether their SMS said "stop" is not that
purpose, and it is not necessary either: the classification does not depend on
the digits.

So nothing crosses the boundary un-redacted. This runs on every string sent to
any model, in both directions. It is applied in the gateway rather than at each
call site, because a control that each caller has to remember to apply is a
control that will eventually be forgotten.

Redaction is intentionally aggressive. A false positive costs a placeholder in a
prompt; a false negative leaks a customer's account number to a vendor. The order
of the patterns matters, since the longer numeric identifiers must be consumed
before the shorter ones can match inside them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Order is significant. Card and Aadhaar length runs are matched before the
# 10-digit phone pattern, otherwise a 16-digit card would be partly eaten by the
# phone rule and the remainder would leak.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Card-like: 13 to 19 digits, optionally separated by spaces or hyphens.
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    # UPI VPA. Runs after EMAIL so that name@bank does not get tagged as email;
    # the handle set here excludes dots before the @ to reduce that overlap.
    ("UPI_VPA", re.compile(r"\b[\w.-]{2,}@(?:ok\w+|paytm|ybl|upi|axl|ibl|apl|sbi|hdfcbank|icici)\b", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")),
    ("ACCOUNT", re.compile(r"(?<!\d)\d{9,18}(?!\d)")),
]


@dataclass
class Redaction:
    text: str
    replacements: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def found_pii(self) -> bool:
        return bool(self.replacements)

    def restore(self, text: str) -> str:
        """Put the originals back.

        Only used for content that is about to be rendered into a template slot
        and then re-validated, never for anything that is logged or persisted.
        """
        for token, original in self.replacements.items():
            text = text.replace(token, original)
        return text

    def to_dict(self) -> dict[str, Any]:
        return {"counts": dict(sorted(self.counts.items())), "found_pii": self.found_pii}


def redact(text: str, *, max_len: int = 2000) -> Redaction:
    """Replace anything that looks like personal data with a stable token.

    Also truncates. An unbounded customer reply is both a cost problem and a
    prompt-injection surface: the longer the attacker-controlled span, the more
    room there is to construct something that looks like an instruction.
    """
    if not isinstance(text, str):
        text = str(text)
    truncated = text[:max_len]
    replacements: dict[str, str] = {}
    counts: dict[str, int] = {}
    counter = 0

    for label, pattern in _PATTERNS:

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            # Reuse the same token for a repeated value so the model can still
            # see that two mentions refer to one thing.
            for token, existing in replacements.items():
                if existing == original:
                    return token
            counter += 1
            token = f"[{label}_{counter}]"
            replacements[token] = original
            counts[label] = counts.get(label, 0) + 1
            return token

        truncated = pattern.sub(_replace, truncated)

    return Redaction(text=truncated, replacements=replacements, counts=counts)


def assert_clean(text: str) -> None:
    """Raise if any pattern still matches. Used as a boundary assertion.

    This is the belt to redact()'s braces. If a future refactor sends a string
    down a path that skips redaction, this fires in tests rather than in
    production.
    """
    leaked = [label for label, pattern in _PATTERNS if pattern.search(text)]
    if leaked:
        raise ValueError(f"unredacted PII would cross the model boundary: {leaked}")
