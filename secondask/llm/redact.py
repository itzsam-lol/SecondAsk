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

Covered: email, card PAN, Aadhaar, PAN card, IFSC, UPI VPA, phone, bank account,
and names. Names are handled by exact match against the records you already
hold, plus a narrow pattern for names the customer volunteers. See ``redact``
for why arbitrary name detection is deliberately not attempted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

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

# Phrases that introduce a name, so the words after them can be redacted with
# reasonable confidence. Deliberately narrow: a general "capitalised word"
# heuristic would eat "Monday", "Diwali", "HDFC" and every sentence-initial word,
# shredding the prompt so badly the model can no longer classify it.
_SELF_INTRO = re.compile(
    r"\b(?:my name is|this is|i am|i'm|mera naam|naam hai|name\s*[-:])\s+"
    r"([A-Za-z][A-Za-z']{1,20}(?:\s+[A-Za-z][A-Za-z']{1,20}){0,2})",
    re.I,
)

# Words that follow an introduction phrase but are not names. Without this,
# "I am not paying" redacts "not paying" as a person.
_NOT_A_NAME = frozenset({
    "not", "no", "still", "very", "really", "just", "going", "unable", "sorry",
    "waiting", "trying", "paying", "done", "the", "a", "an", "in", "at", "on",
    "already", "abhi", "nahi", "nai", "bahut", "thoda",
})


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


def redact(
    text: str,
    *,
    max_len: int = 2000,
    known_names: Optional[Iterable[str]] = None,
) -> Redaction:
    """Replace anything that looks like personal data with a stable token.

    Also truncates. An unbounded customer reply is both a cost problem and a
    prompt-injection surface: the longer the attacker-controlled span, the more
    room there is to construct something that looks like an instruction.

    **On names.** ``known_names`` is the reliable path and the one production
    should use. You already hold the customer's name in your own records, so
    redacting that exact string is precise, has no false negatives on the name
    that actually matters, and needs no model.

    Finding *arbitrary* names in free text is the unreliable path. Real
    named-entity recognition would put a second model on the privacy boundary,
    which is the thing this module exists to avoid, and a capitalisation
    heuristic mangles ordinary words. So what is implemented is exact matching
    plus a narrow pattern for names a customer volunteers, and the gap is stated
    here rather than papered over with a regex that looks thorough and is not.
    """
    if not isinstance(text, str):
        text = str(text)
    working = text[:max_len]
    replacements: dict[str, str] = {}
    counts: dict[str, int] = {}
    counter = 0

    def swap(original: str, label: str) -> str:
        """Token for a value, reusing one if the value has been seen before.

        Reuse matters: the model can still tell that two mentions refer to the
        same thing, which it needs in order to classify "I already paid from
        [PHONE_1], not [PHONE_2]".
        """
        nonlocal counter
        for token, existing in replacements.items():
            if existing == original:
                return token
        counter += 1
        token = f"[{label}_{counter}]"
        replacements[token] = original
        counts[label] = counts.get(label, 0) + 1
        return token

    # Structured identifiers first, names second.
    #
    # The opposite order looks safer and is not. Redacting the name out of
    # "rahul@okhdfcbank" first leaves "[NAME_1]@okhdfcbank", which no longer
    # matches the VPA pattern, so the handle is hidden but the bank is published.
    # Consuming the whole identifier first yields "[UPI_VPA_1]" and leaks
    # nothing. Structured patterns are higher confidence than a name list, so
    # they get first claim on any overlapping span.
    for label, pattern in _PATTERNS:
        working = pattern.sub(lambda m, _label=label: swap(m.group(0), _label), working)

    for name in known_names or ():
        if not isinstance(name, str):
            continue
        cleaned = " ".join(name.split())
        if len(cleaned) < 3:
            continue
        # Full name first, then each part, so "Rahul Sharma" produces one token
        # rather than two adjacent ones.
        candidates = [cleaned] + [p for p in cleaned.split() if len(p) >= 3]
        for part in candidates:
            pattern = re.compile(r"\b" + re.escape(part) + r"\b", re.IGNORECASE)
            working = pattern.sub(lambda m: swap(m.group(0), "NAME"), working)

    def replace_intro(match: re.Match[str]) -> str:
        captured = match.group(1)
        first = captured.split()[0].lower() if captured.split() else ""
        if first in _NOT_A_NAME:
            return match.group(0)
        return match.group(0).replace(captured, swap(captured, "NAME"))

    working = _SELF_INTRO.sub(replace_intro, working)

    return Redaction(text=working, replacements=replacements, counts=counts)


def assert_clean(text: str) -> None:
    """Raise if any pattern still matches. Used as a boundary assertion.

    This is the belt to redact()'s braces. If a future refactor sends a string
    down a path that skips redaction, this fires in tests rather than in
    production. Names are not checked here, because absence of a known name in
    an arbitrary string is not something this module can assert.
    """
    leaked = [label for label, pattern in _PATTERNS if pattern.search(text)]
    if leaked:
        raise ValueError(f"unredacted PII would cross the model boundary: {leaked}")
