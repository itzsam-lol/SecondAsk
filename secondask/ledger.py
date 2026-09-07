"""Append-only, hash-chained decision ledger.

The Track 03 brief asks for an audit trail. A list of log lines is not an audit
trail: nothing stops it being edited after the fact, and nothing proves the
reported "money recovered" corresponds to the decisions actually taken.

This ledger makes two guarantees.

**Append-only.** There is no update or delete path. The only public mutation is
``append``.

**Tamper-evident.** Each entry stores ``prev_hash`` and its own ``entry_hash``,
computed over a canonical JSON encoding of the entry. Editing any historical
entry changes its hash, which breaks the link to the following entry, and
``verify()`` reports the exact index where the chain first diverges. The chain
head is therefore a single 64-character commitment to the entire decision
history of a run, which is also what makes reproducibility checkable: two runs
of the same seed must produce the same head.

Canonicalisation matters. ``json.dumps`` with ``sort_keys=True``,
``separators`` fixed and ``ensure_ascii=True`` gives a byte-identical encoding
across platforms and Python versions. Without that, a Hinglish message body
containing Devanagari would hash differently on two machines and the
reproducibility claim would quietly be false.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

from .clock import iso

GENESIS = "0" * 64


def canonical_json(payload: Any) -> str:
    """Deterministic JSON encoding used for hashing.

    ``ensure_ascii=True`` escapes non-ASCII to \\uXXXX so the byte stream is
    identical regardless of the platform's default encoding.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_fallback,
    )


def _fallback(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return iso(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


@dataclass(frozen=True)
class LedgerEntry:
    index: int
    at: str
    kind: str
    item_id: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at": self.at,
            "kind": self.kind,
            "item_id": self.item_id,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


def _hash_entry(index: int, at: str, kind: str, item_id: str, payload: Any, prev_hash: str) -> str:
    body = canonical_json(
        {
            "index": index,
            "at": at,
            "kind": kind,
            "item_id": item_id,
            "payload": payload,
            "prev_hash": prev_hash,
        }
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class Ledger:
    """The decision log for one agent run."""

    run_id: str = "run"
    _entries: list[LedgerEntry] = field(default_factory=list, repr=False)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self._entries)

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Read-only view. Returning a tuple prevents callers appending directly."""
        return tuple(self._entries)

    def append(self, at: datetime, kind: str, item_id: str, payload: dict[str, Any]) -> LedgerEntry:
        index = len(self._entries)
        at_str = iso(at)
        prev = self.head
        entry = LedgerEntry(
            index=index,
            at=at_str,
            kind=kind,
            item_id=item_id,
            payload=payload,
            prev_hash=prev,
            entry_hash=_hash_entry(index, at_str, kind, item_id, payload, prev),
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, str | None]:
        """Recompute the whole chain.

        Returns ``(ok, message)``. On failure the message names the first index
        that does not verify and why, which is what makes the tamper demo in the
        video legible rather than a bare ``False``.
        """
        prev = GENESIS
        for i, entry in enumerate(self._entries):
            if entry.index != i:
                return False, f"entry {i}: index field is {entry.index}, expected {i}"
            if entry.prev_hash != prev:
                return False, f"entry {i}: prev_hash does not match the previous entry's hash"
            expected = _hash_entry(
                entry.index, entry.at, entry.kind, entry.item_id, entry.payload, entry.prev_hash
            )
            if expected != entry.entry_hash:
                return False, f"entry {i}: contents were modified after it was written"
            prev = entry.entry_hash
        return True, None

    def filter(self, *, kind: str | None = None, item_id: str | None = None) -> list[LedgerEntry]:
        return [
            e
            for e in self._entries
            if (kind is None or e.kind == kind) and (item_id is None or e.item_id == item_id)
        ]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in self._entries:
            out[entry.kind] = out.get(entry.kind, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "head": self.head,
            "count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def write_jsonl(self, path: str) -> None:
        """Persist as JSON lines. Encoding is pinned: Windows defaults to cp1252."""
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for entry in self._entries:
                handle.write(canonical_json(entry.to_dict()) + "\n")

    @classmethod
    def read_jsonl(cls, path: str, run_id: str = "run") -> "Ledger":
        ledger = cls(run_id=run_id)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                ledger._entries.append(
                    LedgerEntry(
                        index=raw["index"],
                        at=raw["at"],
                        kind=raw["kind"],
                        item_id=raw["item_id"],
                        payload=raw["payload"],
                        prev_hash=raw["prev_hash"],
                        entry_hash=raw["entry_hash"],
                    )
                )
        return ledger
