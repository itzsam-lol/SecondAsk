"""Append-only, hash-chained decision ledgers.

The Track 03 brief asks for an audit trail. A list of log lines is not an audit
trail: nothing stops it being edited after the fact, and nothing proves the
reported "money recovered" corresponds to the decisions actually taken.

Every ledger here makes two guarantees.

**Append-only.** There is no update or delete path. The only public mutation is
``append``.

**Tamper-evident.** Each entry stores ``prev_hash`` and its own ``entry_hash``,
computed over a canonical JSON encoding. Editing any historical entry changes
its hash, breaks the link to the next entry, and ``verify()`` reports the exact
index where the chain first diverges. The chain head is a single 64-character
commitment to the entire decision history of a run, which is also what makes
reproducibility checkable: two runs of the same seed must produce the same head.

Three implementations behind one interface:

``FileLedger``
    In-memory chain with optional durable append to a JSONL file. Thread-safe.
    The default, and the one the simulation uses, because a single-writer
    deterministic run needs nothing more.

``AsyncLedger``
    For concurrent workers. Adds an asyncio lock for coroutines and an OS-level
    file lock so that several *processes* appending to the same chain cannot
    interleave and fork it. Correct and slower, for the reasons in its docstring.

Canonicalisation matters. ``json.dumps`` with ``sort_keys=True``, fixed
separators and ``ensure_ascii=True`` gives a byte-identical encoding across
platforms and Python versions. Without it a Hinglish message body containing
Devanagari would hash differently on two machines and the reproducibility claim
would quietly be false.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

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


def hash_entry(index: int, at: str, kind: str, item_id: str, payload: Any, prev_hash: str) -> str:
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


_hash_entry = hash_entry  # retained: referenced by name in earlier revisions


# ---------------------------------------------------------------------------
# Cross-process locking
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def file_lock(path: str, timeout: float = 10.0):
    """Advisory OS-level lock on a sidecar file.

    Uses ``msvcrt`` on Windows and ``fcntl`` on POSIX. Both are advisory, which
    is fine: every writer to this chain goes through this function.

    A lock is genuinely required rather than decorative. Two processes appending
    to one chain must serialise around *read tail hash, compute, write*, because
    if both read the same tail they produce two entries claiming the same
    predecessor and the chain forks. A fork is worse than a gap: ``verify`` will
    reject one branch and there is no way to tell afterwards which side was real.

    Degrades to a no-op if neither locking module is importable, rather than
    failing the run. That is stated here because a silently absent lock is
    exactly the kind of thing that looks fine until two workers are deployed.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path + ".lock", "a+b")
    acquired = False
    try:
        try:
            import msvcrt  # Windows

            deadline = timeout
            while deadline > 0:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    import time as _time

                    _time.sleep(0.05)
                    deadline -= 0.05
        except ImportError:
            try:
                import fcntl  # POSIX

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                acquired = True
            except ImportError:
                acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except (ImportError, OSError):
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
        handle.close()


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class BaseLedger(ABC):
    """The contract every ledger honours.

    Deliberately narrow. There is no ``update``, no ``delete`` and no way to set
    a hash from outside, so an implementation cannot offer one by accident.
    """

    run_id: str

    @abstractmethod
    def append(self, at: datetime, kind: str, item_id: str, payload: dict[str, Any]) -> LedgerEntry:
        ...

    @property
    @abstractmethod
    def head(self) -> str:
        ...

    @property
    @abstractmethod
    def entries(self) -> tuple[LedgerEntry, ...]:
        ...

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self.entries)

    def verify(self) -> tuple[bool, Optional[str]]:
        """Recompute the whole chain.

        Returns ``(ok, message)``. On failure the message names the first index
        that does not verify and why, which is what makes the tamper demo
        legible rather than a bare ``False``.
        """
        prev = GENESIS
        for i, entry in enumerate(self.entries):
            if entry.index != i:
                return False, f"entry {i}: index field is {entry.index}, expected {i}"
            if entry.prev_hash != prev:
                return False, f"entry {i}: prev_hash does not match the previous entry's hash"
            expected = hash_entry(
                entry.index, entry.at, entry.kind, entry.item_id, entry.payload, entry.prev_hash
            )
            if expected != entry.entry_hash:
                return False, f"entry {i}: contents were modified after it was written"
            prev = entry.entry_hash
        return True, None

    def filter(self, *, kind: Optional[str] = None, item_id: Optional[str] = None) -> list[LedgerEntry]:
        return [
            e
            for e in self.entries
            if (kind is None or e.kind == kind) and (item_id is None or e.item_id == item_id)
        ]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in self.entries:
            out[entry.kind] = out.get(entry.kind, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "head": self.head,
            "count": len(self),
            "entries": [e.to_dict() for e in self.entries],
        }

    def write_jsonl(self, path: str) -> None:
        """Persist as JSON lines. Encoding is pinned: Windows defaults to cp1252."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for entry in self.entries:
                handle.write(canonical_json(entry.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# File-backed, single process
# ---------------------------------------------------------------------------


@dataclass
class FileLedger(BaseLedger):
    """In-memory chain, optionally appended to a JSONL file as it grows.

    Thread-safe through a plain lock. That is enough for one process: the lock
    serialises the read-modify-write of the tail hash, which is the only
    critical section.

    ``path=None`` keeps everything in memory, which is what the simulation wants
    since a 3,000 item benchmark writes 30,000 entries and only the head matters.
    """

    run_id: str = "run"
    path: Optional[str] = None
    _entries: list[LedgerEntry] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.path:
            directory = os.path.dirname(os.path.abspath(self.path))
            if directory:
                os.makedirs(directory, exist_ok=True)

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Read-only view. A tuple, so a caller cannot append behind our back."""
        return tuple(self._entries)

    def append(self, at: datetime, kind: str, item_id: str, payload: dict[str, Any]) -> LedgerEntry:
        with self._lock:
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
                entry_hash=hash_entry(index, at_str, kind, item_id, payload, prev),
            )
            self._entries.append(entry)
            if self.path:
                with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(canonical_json(entry.to_dict()) + "\n")
            return entry

    @classmethod
    def read_jsonl(cls, path: str, run_id: str = "run") -> "FileLedger":
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


# ---------------------------------------------------------------------------
# Concurrent
# ---------------------------------------------------------------------------


class AsyncLedger(BaseLedger):
    """Ledger for concurrent workers, and across processes.

    Two locks, because there are two distinct races:

    * an ``asyncio.Lock`` so interleaved coroutines in one event loop cannot both
      read the same tail hash,
    * an OS file lock so separate *processes* writing the same chain file
      serialise the same critical section.

    The cost is real and worth stating. When ``path`` is set, every append
    re-reads the tail hash from the file under the lock, because another process
    may have appended since. That is one lock acquisition and one seek per entry,
    which is roughly an order of magnitude slower than ``FileLedger``. It is the
    price of a chain that several writers can share, and it is why the simulation
    does not use it: a single-writer deterministic run gains nothing and would
    pay for all of it.

    Both a sync ``append`` and an ``append_async`` are provided so the same
    ledger can be used from a worker thread and from a coroutine. They share the
    threading lock.
    """

    def __init__(self, run_id: str = "run", path: Optional[str] = None) -> None:
        self.run_id = run_id
        self.path = path
        self._entries: list[LedgerEntry] = []
        self._thread_lock = threading.RLock()
        self._async_lock: Any = None  # created lazily, needs a running loop
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)

    @property
    def head(self) -> str:
        with self._thread_lock:
            return self._entries[-1].entry_hash if self._entries else GENESIS

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._thread_lock:
            return tuple(self._entries)

    def _tail_hash_from_file(self) -> Optional[str]:
        """Last entry hash on disk, or None if the file is empty or absent.

        Reads the final non-empty line rather than the whole file. A malformed
        tail returns None, which makes the caller fall back to the in-memory
        head instead of chaining onto something unparseable.
        """
        if not self.path or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size == 0:
                    return None
                # Walk back in blocks until a newline is found. Reading the whole
                # file would make append O(n) in the length of the chain.
                block = 4096
                data = b""
                position = size
                while position > 0:
                    step = min(block, position)
                    position -= step
                    handle.seek(position)
                    data = handle.read(step) + data
                    if data.count(b"\n") >= 2 or position == 0:
                        break
            lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln.strip()]
            if not lines:
                return None
            return json.loads(lines[-1]).get("entry_hash")
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _append_locked(self, at: datetime, kind: str, item_id: str, payload: dict[str, Any]) -> LedgerEntry:
        index = len(self._entries)
        at_str = iso(at)
        prev = self._tail_hash_from_file() if self.path else None
        if prev is None:
            prev = self._entries[-1].entry_hash if self._entries else GENESIS
        entry = LedgerEntry(
            index=index,
            at=at_str,
            kind=kind,
            item_id=item_id,
            payload=payload,
            prev_hash=prev,
            entry_hash=hash_entry(index, at_str, kind, item_id, payload, prev),
        )
        self._entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(entry.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def append(self, at: datetime, kind: str, item_id: str, payload: dict[str, Any]) -> LedgerEntry:
        with self._thread_lock:
            if self.path:
                with file_lock(self.path):
                    return self._append_locked(at, kind, item_id, payload)
            return self._append_locked(at, kind, item_id, payload)

    async def append_async(
        self, at: datetime, kind: str, item_id: str, payload: dict[str, Any]
    ) -> LedgerEntry:
        import asyncio

        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            # The file lock and fsync are blocking, so they go to a worker
            # thread. Holding the event loop through an fsync would stall every
            # other coroutine, which defeats the point of being async.
            return await asyncio.get_running_loop().run_in_executor(
                None, self.append, at, kind, item_id, payload
            )


# Backwards compatibility. Every existing import of ``Ledger`` keeps working,
# and the CLI, the runtime and the dashboard are unchanged.
Ledger = FileLedger
