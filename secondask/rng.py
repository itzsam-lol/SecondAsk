"""Deterministic randomness, including common random numbers (CRN).

Two distinct needs, deliberately kept apart:

``Stream``
    Ordinary seeded generation for building a world (amounts, customers, latent
    blockers). Sequential draws are fine here because world construction happens
    once per seed, before any policy runs.

``crn_uniform``
    Outcome draws during evaluation. These must **not** be sequential, because
    different policies take different numbers of actions in different orders. If
    outcomes came from a shared sequential stream, policy A taking one extra
    action would shift every subsequent draw for policy B, and the measured
    difference between them would be dominated by that shift rather than by the
    quality of their decisions.

    Instead every outcome is a pure function of
    ``(seed, item_id, action_kind, hour_bucket)``. Two policies that take the
    same action on the same item in the same hour observe the *same* draw, so a
    difference in results is attributable to a difference in decisions. This is
    the standard variance-reduction technique for paired simulation comparison,
    and it is why the reported deltas between agents are meaningful at n=1000
    rather than needing tens of thousands of replications.

Note that CRN shares the *draw*, not the *probability*. Two policies can take the
same action at the same time and still get different outcomes if their prior
behaviour left the customer in a different state (more annoyed, already
notified). That is correct: the coin is shared, the bias is not.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")

_UINT64_MAX = (1 << 64) - 1


def _digest_u64(*parts: object) -> int:
    """Stable 64-bit hash of the given parts.

    ``hash()`` is deliberately not used: Python randomises string hashing per
    process (PYTHONHASHSEED), which would make runs irreproducible across
    invocations. Everything here goes through SHA-256 of a canonical encoding.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")  # unit separator, prevents ("ab","c") == ("a","bc")
    return struct.unpack(">Q", hasher.digest()[:8])[0]


def stable_hash(*parts: object) -> int:
    """A stable non-negative integer hash of the given parts.

    Public because the policy engine needs deterministic per-item jitter for
    dispatch staggering, and ``hash()`` cannot be used for that: Python
    randomises string hashing per process, so the same item would be staggered
    differently on every run and reproducibility would silently break.
    """
    return _digest_u64(*parts)


def crn_uniform(*parts: object) -> float:
    """A uniform(0,1) draw determined entirely by ``parts``.

    Returns a value in [0, 1). Calling with identical parts always yields an
    identical result, in this process and any other.
    """
    return _digest_u64(*parts) / (_UINT64_MAX + 1.0)


def crn_choice(seq: Sequence[T], *parts: object) -> T:
    if not seq:
        raise ValueError("crn_choice on an empty sequence")
    return seq[_digest_u64(*parts) % len(seq)]


class Stream:
    """A named, seeded random stream.

    Sub-streams are derived by name rather than by consuming the parent, so
    adding a new random draw to (say) customer generation cannot perturb the
    amounts drawn elsewhere. Without this, any change to world-generation code
    would silently invalidate every previously recorded benchmark number.
    """

    __slots__ = ("_rand", "seed", "name")

    def __init__(self, seed: int, name: str = "root") -> None:
        self.seed = seed
        self.name = name
        self._rand = random.Random(_digest_u64(seed, name))

    def sub(self, name: str) -> "Stream":
        return Stream(self.seed, f"{self.name}/{name}")

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return self._rand.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        """Inclusive on both ends, matching ``random.randint``."""
        return self._rand.randint(low, high)

    def chance(self, probability: float) -> bool:
        return self._rand.random() < probability

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("choice on an empty sequence")
        return self._rand.choice(seq)

    def weighted(self, options: Sequence[tuple[T, float]]) -> T:
        """Pick from ``(value, weight)`` pairs.

        Guards the two degenerate inputs that would otherwise produce a
        confusing IndexError deep inside world generation: an empty option list,
        and a list whose weights sum to zero.
        """
        if not options:
            raise ValueError("weighted() needs at least one option")
        total = sum(max(0.0, w) for _, w in options)
        if total <= 0:
            return options[0][0]
        target = self._rand.random() * total
        cumulative = 0.0
        for value, weight in options:
            cumulative += max(0.0, weight)
            if target < cumulative:
                return value
        return options[-1][0]

    def lognormal_int(self, median: int, sigma: float, *, low: int, high: int) -> int:
        """A lognormal draw clamped to ``[low, high]``.

        Payment amounts are strongly right-skewed, which is why the mean ticket
        size of a failure batch is a poor summary and the median is used here.
        Clamping keeps a fat tail from producing a single crore-rupee item that
        dominates every aggregate metric and makes agent comparison meaningless.
        """
        if median <= 0:
            raise ValueError("median must be positive")
        value = int(math.exp(self._rand.gauss(math.log(median), sigma)))
        return max(low, min(high, value))

    def shuffled(self, items: Iterable[T]) -> list[T]:
        out = list(items)
        self._rand.shuffle(out)
        return out
