"""Async gateway and channel I/O.

Built on ``asyncio.to_thread`` over the existing synchronous client rather than
on an async HTTP library. Two reasons, and the first is the one that matters:

**It keeps every guarantee the sync client already has.** Idempotency caching,
the circuit breaker, deterministic failure injection and the refusal to run with
a live key all live in ``RazorpayClient``. Reimplementing them against an async
transport would mean two copies of the retry logic, and the copy that drifts is
always the one that matters at 3 AM.

**It needs no dependency.** ``aiohttp`` or ``httpx`` would break the
zero-dependency line for a workload that is entirely I/O-bound. A thread pool
gives real concurrency for blocking sockets; the GIL is released during the
socket wait, which is where all the time goes.

Concurrency is bounded by a semaphore. Unbounded ``gather`` over ten thousand
items would open ten thousand sockets, exhaust the pool, and turn a recovery
batch into a self-inflicted denial of service against the payment gateway. The
default of 16 is deliberately conservative.

**Results are applied in deterministic order.** ``gather_bounded`` preserves
input order regardless of completion order, so a batch of concurrent calls
produces the same sequence of state changes on every run. Without that, adding
concurrency would silently destroy the reproducibility guarantee that the rest
of the project rests on, and the ledger head would differ between runs of the
same seed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence, TypeVar

from .razorpay_client import RazorpayClient, RazorpayError

T = TypeVar("T")

DEFAULT_CONCURRENCY = 16


async def gather_bounded(
    factories: Sequence[Callable[[], Awaitable[T]]],
    *,
    limit: int = DEFAULT_CONCURRENCY,
    return_exceptions: bool = True,
) -> list[Any]:
    """Run awaitables with bounded concurrency, results in input order.

    Takes zero-argument factories rather than coroutines so that nothing is
    scheduled until a semaphore slot is free. Passing live coroutine objects to a
    semaphore-wrapped gather creates every coroutine up front, which is fine for
    a hundred and is not for a hundred thousand.
    """
    if not factories:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(factory: Callable[[], Awaitable[T]]) -> Any:
        async with semaphore:
            return await factory()

    return await asyncio.gather(
        *(run(f) for f in factories), return_exceptions=return_exceptions
    )


class AsyncRazorpayClient:
    """Async facade over the synchronous client.

    Shares the underlying client, so the idempotency cache and the circuit
    breaker are shared too. That is the point: two concurrent workers hitting the
    same idempotency key must collapse to one call, and a breaker that only one
    worker can see protects nothing.
    """

    def __init__(self, client: Optional[RazorpayClient] = None, *, concurrency: int = DEFAULT_CONCURRENCY) -> None:
        self.client = client or RazorpayClient(mode="mock")
        self.concurrency = concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _sem(self) -> asyncio.Semaphore:
        # Created lazily: a Semaphore binds to the running loop, and building one
        # in __init__ ties the client to whichever loop happened to be current.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.concurrency))
        return self._semaphore

    async def create_payment_link(self, **kwargs: Any) -> dict[str, Any]:
        async with self._sem():
            return await asyncio.to_thread(lambda: self.client.create_payment_link(**kwargs))

    async def fetch_payment_link(self, link_id: str, *, now: datetime) -> dict[str, Any]:
        async with self._sem():
            return await asyncio.to_thread(self.client.fetch_payment_link, link_id, now=now)

    async def create_order(self, **kwargs: Any) -> dict[str, Any]:
        async with self._sem():
            return await asyncio.to_thread(lambda: self.client.create_order(**kwargs))

    async def create_payment_links(self, requests: Iterable[dict[str, Any]]) -> list[Any]:
        """Create many links concurrently. Results stay in request order.

        Exceptions are returned rather than raised, so one gateway failure does
        not discard the successful links created alongside it. The caller
        inspects each result, which is the only correct way to handle a partial
        batch when money is involved.
        """
        payloads = list(requests)
        return await gather_bounded(
            [lambda p=p: self.create_payment_link(**p) for p in payloads],
            limit=self.concurrency,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.client.to_dict()


class AsyncChannel:
    """Simulated notification delivery with realistic latency.

    Exists so the async path has something with a genuine await in it other than
    the gateway. In production this is where an SMS or WhatsApp provider client
    goes, and the shape (bounded concurrency, per-send result, no exception
    escaping into the loop) is the shape that provider needs anyway.
    """

    def __init__(self, *, concurrency: int = DEFAULT_CONCURRENCY, latency_seconds: float = 0.0) -> None:
        self.concurrency = concurrency
        self.latency_seconds = latency_seconds
        self.sent = 0

    async def send(self, channel: str, body: str) -> dict[str, Any]:
        if self.latency_seconds:
            await asyncio.sleep(self.latency_seconds)
        self.sent += 1
        return {"channel": channel, "bytes": len(body.encode("utf-8")), "accepted": True}

    async def send_many(self, messages: Sequence[tuple[str, str]]) -> list[Any]:
        return await gather_bounded(
            [lambda m=m: self.send(*m) for m in messages], limit=self.concurrency
        )
