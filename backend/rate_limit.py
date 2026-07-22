"""
rate_limit.py — In-process sliding-window rate limiter (two tiers, e.g.
per-minute burst + per-day budget) + a bounded concurrency guard for
generation endpoints.

No Redis / external store — fine for a single-instance deployment. If you
ever run multiple worker processes or replicas, this needs to move to a
shared store (Redis INCR + EXPIRE), since each process would otherwise
track its own counters and the real limit becomes (per-process limit ×
number of processes).
"""

import time
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from fastapi import Request


class SlidingWindowRateLimiter:
    """Per-key sliding window: at most `limit` events per `window_seconds`."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Registers the hit if allowed."""
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > self.window:
                dq.popleft()

            if len(dq) >= self.limit:
                retry_after = self.window - (now - dq[0])
                return False, max(retry_after, 0.1)

            dq.append(now)
            return True, 0.0

    def prune(self, max_idle_seconds: float = 3600.0) -> int:
        """Drop keys with no recent activity — call periodically to bound
        memory growth from one-off visitors. Returns keys dropped."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, dq in self._hits.items()
                     if not dq or now - dq[-1] > max_idle_seconds]
            for k in stale:
                del self._hits[k]
            return len(stale)


class ConcurrencyGuard:
    """
    Rejects immediately (instead of queueing indefinitely) once more than
    `max_concurrent` generation requests are already in flight.
    """

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._in_flight = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self.max_concurrent:
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight


def get_client_key(request: Request, trust_forwarded_for: bool = False) -> str:
    """
    Best-effort client identity for rate limiting.

    trust_forwarded_for=True only makes sense behind a reverse proxy you
    control that overwrites X-Forwarded-For itself — otherwise any client
    can set that header to dodge limits. If serving directly, leave False.
    """
    if trust_forwarded_for:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"