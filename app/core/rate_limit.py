from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException, Request


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int
    remaining: int


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def check(
        self,
        key: str,
        *,
        max_runs: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        now = time.monotonic()
        cutoff = now - window_seconds

        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= max_runs:
                retry_after = max(1, int(window_seconds - (now - hits[0])))
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=retry_after,
                    remaining=0,
                )

            hits.append(now)
            return RateLimitDecision(
                allowed=True,
                retry_after_seconds=0,
                remaining=max(0, max_runs - len(hits)),
            )


orchflow_rate_limiter = InMemoryRateLimiter()


def client_rate_limit_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    if request.client is None:
        return "unknown"
    return request.client.host


def raise_rate_limit_exceeded(retry_after_seconds: int) -> None:
    raise HTTPException(
        status_code=429,
        detail=(
            "Demo rate limit reached. Please wait "
            f"{retry_after_seconds} seconds before running it again."
        ),
        headers={"Retry-After": str(retry_after_seconds)},
    )
