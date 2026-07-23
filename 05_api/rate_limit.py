"""
Per-user sliding-window rate limiter (in-memory).

Thread-safe via a threading.Lock.
In production with multiple replicas, replace with Redis ZRANGEBYSCORE.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status
from loguru import logger

from api_config import api_settings
from auth import get_current_user
from models import UserInfo

_lock = threading.Lock()
# { user_sub: [timestamp, timestamp, ...] }
_windows: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(user_sub: str, limit: int) -> None:
    """
    Enforce sliding-window rate limit.
    Raises HTTP 429 if the user has exceeded `limit` requests in the last 60s.
    `limit=0` disables rate limiting.
    """
    if limit <= 0:
        return

    now = time.monotonic()
    cutoff = now - 60.0

    with _lock:
        # Evict timestamps older than 60 seconds
        _windows[user_sub] = [t for t in _windows[user_sub] if t > cutoff]
        count = len(_windows[user_sub])

        if count >= limit:
            logger.warning(f"Rate limit hit for user {user_sub} ({count}/{limit})")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {limit} queries per minute. "
                    "Please wait before sending another message."
                ),
                headers={"Retry-After": "60"},
            )

        _windows[user_sub].append(now)


# ── FastAPI dependency ────────────────────────────────────────────────────────

def rate_limited(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """
    FastAPI dependency — applies per-user rate limiting then returns the user.
    Attach to any route that should be rate-limited.
    """
    _check_rate_limit(user.sub, api_settings.rate_limit_per_minute)
    return user
