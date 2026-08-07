"""
Redis-backed sliding-window rate limiter.

Thread-safe and works across multiple API replicas.
Falls back to in-memory limiter if Redis is unavailable.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional

import redis
from fastapi import Depends, HTTPException, status
from loguru import logger

from api_config import api_settings
from auth import get_current_user
from models import UserInfo

# ── In-memory fallback (used when Redis unavailable) ──────────────────────────
_lock = threading.Lock()
_windows: dict[str, list[float]] = defaultdict(list)
_cleanup_counter = 0


def _check_rate_limit_in_memory(user_sub: str, limit: int) -> None:
    """Fallback in-memory rate limiter with periodic cleanup."""
    global _cleanup_counter
    if limit <= 0:
        return

    now = time.monotonic()
    cutoff = now - 60.0

    with _lock:
        # Periodic cleanup every 1000 requests to prevent memory leak
        _cleanup_counter += 1
        if _cleanup_counter >= 1000:
            _cleanup_counter = 0
            for user in list(_windows.keys()):
                _windows[user] = [t for t in _windows[user] if t > cutoff]
                if not _windows[user]:
                    del _windows[user]

        # Evict timestamps older than 60 seconds for this user
        _windows[user_sub] = [t for t in _windows[user_sub] if t > cutoff]
        count = len(_windows[user_sub])

        if count >= limit:
            logger.warning(f"Rate limit hit for user {user_sub} ({count}/{limit}) [in-memory]")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {limit} queries per minute. "
                    "Please wait before sending another message."
                ),
                headers={"Retry-After": "60"},
            )

        _windows[user_sub].append(now)


# ── Redis-backed rate limiter ─────────────────────────────────────────────────
_redis_client: Optional[redis.Redis] = None
_redis_lock = threading.Lock()
_redis_available = False


def _get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client with connection pooling."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            _redis_client = redis.from_url(
                api_settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
                max_connections=10,
            )
            # Test connection
            _redis_client.ping()
            _redis_available = True
            logger.info("Redis rate limiter connected successfully")
            return _redis_client
        except Exception as e:
            logger.warning(f"Redis unavailable, falling back to in-memory rate limiter: {e}")
            _redis_client = None
            _redis_available = False
            return None


def _check_rate_limit_redis(user_sub: str, limit: int) -> None:
    """Redis-backed sliding window rate limiter using sorted sets."""
    if limit <= 0:
        return

    client = _get_redis_client()
    if client is None:
        _check_rate_limit_in_memory(user_sub, limit)
        return

    now = time.time()
    cutoff = now - 60.0
    key = f"ratelimit:{user_sub}"

    try:
        pipe = client.pipeline()
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, cutoff)
        # Count current requests
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry on key (120 seconds to cover the window)
        pipe.expire(key, 120)
        results = pipe.execute()

        count = results[1]
        if count >= limit:
            logger.warning(f"Rate limit hit for user {user_sub} ({count}/{limit}) [redis]")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {limit} queries per minute. "
                    "Please wait before sending another message."
                ),
                headers={"Retry-After": "60"},
            )
    except redis.RedisError as e:
        logger.error(f"Redis rate limiter error, falling back: {e}")
        _check_rate_limit_in_memory(user_sub, limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected rate limiter error: {e}")
        _check_rate_limit_in_memory(user_sub, limit)


# ── FastAPI dependency ────────────────────────────────────────────────────────

def rate_limited(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """
    FastAPI dependency — applies per-user rate limiting then returns the user.
    Attach to any route that should be rate-limited.
    """
    _check_rate_limit_redis(user.sub, api_settings.rate_limit_per_minute)
    return user