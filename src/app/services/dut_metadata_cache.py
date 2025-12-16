import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_TTL = int(os.getenv("DUT_METADATA_CACHE_TTL", "300"))
DEFAULT_RECORD_TTL = int(os.getenv("DUT_RECORD_CACHE_TTL", "120"))

_redis_client: redis.Redis | None = None

try:
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    _redis_client.ping()
except Exception as exc:  # pragma: no cover - best-effort cache setup
    logger.warning("Redis unavailable for DUT metadata cache at %s: %s", REDIS_URL, exc)
    _redis_client = None

_local_cache: dict[str, tuple[float, Any]] = {}
_locks: dict[str, asyncio.Lock] = {}
_global_lock = asyncio.Lock()


def build_metadata_key(base_url: str, category: str, identifier: str | None = None) -> str:
    normalized_base = base_url.replace("://", "_").replace("/", "_")
    if identifier is None:
        return f"dut:meta:{normalized_base}:{category}"
    return f"dut:meta:{normalized_base}:{category}:{identifier}"


def get_cached(key: str) -> Any | None:
    now = time.time()

    entry = _local_cache.get(key)
    if entry:
        expires_at, value = entry
        if expires_at > now:
            return value
        _local_cache.pop(key, None)

    if _redis_client is not None:
        try:
            payload = _redis_client.get(key)
            if payload is None:
                return None
            value = json.loads(payload)
            _local_cache[key] = (now + DEFAULT_TTL, value)
            return value
        except redis.exceptions.RedisError as exc:
            logger.debug("Redis get failed for key %s: %s", key, exc)
    return None


def set_cached(key: str, value: Any, ttl: int) -> None:
    expires_at = time.time() + ttl
    _local_cache[key] = (expires_at, value)
    if _redis_client is not None:
        try:
            _redis_client.setex(key, ttl, json.dumps(value))
        except redis.exceptions.RedisError as exc:
            logger.debug("Redis setex failed for key %s: %s", key, exc)


async def get_or_set(
    key: str,
    loader: Callable[[], Awaitable[Any] | Any],
    ttl: int = DEFAULT_TTL,
) -> Any:
    cached = get_cached(key)
    if cached is not None:
        return cached

    lock = await _get_lock(key)
    async with lock:
        cached = get_cached(key)
        if cached is not None:
            return cached

        result = loader()
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            value = await result
        else:
            value = result
        set_cached(key, value, ttl)
        return value


async def _get_lock(key: str) -> asyncio.Lock:
    # Ensure a dedicated lock per key using a global guard to avoid race conditions
    async with _global_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
        return lock
