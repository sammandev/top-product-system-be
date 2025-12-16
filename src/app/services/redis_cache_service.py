"""
Redis caching service for DUT API responses.

Provides automatic caching with TTL, cache invalidation, and fallback to direct API calls.
"""

import hashlib
import logging
from collections.abc import Callable
from typing import Any

import orjson
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisCacheService:
    """
    Redis-based cache manager for DUT API endpoints.

    Features:
    - Automatic cache key generation from function args
    - Configurable TTL per endpoint type
    - Graceful fallback when Redis unavailable
    - JSON serialization with orjson for performance
    """

    def __init__(
        self,
        redis_client: Redis | None,
        default_ttl: int = 300,  # 5 minutes default
        enabled: bool = True,
    ):
        """
        Initialize Redis cache service.

        Args:
            redis_client: Redis client instance (None = caching disabled)
            default_ttl: Default cache TTL in seconds
            enabled: Master switch for caching
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.enabled = enabled and redis_client is not None

        if self.enabled:
            try:
                self.redis.ping()
                logger.info("Redis cache service initialized successfully")
            except (RedisConnectionError, RedisError, AttributeError) as exc:
                logger.warning("Redis unavailable, caching disabled: %s", exc)
                self.enabled = False

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate deterministic cache key from function arguments.

        Args:
            prefix: Namespace prefix (e.g., "dut:records")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string like "dut:records:abc123def456"
        """
        # Serialize args/kwargs to stable JSON
        key_data = {
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items()) if v is not None},
        }

        # Create hash from serialized data
        json_bytes = orjson.dumps(key_data, option=orjson.OPT_SORT_KEYS)
        key_hash = hashlib.sha256(json_bytes).hexdigest()[:16]

        return f"{prefix}:{key_hash}"

    async def get_or_fetch(
        self,
        key_prefix: str,
        fetch_func: Callable,
        *args,
        ttl: int | None = None,
        **kwargs,
    ) -> Any:
        """
        Get cached value or fetch from source and cache it.

        Args:
            key_prefix: Cache key namespace
            fetch_func: Async function to call if cache miss
            *args: Arguments to pass to fetch_func
            ttl: Cache TTL in seconds (None = use default)
            **kwargs: Keyword arguments for fetch_func

        Returns:
            Cached or freshly fetched data
        """
        if not self.enabled:
            # Cache disabled, call function directly
            return await fetch_func(*args, **kwargs)

        cache_key = self._generate_cache_key(key_prefix, *args, **kwargs)
        ttl = ttl or self.default_ttl

        try:
            # Try to get from cache
            cached = self.redis.get(cache_key)
            if cached:
                logger.debug("Cache HIT: %s", cache_key)
                return orjson.loads(cached)
        except (RedisConnectionError, RedisError) as exc:
            logger.warning("Redis GET error for %s: %s", cache_key, exc)

        # Cache miss - fetch fresh data
        logger.debug("Cache MISS: %s", cache_key)
        data = await fetch_func(*args, **kwargs)

        # Store in cache
        try:
            serialized = orjson.dumps(data)
            self.redis.setex(cache_key, ttl, serialized)
            logger.debug("Cached: %s (TTL=%ds)", cache_key, ttl)
        except (RedisConnectionError, RedisError, TypeError) as exc:
            logger.warning("Redis SET error for %s: %s", cache_key, exc)

        return data

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache keys matching pattern.

        Args:
            pattern: Redis key pattern (e.g., "dut:records:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            keys = list(self.redis.scan_iter(match=pattern, count=100))
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info("Invalidated %d keys matching %s", deleted, pattern)
                return deleted
        except (RedisConnectionError, RedisError) as exc:
            logger.warning("Redis invalidation error for %s: %s", pattern, exc)

        return 0

    def invalidate_key(self, key: str) -> bool:
        """
        Invalidate specific cache key.

        Args:
            key: Exact cache key to delete

        Returns:
            True if key was deleted
        """
        if not self.enabled:
            return False

        try:
            deleted = self.redis.delete(key)
            if deleted:
                logger.debug("Invalidated key: %s", key)
            return bool(deleted)
        except (RedisConnectionError, RedisError) as exc:
            logger.warning("Redis DELETE error for %s: %s", key, exc)
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (keys count, memory usage, etc.)
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            info = self.redis.info("stats")
            memory = self.redis.info("memory")

            return {
                "enabled": True,
                "total_keys": self.redis.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "memory_used_mb": memory.get("used_memory", 0) / (1024 * 1024),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0),
                ),
            }
        except (RedisConnectionError, RedisError) as exc:
            logger.warning("Failed to get Redis stats: %s", exc)
            return {"enabled": True, "error": str(exc)}

    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        return round((hits / total * 100) if total > 0 else 0.0, 2)


# Cache TTL configurations for different endpoint types
CACHE_TTL_CONFIG = {
    # High-frequency, slowly changing data
    "dut:sites": 3600,  # 1 hour
    "dut:models": 3600,  # 1 hour
    "dut:stations": 3600,  # 1 hour
    # Medium-frequency data
    "dut:records": 300,  # 5 minutes
    "dut:summary": 300,  # 5 minutes
    "dut:history": 300,  # 5 minutes
    # Dynamic/real-time data
    "dut:test-items": 60,  # 1 minute
    "dut:pa:trend": 180,  # 3 minutes
    "dut:pa:adjusted-power": 180,  # 3 minutes
    # Aggregated/computed data
    "dut:top-products": 600,  # 10 minutes
}


def get_cache_ttl(endpoint_type: str) -> int:
    """Get appropriate cache TTL for endpoint type."""
    return CACHE_TTL_CONFIG.get(endpoint_type, 300)  # Default 5 minutes
