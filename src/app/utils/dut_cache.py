"""
Cached wrapper functions for DUT API endpoints.

Provides automatic Redis caching for frequently-accessed DUT data.
"""

import logging
from typing import Any

from app.dependencies.external_api_client import get_cache_service
from app.external_services.dut_api_client import DUTAPIClient
from app.services.redis_cache_service import get_cache_ttl

logger = logging.getLogger(__name__)

# Initialize cache service
_cache = get_cache_service()


async def cached_get_sites(client: DUTAPIClient) -> list[dict[str, Any]]:
    """Get all sites with caching."""

    async def fetch():
        return await client.get_sites()

    return await _cache.get_or_fetch(
        "dut:sites",
        fetch,
        ttl=get_cache_ttl("dut:sites"),
    )


async def cached_get_site_by_id(client: DUTAPIClient, site_id: int) -> dict[str, Any]:
    """Get site by ID with caching."""

    async def fetch(sid: int):
        return await client.get_site_by_id(sid)

    return await _cache.get_or_fetch(
        "dut:sites",
        fetch,
        site_id,
        ttl=get_cache_ttl("dut:sites"),
    )


async def cached_get_models(client: DUTAPIClient) -> list[dict[str, Any]]:
    """Get all models with caching."""

    async def fetch():
        return await client.get_models()

    return await _cache.get_or_fetch(
        "dut:models",
        fetch,
        ttl=get_cache_ttl("dut:models"),
    )


async def cached_get_model_by_id(client: DUTAPIClient, model_id: int) -> dict[str, Any]:
    """Get model by ID with caching."""

    async def fetch(mid: int):
        return await client.get_model_by_id(mid)

    return await _cache.get_or_fetch(
        "dut:models",
        fetch,
        model_id,
        ttl=get_cache_ttl("dut:models"),
    )


async def cached_get_stations(client: DUTAPIClient) -> list[dict[str, Any]]:
    """Get all stations with caching."""

    async def fetch():
        return await client.get_stations()

    return await _cache.get_or_fetch(
        "dut:stations",
        fetch,
        ttl=get_cache_ttl("dut:stations"),
    )


async def cached_get_station_by_id(client: DUTAPIClient, station_id: int) -> dict[str, Any]:
    """Get station by ID with caching."""

    async def fetch(sid: int):
        return await client.get_station_by_id(sid)

    return await _cache.get_or_fetch(
        "dut:stations",
        fetch,
        station_id,
        ttl=get_cache_ttl("dut:stations"),
    )


async def cached_get_dut_records(
    client: DUTAPIClient,
    dut_isn: str,
    site_id: int | None = None,
    model_id: int | None = None,
) -> dict[str, Any]:
    """Get DUT records with caching."""

    async def fetch(isn: str, site_id: int | None = None, model_id: int | None = None):
        return await client.get_dut_records(isn, site_id, model_id)

    return await _cache.get_or_fetch(
        "dut:records",
        fetch,
        dut_isn,
        site_id=site_id,
        model_id=model_id,
        ttl=get_cache_ttl("dut:records"),
    )


async def cached_get_dut_summary(
    client: DUTAPIClient,
    dut_isn: str,
    site_id: int | None = None,
    model_id: int | None = None,
) -> dict[str, Any]:
    """Get DUT summary with caching."""

    async def fetch(isn: str, site_id: int | None = None, model_id: int | None = None):
        return await client.get_dut_summary(isn, site_id, model_id)

    return await _cache.get_or_fetch(
        "dut:summary",
        fetch,
        dut_isn,
        site_id=site_id,
        model_id=model_id,
        ttl=get_cache_ttl("dut:summary"),
    )


async def cached_get_station_nonvalue_records(
    client: DUTAPIClient,
    station_id: int,
    dut_id: int,
) -> dict[str, Any]:
    """Get station nonvalue records with caching."""

    async def fetch(sid: int, did: int):
        return await client.get_station_nonvalue_records(sid, did)

    return await _cache.get_or_fetch(
        "dut:records",
        fetch,
        station_id,
        dut_id,
        ttl=get_cache_ttl("dut:records"),
    )


async def cached_get_pa_test_items_trend(
    client: DUTAPIClient,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Get PA test items trend with caching."""

    async def fetch(payload_data: dict[str, Any], *_args):
        # Ignore cache key parts args, use payload_data which contains the original payload
        return await client.get_pa_test_items_trend(payload_data)

    # Convert payload to hashable key
    cache_key_parts = (
        payload.get("station_id"),
        payload.get("start_time"),
        payload.get("end_time"),
        tuple(sorted(payload.get("test_items", []))),
        payload.get("model", ""),
    )

    return await _cache.get_or_fetch(
        "dut:pa:trend",
        fetch,
        payload,
        *cache_key_parts,
        ttl=get_cache_ttl("dut:pa:trend"),
    )


def invalidate_dut_cache(dut_isn: str | None = None, pattern: str | None = None) -> int:
    """
    Invalidate DUT cache entries.

    Args:
        dut_isn: Specific DUT ISN to invalidate (invalidates all related caches)
        pattern: Redis key pattern to invalidate (e.g., "dut:records:*")

    Returns:
        Number of cache keys invalidated
    """
    if pattern:
        return _cache.invalidate_pattern(pattern)

    if dut_isn:
        # Invalidate all caches related to this DUT
        total = 0
        for prefix in ["dut:records", "dut:summary", "dut:pa:trend", "dut:pa:adjusted-power"]:
            total += _cache.invalidate_pattern(f"{prefix}:*{dut_isn}*")
        return total

    # No specific target, invalidate all DUT caches
    return _cache.invalidate_pattern("dut:*")


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    return _cache.get_stats()
