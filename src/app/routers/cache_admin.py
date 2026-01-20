"""
Cache Management Admin Router

Provides admin endpoints for monitoring and managing the FastAPI cache system.
"""

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.dependencies.authz import get_current_user
from app.utils.admin_access import is_user_admin
from app.models.user import User
from app.utils.cache_manager import (
    adjust_cache_ttl,
    clear_all_cache,
    clear_cache_by_pattern,
    get_cache_keys_info,
    get_cache_stats,
    invalidate_dut_cache,
    invalidate_station_cache,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/cache", tags=["Cache_Management"])

# Module-level dependency to avoid linting issues
current_user_dependency = Depends(get_current_user)


# ============================================================================
# Response Models
# ============================================================================


class CacheStatsResponse(BaseModel):
    """Cache statistics response schema."""

    status: str
    hit_rate: float | None = None
    hits: int | None = None
    misses: int | None = None
    total_requests: int | None = None
    total_keys: int | None = None
    memory_usage: str | None = None
    memory_usage_bytes: int | None = None
    connection_status: str | None = None
    backend_type: str | None = None
    message: str | None = None


class CacheInvalidationResponse(BaseModel):
    """Cache invalidation response schema."""

    status: str
    keys_deleted: int | None = None
    pattern: str | None = None
    message: str


class DUTCacheInvalidationResponse(BaseModel):
    """DUT-specific cache invalidation response."""

    status: str
    dut_isn: str
    total_keys_deleted: int
    patterns_cleared: list[dict] | None = None
    message: str


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/stats",
    response_model=CacheStatsResponse,
    summary="Get cache statistics",
    description="Retrieve comprehensive cache performance statistics",
)
async def get_cache_statistics(
    current_user: User = current_user_dependency,
):
    """
    Get cache statistics including hit rate, memory usage, and connection status.

    **Requires:** Authentication

    **Returns:**
    - status: Cache system status
    - hit_rate: Cache hit rate percentage
    - hits/misses: Cache hit and miss counts
    - total_keys: Number of cached items
    - memory_usage: Memory used by cache
    - connection_status: Redis connection status
    """
    stats = await get_cache_stats()
    return CacheStatsResponse(**stats)


@router.post(
    "/clear",
    response_model=CacheInvalidationResponse,
    summary="Clear all cache",
    description="Clear all API cache entries (Admin only)",
)
async def clear_all_cache_endpoint(
    current_user: User = current_user_dependency,
):
    """
    Clear all API cache entries.

    **Requires:** Authentication

    **Warning:** This will clear ALL cached data and may temporarily impact performance.
    """
    if not is_user_admin(current_user):
        return CacheInvalidationResponse(
            status="error",
            keys_deleted=0,
            message="Admin privileges required"
        )

    result = await clear_all_cache()
    return CacheInvalidationResponse(**result)


@router.post(
    "/clear/pattern",
    response_model=CacheInvalidationResponse,
    summary="Clear cache by pattern",
    description="Clear cache entries matching a specific Redis key pattern (Admin only)",
)
async def clear_cache_by_pattern_endpoint(
    pattern: str = Query(..., description="Redis key pattern (e.g., 'api-cache:*/records/*')"),
    current_user: User = current_user_dependency,
):
    """
    Clear cache entries matching a specific pattern.

    **Requires:** Admin authentication

    **Pattern Examples:**
    - `api-cache:*/records/*` - All record endpoints
    - `api-cache:*/sites*` - All site-related endpoints
    - `api-cache:*/pa/*` - All PA endpoints

    **Returns:** Number of keys deleted
    """
    if not is_user_admin(current_user):
        return CacheInvalidationResponse(
            status="error",
            keys_deleted=0,
            pattern=pattern,
            message="Admin privileges required"
        )

    result = await clear_cache_by_pattern(pattern)
    return CacheInvalidationResponse(**result)


@router.post(
    "/invalidate/dut/{dut_isn}",
    response_model=DUTCacheInvalidationResponse,
    summary="Invalidate DUT cache",
    description="Clear all cache entries for a specific DUT ISN",
)
async def invalidate_dut_cache_endpoint(
    dut_isn: str,
    current_user: User = current_user_dependency,
):
    """
    Invalidate all cache entries related to a specific DUT.

    **Requires:** Authentication

    **Use Cases:**
    - After DUT data is updated
    - After new test results are added
    - To force fresh data retrieval

    **Returns:** Number of cache entries invalidated
    """
    if not is_user_admin(current_user):
        return DUTCacheInvalidationResponse(
            status="error",
            dut_isn=dut_isn,
            total_keys_deleted=0,
            message="Admin privileges required"
        )

    result = await invalidate_dut_cache(dut_isn)
    return DUTCacheInvalidationResponse(**result)


@router.post(
    "/invalidate/station/{station_id}",
    response_model=CacheInvalidationResponse,
    summary="Invalidate station cache",
    description="Clear all cache entries for a specific station",
)
async def invalidate_station_cache_endpoint(
    station_id: str,
    current_user: User = current_user_dependency,
):
    """
    Invalidate all cache entries related to a specific station.

    **Requires:** Authentication

    **Use Cases:**
    - After station configuration changes
    - After new devices are added to station
    - To force fresh data retrieval

    **Returns:** Number of cache entries invalidated
    """
    result = await invalidate_station_cache(station_id)
    return CacheInvalidationResponse(**result)


@router.get(
    "/keys",
    summary="Get cached keys information",
    description="List cached keys with TTL information (Admin only)",
)
async def get_cached_keys(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of keys to return"),
    current_user: User = current_user_dependency,
):
    """
    Get information about cached keys.

    **Requires:** Admin authentication

    **Returns:**
    - total_keys: Total number of cached keys
    - keys: List of keys with TTL and type information
    """
    if not is_user_admin(current_user):
        return {"status": "error", "message": "Admin privileges required"}

    result = await get_cache_keys_info(limit)
    return result


@router.post(
    "/adjust-ttl",
    response_model=CacheInvalidationResponse,
    summary="Adjust cache TTL",
    description="Adjust TTL for cache entries matching a pattern (Admin only)",
)
async def adjust_cache_ttl_endpoint(
    pattern: str = Query(..., description="Redis key pattern"),
    ttl: int = Query(..., ge=10, le=86400, description="New TTL in seconds (10s - 24h)"),
    current_user: User = current_user_dependency,
):
    """
    Adjust TTL (Time To Live) for cache entries matching a pattern.

    **Requires:** Admin authentication

    **Use Cases:**
    - Increase TTL for stable data (e.g., metadata)
    - Decrease TTL for frequently changing data
    - Fine-tune cache based on hit rate analysis

    **TTL Guidelines:**
    - 30s: Frequently changing data (dashboard stats, latest records)
    - 60s: Normal data (records, history)
    - 120s: Stable data (roles, permissions, devices)
    - 300s: Very stable metadata (sites, models, stations)

    **Returns:** Number of keys updated
    """
    if not is_user_admin(current_user):
        return CacheInvalidationResponse(
            status="error",
            keys_deleted=0,
            pattern=pattern,
            message="Admin privileges required"
        )

    result = await adjust_cache_ttl(pattern, ttl)
    return CacheInvalidationResponse(
        status=result["status"],
        keys_deleted=result.get("keys_updated", 0),
        pattern=pattern,
        message=result.get("message", "")
    )
