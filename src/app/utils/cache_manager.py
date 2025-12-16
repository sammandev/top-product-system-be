"""
Cache Management Utility

Provides cache monitoring, statistics, and invalidation functionality
for the FastAPI Cache system.
"""

import logging
from typing import Any

from fastapi_cache import FastAPICache

logger = logging.getLogger(__name__)


async def get_cache_stats() -> dict[str, Any]:
    """
    Get comprehensive cache statistics.
    
    Returns:
        Dictionary with cache statistics including:
        - hit_rate: Percentage of cache hits
        - total_keys: Number of cached items
        - memory_usage: Memory used by cache
        - connection_status: Redis connection status
    """
    try:
        backend = FastAPICache.get_backend()
        if not backend:
            return {
                "status": "disabled",
                "message": "Cache backend not initialized"
            }
        
        redis_client = backend.redis
        
        # Get Redis info
        info = await redis_client.info("stats")
        memory_info = await redis_client.info("memory")
        
        # Calculate hit rate
        hits = int(info.get("keyspace_hits", 0))
        misses = int(info.get("keyspace_misses", 0))
        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        # Get key count for api-cache prefix
        keys = await redis_client.keys("api-cache:*")
        total_keys = len(keys)
        
        # Memory usage
        used_memory = memory_info.get("used_memory_human", "N/A")
        used_memory_bytes = int(memory_info.get("used_memory", 0))
        
        return {
            "status": "active",
            "hit_rate": round(hit_rate, 2),
            "hits": hits,
            "misses": misses,
            "total_requests": total_requests,
            "total_keys": total_keys,
            "memory_usage": used_memory,
            "memory_usage_bytes": used_memory_bytes,
            "connection_status": "connected",
            "backend_type": "redis"
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "connection_status": "disconnected"
        }


async def clear_cache_by_pattern(pattern: str) -> dict[str, Any]:
    """
    Clear cache entries matching a specific pattern.
    
    Args:
        pattern: Redis key pattern (e.g., "api-cache:*", "api-cache:/api/dut/records/*")
        
    Returns:
        Dictionary with operation results
    """
    try:
        backend = FastAPICache.get_backend()
        if not backend:
            return {"status": "error", "message": "Cache backend not initialized"}
        
        redis_client = backend.redis
        
        # Find matching keys
        keys = await redis_client.keys(pattern)
        
        if not keys:
            return {
                "status": "success",
                "keys_deleted": 0,
                "pattern": pattern,
                "message": "No keys found matching pattern"
            }
        
        # Delete keys
        deleted_count = await redis_client.delete(*keys)
        
        logger.info(f"Cleared {deleted_count} cache keys matching pattern: {pattern}")
        
        return {
            "status": "success",
            "keys_deleted": deleted_count,
            "pattern": pattern,
            "message": f"Successfully deleted {deleted_count} cache entries"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache by pattern '{pattern}': {e}", exc_info=True)
        return {
            "status": "error",
            "pattern": pattern,
            "message": str(e)
        }


async def clear_cache_by_tag(tag: str) -> dict[str, Any]:
    """
    Clear cache entries for a specific tag/category.
    
    Args:
        tag: Cache tag (e.g., "dut_records", "sites", "models")
        
    Returns:
        Dictionary with operation results
    """
    pattern = f"api-cache:*{tag}*"
    return await clear_cache_by_pattern(pattern)


async def clear_all_cache() -> dict[str, Any]:
    """
    Clear all API cache entries.
    
    Returns:
        Dictionary with operation results
    """
    return await clear_cache_by_pattern("api-cache:*")


async def invalidate_dut_cache(dut_isn: str) -> dict[str, Any]:
    """
    Invalidate all cache entries related to a specific DUT ISN.
    
    Args:
        dut_isn: DUT ISN identifier
        
    Returns:
        Dictionary with operation results
    """
    # Clear all endpoints that might contain this DUT ISN
    patterns = [
        f"api-cache:*dut_isn={dut_isn}*",
        f"api-cache:*{dut_isn}*",
        f"api-cache:*/records/*{dut_isn}*",
        f"api-cache:*/history/*{dut_isn}*",
        f"api-cache:*/pa/*{dut_isn}*",
        f"api-cache:*/top-product*{dut_isn}*",
    ]
    
    total_deleted = 0
    results = []
    
    for pattern in patterns:
        result = await clear_cache_by_pattern(pattern)
        deleted = result.get("keys_deleted", 0)
        total_deleted += deleted
        if deleted > 0:
            results.append({"pattern": pattern, "deleted": deleted})
    
    return {
        "status": "success",
        "dut_isn": dut_isn,
        "total_keys_deleted": total_deleted,
        "patterns_cleared": results,
        "message": f"Invalidated {total_deleted} cache entries for DUT {dut_isn}"
    }


async def invalidate_station_cache(station_id: str) -> dict[str, Any]:
    """
    Invalidate all cache entries related to a specific station.
    
    Args:
        station_id: Station identifier
        
    Returns:
        Dictionary with operation results
    """
    patterns = [
        f"api-cache:*station_id={station_id}*",
        f"api-cache:*/stations/{station_id}/*",
        f"api-cache:*{station_id}*",
    ]
    
    total_deleted = 0
    for pattern in patterns:
        result = await clear_cache_by_pattern(pattern)
        total_deleted += result.get("keys_deleted", 0)
    
    return {
        "status": "success",
        "station_id": station_id,
        "total_keys_deleted": total_deleted,
        "message": f"Invalidated {total_deleted} cache entries for station {station_id}"
    }


async def get_cache_keys_info(limit: int = 100) -> dict[str, Any]:
    """
    Get information about cached keys.
    
    Args:
        limit: Maximum number of keys to return
        
    Returns:
        Dictionary with key information
    """
    try:
        backend = FastAPICache.get_backend()
        if not backend:
            return {"status": "error", "message": "Cache backend not initialized"}
        
        redis_client = backend.redis
        
        # Get all cache keys
        keys = await redis_client.keys("api-cache:*")
        total_keys = len(keys)
        
        # Get detailed info for first 'limit' keys
        key_details = []
        for key in keys[:limit]:
            try:
                ttl = await redis_client.ttl(key)
                key_type = await redis_client.type(key)
                
                key_details.append({
                    "key": key.decode() if isinstance(key, bytes) else key,
                    "ttl": ttl,
                    "type": key_type.decode() if isinstance(key_type, bytes) else key_type
                })
            except Exception as e:
                logger.warning(f"Error getting info for key {key}: {e}")
        
        return {
            "status": "success",
            "total_keys": total_keys,
            "keys_shown": len(key_details),
            "keys": key_details
        }
        
    except Exception as e:
        logger.error(f"Error getting cache keys info: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


async def adjust_cache_ttl(pattern: str, new_ttl: int) -> dict[str, Any]:
    """
    Adjust TTL for cache entries matching a pattern.
    
    Args:
        pattern: Redis key pattern
        new_ttl: New TTL in seconds
        
    Returns:
        Dictionary with operation results
    """
    try:
        backend = FastAPICache.get_backend()
        if not backend:
            return {"status": "error", "message": "Cache backend not initialized"}
        
        redis_client = backend.redis
        
        # Find matching keys
        keys = await redis_client.keys(pattern)
        
        if not keys:
            return {
                "status": "success",
                "keys_updated": 0,
                "pattern": pattern,
                "message": "No keys found matching pattern"
            }
        
        # Update TTL for each key
        updated_count = 0
        for key in keys:
            try:
                await redis_client.expire(key, new_ttl)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Error updating TTL for key {key}: {e}")
        
        return {
            "status": "success",
            "keys_updated": updated_count,
            "pattern": pattern,
            "new_ttl": new_ttl,
            "message": f"Updated TTL for {updated_count} cache entries to {new_ttl}s"
        }
        
    except Exception as e:
        logger.error(f"Error adjusting cache TTL: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
