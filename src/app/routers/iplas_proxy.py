"""
iPLAS External API Proxy Router.

Provides cached proxy endpoints for iPLAS v1 and v2 APIs with:
- Multi-site support (PTB, PSZ, PXD, PVN, PTY)
- Redis caching for performance
- Request deduplication (in-flight coalescing) for concurrent requests
- Server-side filtering to reduce browser memory usage
- Automatic time-range chunking to bypass 7-day query limit
- Aggregation of chunked results for large date ranges
- Connection pooling for improved throughput
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from redis import Redis
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.cached_test_item import CachedTestItem

from app.schemas.iplas_schemas import (
    CompactCsvTestItemRecord,
    CompactCsvTestItemResponse,
    IplasBatchDownloadRequest,
    IplasBatchDownloadResponse,
    IplasCsvTestItemRequest,
    IplasCsvTestItemResponse,
    IplasDeviceListRequest,
    IplasDeviceListResponse,
    IplasDownloadAttachmentRequest,
    IplasDownloadAttachmentResponse,
    IplasDownloadCsvLogRequest,
    IplasDownloadCsvLogResponse,
    IplasIsnProjectInfo,
    IplasIsnSearchRequest,
    IplasIsnSearchResponse,
    IplasRecordTestItemsRequest,
    IplasRecordTestItemsResponse,
    IplasSiteProjectListResponse,
    IplasStation,
    IplasStationListRequest,
    IplasStationListResponse,
    IplasStationsFromIsnBatchItem,
    IplasStationsFromIsnBatchRequest,
    IplasStationsFromIsnBatchResponse,
    IplasStationsFromIsnRequest,
    IplasStationsFromIsnResponse,
    IplasTestItemByIsnRequest,
    IplasTestItemByIsnResponse,
    IplasTestItemByIsnRecord,
    IplasTestItemByIsnTestItem,
    IplasTestItemInfo,
    IplasTestItemNamesRequest,
    IplasTestItemNamesResponse,
    IplasCachedTestItemNamesRequest,
    IplasCachedTestItemNamesResponse,
    IplasVerifyRequest,
    IplasVerifyResponse,
    SiteProject,
)

logger = logging.getLogger(__name__)

# ============================================================================
# JSON Serialization with orjson Fallback
# ============================================================================

# Try to use orjson for faster JSON serialization on the NDJSON streaming path.
# Falls back to standard json module if orjson is unavailable.
try:
    import orjson as _orjson

    _HAS_ORJSON = True

    def json_dumps(obj: Any) -> bytes:
        """Serialize object to JSON bytes using orjson (fast path)."""
        return _orjson.dumps(obj)

    def json_loads(data: bytes | str) -> Any:
        """Deserialize JSON bytes/string using orjson (fast path)."""
        return _orjson.loads(data)

    logger.debug("Using orjson for JSON serialization (optimized path)")

except ImportError:
    import json as _json

    _HAS_ORJSON = False

    def json_dumps(obj: Any) -> bytes:
        """Serialize object to JSON bytes using stdlib json (fallback path)."""
        return _json.dumps(obj, separators=(",", ":"), default=str).encode("utf-8")

    def json_loads(data: bytes | str) -> Any:
        """Deserialize JSON bytes/string using stdlib json (fallback path)."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return _json.loads(data)

    logger.warning("orjson not available, falling back to stdlib json (slower)")


router = APIRouter(prefix="/api/iplas", tags=["iPLAS_Proxy"])

# ============================================================================
# Multi-Site Configuration
# ============================================================================

# Supported sites with their base URLs and tokens
IPLAS_SITES = {
    "PTB": {
        "base_url": os.getenv("IPLAS_API_PTB_BASE_URL", "http://10.176.33.89"),
        "token": os.getenv("IPLAS_API_TOKEN_PTB", ""),
    },
    "PSZ": {
        "base_url": os.getenv("IPLAS_API_PSZ_BASE_URL", "http://172.24.255.25"),
        "token": os.getenv("IPLAS_API_TOKEN_PSZ", ""),
    },
    "PXD": {
        "base_url": os.getenv("IPLAS_API_PXD_BASE_URL", "http://172.18.212.129"),
        "token": os.getenv("IPLAS_API_TOKEN_PXD", ""),
    },
    "PVN": {
        "base_url": os.getenv("IPLAS_API_PVN_BASE_URL", "http://10.177.240.150"),
        "token": os.getenv("IPLAS_API_TOKEN_PVN", ""),
    },
    "PTY": {
        "base_url": os.getenv("IPLAS_API_PTY_BASE_URL", "http://172.18.212.129"),
        "token": os.getenv("IPLAS_API_TOKEN_PTY", ""),
    },
}

# Common configuration
IPLAS_PORT = int(os.getenv("IPLAS_API_PORT", "32678"))
IPLAS_V1_VERSION = os.getenv("IPLAS_V1_API_VERSION", "/api/v1")
IPLAS_V2_VERSION = os.getenv("IPLAS_V2_API_VERSION", "/api/v2")
IPLAS_TIMEOUT = int(os.getenv("IPLAS_API_TIMEOUT", "180"))

# Cache TTLs
IPLAS_CACHE_TTL = int(os.getenv("IPLAS_CACHE_TTL", "180"))  # 3 minutes for test items
IPLAS_CACHE_TTL_SITE_PROJECTS = int(os.getenv("IPLAS_CACHE_TTL_SITE_PROJECTS", "86400"))  # 24 hours
IPLAS_CACHE_TTL_STATIONS = int(os.getenv("IPLAS_CACHE_TTL_STATIONS", "3600"))  # 1 hour
IPLAS_CACHE_TTL_DEVICES = int(os.getenv("IPLAS_CACHE_TTL_DEVICES", "300"))  # 5 minutes
IPLAS_CACHE_TTL_ISN = int(os.getenv("IPLAS_CACHE_TTL_ISN", "300"))  # 5 minutes

# Fallback for legacy env vars
IPLAS_V1_BASE_URL = os.getenv("DUT2_API_BASE_URL", "http://10.176.33.89:32678/api/v1")
IPLAS_TOKEN = os.getenv("DUT2_API_TOKEN", "")


# ============================================================================
# Request Deduplication (In-Flight Request Coalescing)
# ============================================================================
# When multiple users request the same data simultaneously, we only make one
# upstream request and share the result. This dramatically reduces load on
# the iPLAS server and improves response times for concurrent requests.

# Dictionary to track in-flight requests: cache_key -> asyncio.Event
_in_flight_requests: dict[str, asyncio.Event] = {}
# Lock to safely access _in_flight_requests
_in_flight_lock = asyncio.Lock()
# Temporary storage for in-flight results
_in_flight_results: dict[str, tuple[list[dict[str, Any]], bool, int, int, bool]] = {}


async def _get_or_wait_in_flight(cache_key: str) -> tuple[list[dict[str, Any]], bool, int, int, bool] | None:
    """
    Check if there's already an in-flight request for this cache key.
    If so, wait for it to complete and return its result.

    Returns:
        The result tuple if we waited for an in-flight request, None if we should start a new request.
    """
    async with _in_flight_lock:
        if cache_key in _in_flight_requests:
            event = _in_flight_requests[cache_key]
            logger.info(f"Request coalescing: waiting for in-flight request {cache_key[:50]}...")
        else:
            # No in-flight request, we'll be the one to make it
            return None

    # Wait outside the lock for the in-flight request to complete
    try:
        await asyncio.wait_for(event.wait(), timeout=IPLAS_TIMEOUT + 30)
        result = _in_flight_results.get(cache_key)
        if result:
            logger.info(f"Request coalescing: got result from in-flight request {cache_key[:50]}")
            return result
    except asyncio.TimeoutError:
        logger.warning(f"Request coalescing: timed out waiting for {cache_key[:50]}")

    return None


async def _register_in_flight(cache_key: str) -> asyncio.Event:
    """Register that we're starting an in-flight request."""
    async with _in_flight_lock:
        event = asyncio.Event()
        _in_flight_requests[cache_key] = event
        return event


async def _complete_in_flight(cache_key: str, result: tuple[list[dict[str, Any]], bool, int, int, bool]) -> None:
    """Mark an in-flight request as complete and store its result."""
    async with _in_flight_lock:
        _in_flight_results[cache_key] = result
        if cache_key in _in_flight_requests:
            _in_flight_requests[cache_key].set()
            # Clean up after a short delay to allow waiting tasks to get the result
            asyncio.create_task(_cleanup_in_flight(cache_key))


async def _cleanup_in_flight(cache_key: str, delay: float = 2.0) -> None:
    """Clean up in-flight tracking after a delay."""
    await asyncio.sleep(delay)
    async with _in_flight_lock:
        _in_flight_requests.pop(cache_key, None)
        _in_flight_results.pop(cache_key, None)


# ============================================================================
# Connection Pool Management
# ============================================================================
# Reusable httpx client pool for better connection reuse and performance

_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared httpx client with connection pooling."""
    global _http_client

    if _http_client is None or _http_client.is_closed:
        async with _http_client_lock:
            # Double-check after acquiring lock
            if _http_client is None or _http_client.is_closed:
                timeout = httpx.Timeout(IPLAS_TIMEOUT, connect=30.0)
                limits = httpx.Limits(
                    max_connections=50,  # Total connections across all hosts
                    max_keepalive_connections=20,  # Keep-alive connections
                    keepalive_expiry=30.0,  # Keep connections alive for 30s
                )
                _http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
                logger.info("Created shared httpx client with connection pooling")

    return _http_client


async def close_http_client() -> None:
    """Close the shared httpx client (call on shutdown)."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("Closed shared httpx client")


def get_redis_client() -> Redis | None:
    """Get Redis client for caching."""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:7071/0")
        client = Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable for iPLAS cache: {e}")
        return None


def _get_site_config(site: str, user_token: str | None = None) -> dict[str, str]:
    """
    Get iPLAS API configuration for a specific site.

    Args:
        site: Site identifier (PTB, PSZ, PXD, PVN, PTY)
        user_token: Optional user-provided token (overrides default)

    Returns:
        Dict with 'base_url', 'token', 'v1_url', 'v2_url'
    """
    # Normalize site to uppercase
    site_upper = site.upper()

    if site_upper in IPLAS_SITES:
        config = IPLAS_SITES[site_upper]
        base = config["base_url"]
        # Use user token if provided, otherwise use default
        token = user_token if user_token else config["token"]
        return {
            "base_url": base,
            "token": token,
            "v1_url": f"{base}:{IPLAS_PORT}{IPLAS_V1_VERSION}",
            "v2_url": f"{base}:{IPLAS_PORT}{IPLAS_V2_VERSION}",
        }

    # Fallback to PTB if site not found
    logger.warning(f"Unknown site '{site}', falling back to PTB")
    config = IPLAS_SITES["PTB"]
    base = config["base_url"]
    token = user_token if user_token else config["token"]
    return {
        "base_url": base,
        "token": token,
        "v1_url": f"{base}:{IPLAS_PORT}{IPLAS_V1_VERSION}",
        "v2_url": f"{base}:{IPLAS_PORT}{IPLAS_V2_VERSION}",
    }


def _format_datetime_for_iplas(dt: datetime) -> str:
    """Format datetime for iPLAS v1 API (YYYY/MM/DD HH:mm:ss)."""
    return dt.strftime("%Y/%m/%d %H:%M:%S")


def _generate_cache_key(
    site: str,
    project: str,
    station: str,
    device_id: str,
    begin_time: datetime,
    end_time: datetime,
    test_status: str,
) -> str:
    """Generate deterministic cache key for iPLAS test item data."""
    begin_str = begin_time.strftime("%Y%m%d%H%M%S")
    end_str = end_time.strftime("%Y%m%d%H%M%S")
    return f"iplas:csv-testitem:{site}:{project}:{station}:{device_id}:{begin_str}:{end_str}:{test_status}"


# ============================================================================
# Custom Exception for 5000 Record Limit
# ============================================================================


class IplasRecordLimitError(Exception):
    """
    Raised when iPLAS API returns an error indicating the query would exceed
    the 5000 record limit. This allows callers to retry with smaller time
    ranges or per-device fetching.
    """

    pass


async def _fetch_from_iplas(
    site: str,
    project: str,
    station: str,
    device_id: str,
    begin_time: datetime,
    end_time: datetime,
    test_status: str,
    user_token: str | None = None,
    raise_on_limit: bool = False,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch test item data from iPLAS v1 API with multi-site support.

    Args:
        site: Site identifier (PTB, PSZ, PXD, PVN, PTY)
        project: Project name
        station: Station name
        device_id: Device ID (or "ALL")
        begin_time: Start of date range
        end_time: End of date range
        test_status: PASS, FAIL, or ALL
        user_token: Optional user-provided token
        raise_on_limit: If True, raises IplasRecordLimitError on 5000 limit
                       instead of HTTPException (allows caller to handle retry)

    Returns:
        List of test item records

    Raises:
        IplasRecordLimitError: If raise_on_limit=True and API returns 5000 limit error
        HTTPException: For other API errors
    """
    site_config = _get_site_config(site, user_token)
    if not site_config.get("token"):
        raise HTTPException(
            status_code=400,
            detail=f"Missing iPLAS token for site '{site}'. Please configure IPLAS_API_TOKEN_{site.upper()} or provide a user token.",
        )
    url = f"{site_config['v1_url']}/raw/{site}/{project}/get_csv_testitem"

    payload = {
        "station": station,
        "model": "ALL",
        "line": "ALL",
        "deviceid": device_id,
        "test_status": test_status,
        "begintime": _format_datetime_for_iplas(begin_time),
        "endtime": _format_datetime_for_iplas(end_time),
        "token": site_config["token"],
    }

    logger.info(f"Fetching iPLAS v1 data: {site}/{project}/{station} device={device_id}")

    last_exc: Exception | None = None

    async def _do_request(active_client: httpx.AsyncClient) -> httpx.Response:
        return await active_client.post(url, json=payload)

    for attempt in range(1, 4):
        try:
            if client is not None:
                response = await _do_request(client)
            else:
                # UPDATED: Use shared connection pool instead of creating new client
                shared_client = await _get_http_client()
                response = await _do_request(shared_client)
            last_exc = None
            break
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning(
                "iPLAS v1 request failed (attempt %s/3) for %s/%s/%s device=%s: %s",
                attempt,
                site,
                project,
                station,
                device_id,
                exc,
            )
            if attempt < 3:
                await asyncio.sleep(0.2 * attempt)
                continue
            logger.error("iPLAS v1 request failed after retries", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail="iPLAS upstream connection failed. Please retry.",
            )

    if last_exc is not None:
        raise HTTPException(
            status_code=502,
            detail="iPLAS upstream connection failed. Please retry.",
        )

    # Check for 5000 record limit error (HTTP 400 with specific message)
    if response.status_code == 400:
        response_text = response.text
        if "documents count > 5000" in response_text or "5000" in response_text:
            logger.warning(f"iPLAS 5000 record limit hit for {site}/{project}/{station} device={device_id}, range={begin_time} to {end_time}")
            if raise_on_limit:
                raise IplasRecordLimitError(f"Query exceeds 5000 record limit for {station}")
            # Fall through to normal error handling if not raise_on_limit

    if response.status_code != 200:
        logger.error(f"iPLAS API error: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"iPLAS API error: {response.text}",
        )

    data = response.json()

    # UPDATED: Check for error_msg in response body (time interval error, etc.)
    if "error_msg" in data:
        error_msg = data.get("error_msg", "Unknown error")
        logger.error(f"iPLAS API error_msg: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=f"iPLAS API error: {error_msg}",
        )

    if data.get("statuscode") != 200:
        raise HTTPException(
            status_code=500,
            detail=f"iPLAS API returned status {data.get('statuscode')}",
        )

    # Add station field to each record since iPLAS API doesn't include it
    records = data.get("data", [])
    for record in records:
        record["station"] = station

    return records


# ============================================================================
# Chunked Fetching (bypass 7-day limit)
# ============================================================================

# Maximum days per chunk (iPLAS API limit is 7 days, use 6 for safety margin)
MAX_DAYS_PER_CHUNK = int(os.getenv("IPLAS_MAX_DAYS_PER_CHUNK", "6"))
MAX_RECORDS_WARNING = 5000  # iPLAS API limit per request

# Hybrid V1/V2 configuration
# When device count exceeds this threshold, fetch per-device to avoid 5000 limit
DEVICE_DENSITY_THRESHOLD = int(os.getenv("IPLAS_DEVICE_DENSITY_THRESHOLD", "10"))
# Maximum devices per parallel batch
DEVICE_BATCH_SIZE = int(os.getenv("IPLAS_DEVICE_BATCH_SIZE", "5"))
# Maximum parallel requests
MAX_PARALLEL_REQUESTS = int(os.getenv("IPLAS_MAX_PARALLEL_REQUESTS", "3"))


# ============================================================================
# Hybrid V1/V2 Fetching Strategy
# ============================================================================


async def _fetch_device_list_v2(
    site: str,
    project: str,
    station: str,
    start_time: datetime,
    end_time: datetime,
    user_token: str | None = None,
) -> list[str]:
    """
    Fetch device list from iPLAS v2 API for a specific station and time range.

    This is used to determine device density before fetching test data.
    If device count > DEVICE_DENSITY_THRESHOLD, we switch to per-device fetching.

    Args:
        site: Site identifier (PTB, PSZ, PXD, PVN, PTY)
        project: Project name
        station: Station display name
        start_time: Start of date range
        end_time: End of date range
        user_token: Optional user-provided token

    Returns:
        List of device IDs active in the time range
    """
    site_config = _get_site_config(site, user_token)
    if not site_config.get("token"):
        logger.warning("Missing iPLAS token for site '%s' in V2 device list request", site)
        return []
    url = f"{site_config['v2_url']}/{site}/{project}/{station}/test_station_device_list"

    logger.debug(f"Fetching device list from V2: {site}/{project}/{station}")

    timeout = httpx.Timeout(IPLAS_TIMEOUT)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    last_exc: Exception | None = None

    for attempt in range(1, 3):
        try:
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                response = await client.get(
                    url,
                    params={
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                    },
                    headers={"Authorization": f"Bearer {site_config['token']}"},
                )

            if response.status_code != 200:
                logger.warning(f"V2 device list failed ({response.status_code}): {response.text}")
                return []

            devices = response.json()
            logger.debug(f"V2 returned {len(devices)} devices for {station}")
            return devices

        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning(
                "V2 device list request failed (attempt %s/2) for %s/%s/%s: %s",
                attempt,
                site,
                project,
                station,
                exc,
            )
            if attempt < 2:
                await asyncio.sleep(0.2 * attempt)
                continue

    if last_exc is not None:
        logger.warning("V2 device list failed after retries: %s", last_exc)
    return []


async def _fetch_devices_in_parallel(
    site: str,
    project: str,
    station: str,
    device_ids: list[str],
    begin_time: datetime,
    end_time: datetime,
    test_status: str,
    user_token: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Fetch test data for multiple devices in parallel batches.

    Splits device list into batches and fetches them concurrently to improve
    throughput while avoiding rate limiting.

    Args:
        site: Site identifier
        project: Project name
        station: Station name
        device_ids: List of device IDs to fetch
        begin_time: Start of date range
        end_time: End of date range
        test_status: PASS, FAIL, or ALL
        user_token: Optional user-provided token

    Returns:
        Tuple of (all_records, possibly_truncated)
    """
    all_records: list[dict[str, Any]] = []
    possibly_truncated = False

    # Split into batches
    batches = [device_ids[i : i + DEVICE_BATCH_SIZE] for i in range(0, len(device_ids), DEVICE_BATCH_SIZE)]

    logger.info(f"Parallel fetch: {len(device_ids)} devices in {len(batches)} batches (batch size={DEVICE_BATCH_SIZE}, max parallel={MAX_PARALLEL_REQUESTS})")

    # Process batches with limited parallelism
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

    async def fetch_batch(batch: list[str], batch_idx: int) -> tuple[list[dict], bool]:
        async with semaphore:
            batch_records: list[dict] = []
            batch_truncated = False

            timeout = httpx.Timeout(IPLAS_TIMEOUT)
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                for device_id in batch:
                    try:
                        records = await _fetch_from_iplas(
                            site,
                            project,
                            station,
                            device_id,
                            begin_time,
                            end_time,
                            test_status,
                            user_token,
                            client=client,
                        )

                        if len(records) >= MAX_RECORDS_WARNING:
                            batch_truncated = True
                            logger.warning(f"Device {device_id} returned {len(records)} records (may be truncated)")

                        batch_records.extend(records)

                    except HTTPException as e:
                        logger.error(f"Device {device_id} fetch failed: {e.detail}")
                        # Continue with other devices

                    # Small delay between devices in same batch
                    await asyncio.sleep(0.05)

            logger.debug(f"Batch {batch_idx + 1}/{len(batches)}: {len(batch_records)} records from {len(batch)} devices")
            return batch_records, batch_truncated

    # Run all batches
    tasks = [fetch_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Batch failed with exception: {result}")
            continue
        records, truncated = result
        all_records.extend(records)
        if truncated:
            possibly_truncated = True

    logger.info(f"Parallel fetch complete: {len(all_records)} total records")
    return all_records, possibly_truncated


async def _fetch_with_hybrid_strategy(
    site: str,
    project: str,
    station: str,
    device_id: str,
    begin_time: datetime,
    end_time: datetime,
    test_status: str,
    user_token: str | None = None,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """
    Fetch test data using hybrid V1/V2 strategy with automatic 5000-limit fallback.

    Strategy:
    1. If device_id != "ALL", use standard V1 fetch
    2. If device_id == "ALL":
       a. First try standard V1 fetch with raise_on_limit=True
       b. If 5000 limit error is raised:
          - Call V2 API to get device list for the time range
          - If devices found, fetch per-device in parallel
          - If V2 fails, propagate the error
       c. If device count > DEVICE_DENSITY_THRESHOLD (proactive), fetch per-device

    This approach is more efficient because:
    - We don't call V2 API unless necessary (saves latency)
    - We automatically handle cases where V2 device list returns too few devices

    Args:
        site: Site identifier
        project: Project name
        station: Station name
        device_id: Device ID or "ALL"
        begin_time: Start of date range
        end_time: End of date range
        test_status: PASS, FAIL, or ALL
        user_token: Optional user-provided token

    Returns:
        Tuple of (records, possibly_truncated, used_hybrid)
    """
    # If specific device requested, use standard fetch
    if device_id.upper() != "ALL":
        records = await _fetch_from_iplas(site, project, station, device_id, begin_time, end_time, test_status, user_token)
        possibly_truncated = len(records) >= MAX_RECORDS_WARNING
        return records, possibly_truncated, False

    # Try standard fetch first with raise_on_limit=True
    # This is more efficient than always calling V2 API first
    try:
        records = await _fetch_from_iplas(
            site,
            project,
            station,
            "ALL",
            begin_time,
            end_time,
            test_status,
            user_token,
            raise_on_limit=True,
        )
        return records, len(records) >= MAX_RECORDS_WARNING, False
    except IplasRecordLimitError:
        logger.info(f"5000 record limit hit for {station}, switching to V2 device list + parallel fetch")

    # Fetch device list from V2
    devices = await _fetch_device_list_v2(site, project, station, begin_time, end_time, user_token)

    if not devices:
        logger.warning(f"V1 hit limit but V2 returned zero devices for {station}. Returning empty.")
        return [], False, True

    # Fetch per-device in parallel
    logger.info(f"Retrying with per-device fetch: {len(devices)} devices found for {station}")
    records, possibly_truncated = await _fetch_devices_in_parallel(site, project, station, devices, begin_time, end_time, test_status, user_token)
    return records, possibly_truncated, True


async def _fetch_chunked_from_iplas(
    site: str,
    project: str,
    station: str,
    device_id: str,
    begin_time: datetime,
    end_time: datetime,
    test_status: str,
    user_token: str | None = None,
    use_hybrid: bool = True,
) -> tuple[list[dict[str, Any]], bool, int, int, bool]:
    """
    Fetch test item data from iPLAS v1 API with automatic time-range chunking.

    If the date range exceeds MAX_DAYS_PER_CHUNK, splits into multiple
    sequential requests and aggregates results. This bypasses the 7-day
    query limit imposed by the iPLAS API.

    When use_hybrid=True and device_id="ALL", uses hybrid V1/V2 strategy:
    - Checks device count via V2 API
    - If device count > DEVICE_DENSITY_THRESHOLD, fetches per-device in parallel
    - This prevents hitting the 5000 record limit per request

    Args:
        site: Site identifier (PTB, PSZ, PXD, PVN, PTY)
        project: Project name
        station: Station name
        device_id: Device ID (or "ALL")
        begin_time: Start of date range
        end_time: End of date range
        test_status: PASS, FAIL, or ALL
        user_token: Optional user-provided token
        use_hybrid: Use hybrid V1/V2 strategy for high-density stations

    Returns:
        Tuple of (records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid) where:
        - possibly_truncated: True if any chunk returned exactly 5000 records
        - chunks_fetched: Number of chunks successfully fetched
        - total_chunks: Estimated total chunks for the query
        - used_hybrid: True if hybrid V1/V2 strategy was used in any chunk
    """
    total_days = (end_time - begin_time).days
    used_hybrid_any = False

    # Single request if within limit
    if total_days <= MAX_DAYS_PER_CHUNK:
        # Use hybrid strategy if enabled and device_id is ALL
        if use_hybrid and device_id.upper() == "ALL":
            records, possibly_truncated, used_hybrid = await _fetch_with_hybrid_strategy(site, project, station, device_id, begin_time, end_time, test_status, user_token)
            if used_hybrid:
                logger.info(f"Single chunk used hybrid V1/V2 strategy")
            return records, possibly_truncated, 1, 1, used_hybrid
        else:
            records = await _fetch_from_iplas(site, project, station, device_id, begin_time, end_time, test_status, user_token)
            possibly_truncated = len(records) >= MAX_RECORDS_WARNING
            return records, possibly_truncated, 1, 1, False

    # Chunked requests for larger date ranges
    all_records: list[dict[str, Any]] = []
    possibly_truncated = False
    current_start = begin_time
    chunk_count = 0
    estimated_chunks = (total_days // MAX_DAYS_PER_CHUNK) + 1

    logger.info(f"Chunked fetch: {site}/{project}/{station} spanning {total_days} days (~{estimated_chunks} chunks){' with hybrid V1/V2 strategy' if use_hybrid else ''}")

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(days=MAX_DAYS_PER_CHUNK), end_time)
        chunk_count += 1

        try:
            # Use hybrid strategy if enabled and device_id is ALL
            if use_hybrid and device_id.upper() == "ALL":
                chunk_records, chunk_truncated, chunk_used_hybrid = await _fetch_with_hybrid_strategy(site, project, station, device_id, current_start, chunk_end, test_status, user_token)
                if chunk_truncated:
                    possibly_truncated = True
                if chunk_used_hybrid:
                    used_hybrid_any = True
            else:
                chunk_records = await _fetch_from_iplas(site, project, station, device_id, current_start, chunk_end, test_status, user_token)
                # Check if chunk hit the 5000 limit
                if len(chunk_records) >= MAX_RECORDS_WARNING:
                    possibly_truncated = True
                    logger.warning(f"Chunk {chunk_count} returned {len(chunk_records)} records (may be truncated). Consider narrowing date range or filters.")

            all_records.extend(chunk_records)
            logger.debug(f"Chunk {chunk_count}/{estimated_chunks}: {len(chunk_records)} records ({current_start.date()} to {chunk_end.date()})")

        except HTTPException as e:
            # Log but continue with other chunks for partial results
            logger.error(f"Chunk {chunk_count} failed: {e.detail}")
            # Re-raise if this is a critical error (auth, etc.)
            if e.status_code in (401, 403):
                raise

        current_start = chunk_end

        # Small delay between chunks to avoid rate limiting
        if current_start < end_time:
            await asyncio.sleep(0.1)

    logger.info(f"Chunked fetch complete: {len(all_records)} total records from {chunk_count} chunks{' (used hybrid strategy)' if used_hybrid_any else ''}")

    return all_records, possibly_truncated, chunk_count, estimated_chunks, used_hybrid_any


def _filter_test_items(records: list[dict[str, Any]], test_item_filters: list[str] | None) -> list[dict[str, Any]]:
    """
    Filter test items within each record based on the filter list.

    Args:
        records: List of test records from iPLAS
        test_item_filters: List of test item names to keep. If None/empty, keep all.

    Returns:
        Records with filtered TestItem arrays
    """
    if not test_item_filters:
        return records

    filter_set = set(test_item_filters)
    filtered_records = []

    for record in records:
        test_items = record.get("TestItem", [])
        filtered_items = [item for item in test_items if item.get("NAME") in filter_set]

        # Only include record if it has matching test items
        if filtered_items:
            filtered_record = {**record, "TestItem": filtered_items}
            filtered_records.append(filtered_record)

    return filtered_records


def _extract_unique_test_items(records: list[dict[str, Any]]) -> list[IplasTestItemInfo]:
    """
    Extract unique test item names from records with type detection.

    Args:
        records: List of test records from iPLAS

    Returns:
        List of unique test item info with name, is_value, is_bin, has_ucl, has_lcl flags
    """
    test_items_map: dict[str, IplasTestItemInfo] = {}

    for record in records:
        test_items = record.get("TestItem", [])
        for item in test_items:
            name = item.get("NAME")
            if name and name not in test_items_map:
                value = item.get("VALUE", "").upper().strip()

                # Check for UCL and LCL values first
                ucl_str = item.get("UCL", "").strip()
                lcl_str = item.get("LCL", "").strip()
                has_ucl = False
                has_lcl = False

                if ucl_str:
                    try:
                        float(ucl_str)
                        has_ucl = True
                    except (ValueError, TypeError):
                        pass

                if lcl_str:
                    try:
                        float(lcl_str)
                        has_lcl = True
                    except (ValueError, TypeError):
                        pass

                # Determine is_value and is_bin based on VALUE and presence of limits
                # UPDATED: If item has UCL or LCL, treat it as a value item (CRITERIA)
                # even if VALUE is 1/0/-1/-999 (these are often numeric test results with limits)
                is_value = False
                is_bin = False

                if has_ucl or has_lcl:
                    # Has control limits - this is a CRITERIA value item
                    is_value = True
                    is_bin = False
                elif value in ("PASS", "FAIL"):
                    # Pure PASS/FAIL status without limits - binary item
                    is_bin = True
                    is_value = False
                elif value in ("1", "0", "-1", "-999"):
                    # Numeric status values without limits - treated as binary
                    is_bin = True
                    is_value = False
                elif value and value not in ("",):
                    # Try to parse as numeric value (NON-CRITERIA)
                    try:
                        float(value)
                        is_value = True
                    except (ValueError, TypeError):
                        # Non-value: not numeric and not PASS/FAIL
                        is_value = False

                test_items_map[name] = IplasTestItemInfo(name=name, is_value=is_value, is_bin=is_bin, has_ucl=has_ucl, has_lcl=has_lcl)

    # Preserve original appearance order from iPLAS records
    return list(test_items_map.values())


@router.post(
    "/csv-test-items",
    response_model=IplasCsvTestItemResponse,
    summary="Get filtered CSV test items from iPLAS",
    description="""
    Fetches test item data from iPLAS with server-side caching and filtering.
    
    This endpoint:
    1. Checks Redis cache for existing data
    2. Uses request deduplication to coalesce concurrent requests for the same data
    3. Fetches from iPLAS if cache miss
    4. Applies test_item_filters on server-side (reduces payload to frontend)
    5. Supports pagination with limit/offset
    
    **Cache TTL**: Data is cached for 3 minutes
    **Request Deduplication**: Concurrent requests for the same data share a single upstream request.
    """,
)
async def get_csv_test_items(
    request: IplasCsvTestItemRequest,
) -> IplasCsvTestItemResponse:
    """Get filtered CSV test items with caching and request deduplication."""
    redis = get_redis_client()
    cache_key = _generate_cache_key(
        request.site,
        request.project,
        request.station,
        request.device_id,
        request.begin_time,
        request.end_time,
        request.test_status,
    )

    cached = False
    records: list[dict[str, Any]] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                records = json_loads(cached_data)
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    possibly_truncated = False
    chunks_fetched = 1
    total_chunks = 1
    used_hybrid_strategy = False

    # Fetch from iPLAS if cache miss (using chunked fetching for large date ranges)
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")

        # UPDATED: Check for in-flight request first (request deduplication)
        in_flight_result = await _get_or_wait_in_flight(cache_key)
        if in_flight_result is not None:
            # Another request fetched this data, use their result
            records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid_strategy = in_flight_result
            logger.info(f"Used coalesced result for {cache_key[:50]}")
        else:
            # We're the first request - register and fetch
            await _register_in_flight(cache_key)
            try:
                records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid_strategy = await _fetch_chunked_from_iplas(
                    request.site,
                    request.project,
                    request.station,
                    request.device_id,
                    request.begin_time,
                    request.end_time,
                    request.test_status,
                    request.token,  # Pass user token if provided
                )

                # Store in cache
                if redis:
                    try:
                        serialized = json_dumps(records)
                        redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                        logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
                    except Exception as e:
                        logger.warning(f"Redis SET error: {e}")

                # Notify waiting requests
                await _complete_in_flight(cache_key, (records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid_strategy))
            except Exception:
                # Clean up in-flight tracking on error
                await _cleanup_in_flight(cache_key, delay=0)
                raise

    # Apply test item filtering
    filtered = bool(request.test_item_filters)
    if filtered:
        records = _filter_test_items(records, request.test_item_filters)

    total_records = len(records)

    # Apply pagination
    if request.offset:
        records = records[request.offset :]
    if request.limit:
        records = records[: request.limit]

    return IplasCsvTestItemResponse(
        data=records,
        total_records=total_records,
        returned_records=len(records),
        filtered=filtered,
        cached=cached,
        possibly_truncated=possibly_truncated,
        chunks_fetched=chunks_fetched,
        total_chunks=total_chunks,
        used_hybrid_strategy=used_hybrid_strategy,
    )


def _convert_to_compact_record(record: dict[str, Any]) -> CompactCsvTestItemRecord:
    """Convert full record to compact record without TestItem array."""
    test_item_count = record.get("__test_item_count")
    if test_item_count is None:
        test_items = record.get("TestItem", [])
        test_item_count = len(test_items) if isinstance(test_items, list) else 0

    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    return CompactCsvTestItemRecord(
        Site=_safe_str(record.get("Site")),
        Project=_safe_str(record.get("Project")),
        station=_safe_str(record.get("station")),
        TSP=_safe_str(record.get("TSP")),
        Model=_safe_str(record.get("Model")),
        MO=_safe_str(record.get("MO")),
        Line=_safe_str(record.get("Line")),
        ISN=_safe_str(record.get("ISN")),
        DeviceId=_safe_str(record.get("DeviceId")),
        TestStatus=_safe_str(record.get("Test Status")),
        TestStartTime=_safe_str(record.get("Test Start Time")),
        TestEndTime=_safe_str(record.get("Test end Time")),
        ErrorCode=_safe_str(record.get("ErrorCode")),
        ErrorName=_safe_str(record.get("ErrorName")),
        TestItemCount=int(test_item_count) if isinstance(test_item_count, int | str) else 0,
    )


def _convert_to_compact_record_dict(record: dict[str, Any]) -> dict[str, Any]:
    """Convert full record to compact dict with alias keys for caching/sorting."""
    test_item_count = record.get("__test_item_count")
    if test_item_count is None:
        test_items = record.get("TestItem", [])
        test_item_count = len(test_items) if isinstance(test_items, list) else 0

    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    return {
        "Site": _safe_str(record.get("Site")),
        "Project": _safe_str(record.get("Project")),
        "station": _safe_str(record.get("station")),
        "TSP": _safe_str(record.get("TSP")),
        "Model": _safe_str(record.get("Model")),
        "MO": _safe_str(record.get("MO")),
        "Line": _safe_str(record.get("Line")),
        "ISN": _safe_str(record.get("ISN")),
        "DeviceId": _safe_str(record.get("DeviceId")),
        "Test Status": _safe_str(record.get("Test Status")),
        "Test Start Time": _safe_str(record.get("Test Start Time")),
        "Test end Time": _safe_str(record.get("Test end Time")),
        "ErrorCode": _safe_str(record.get("ErrorCode")),
        "ErrorName": _safe_str(record.get("ErrorName")),
        "TestItemCount": int(test_item_count) if isinstance(test_item_count, int | str) else 0,
    }


@router.post(
    "/csv-test-items/compact",
    response_model=CompactCsvTestItemResponse,
    summary="Get compact CSV test items (without TestItem arrays)",
    description="""
    Fetches compact test item data from iPLAS - excludes the TestItem arrays 
    to reduce payload size by 60-80%.
    
    Use this endpoint for:
    - Record list views where you don't need test item details
    - Initial page load to show record summaries
    - Performance-critical scenarios
    
    To get full TestItem data for a specific record, use the full endpoint
    with limit=1 and appropriate filters.
    
    **Memory Savings**: Typical response is 10-50KB instead of 500KB+
    """,
)
async def get_csv_test_items_compact(
    request: IplasCsvTestItemRequest,
) -> CompactCsvTestItemResponse:
    """Get compact CSV test items without TestItem arrays."""
    redis = get_redis_client()
    cache_key = _generate_cache_key(
        request.site,
        request.project,
        request.station,
        request.device_id,
        request.begin_time,
        request.end_time,
        request.test_status,
    )
    compact_cache_key = f"{cache_key}:compact"

    cached = False
    cached_compact = False
    records: list[dict[str, Any]] = []
    compact_records: list[dict[str, Any]] = []

    # Try cache first
    if redis:
        try:
            if not request.test_item_filters:
                cached_data = redis.get(compact_cache_key)
                if cached_data:
                    compact_records = json_loads(cached_data)
                    cached = True
                    cached_compact = True
                    logger.debug(f"Cache HIT (compact): {compact_cache_key}")
            if not cached_compact:
                cached_data = redis.get(cache_key)
                if cached_data:
                    records = json_loads(cached_data)
                    cached = True
                    logger.debug(f"Cache HIT (full for compact): {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    possibly_truncated = False
    chunks_fetched = 1
    total_chunks = 1
    used_hybrid_strategy = False

    # Fetch from iPLAS if cache miss
    if not cached_compact and not records:
        logger.debug(f"Cache MISS (compact): {cache_key}")

        # UPDATED: Check for in-flight request first (request deduplication)
        in_flight_result = await _get_or_wait_in_flight(cache_key)
        if in_flight_result is not None:
            # Another request fetched this data, use their result
            records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid_strategy = in_flight_result
            logger.info(f"Used coalesced result for compact {cache_key[:50]}")
        else:
            # We're the first request - register and fetch
            await _register_in_flight(cache_key)
            try:
                records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid_strategy = await _fetch_chunked_from_iplas(
                    request.site,
                    request.project,
                    request.station,
                    request.device_id,
                    request.begin_time,
                    request.end_time,
                    request.test_status,
                    request.token,
                )

                # Store full records in cache (benefits both endpoints)
                if redis:
                    try:
                        serialized = json_dumps(records)
                        redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                        logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
                    except Exception as e:
                        logger.warning(f"Redis SET error: {e}")

                # Notify waiting requests
                await _complete_in_flight(cache_key, (records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid_strategy))
            except Exception:
                # Clean up in-flight tracking on error
                await _cleanup_in_flight(cache_key, delay=0)
                raise

    # If compact cache hit, we already have compact records ready
    filtered = bool(request.test_item_filters)
    if not compact_records:
        # Apply test item filtering (affects TestItemCount)
        if filtered:
            records = _filter_test_items(records, request.test_item_filters)

        for record in records:
            test_items = record.get("TestItem", [])
            record["__test_item_count"] = len(test_items) if isinstance(test_items, list) else 0
            record.pop("TestItem", None)

        compact_records = [_convert_to_compact_record_dict(r) for r in records]

        # Cache compact records when unfiltered (safe cache key)
        if redis and not filtered:
            try:
                serialized = json_dumps(compact_records)
                redis.setex(compact_cache_key, IPLAS_CACHE_TTL, serialized)
                logger.debug(f"Cached compact: {compact_cache_key} (TTL={IPLAS_CACHE_TTL}s)")
            except Exception as e:
                logger.warning(f"Redis SET error (compact): {e}")

    total_records = len(compact_records)

    # Apply sorting if specified
    if request.sort_by:
        # Map frontend field names to backend record keys
        field_mapping = {
            "TestStartTime": "Test Start Time",
            "TestEndTime": "Test end Time",
            "TestStatus": "Test Status",
            "ISN": "ISN",
            "DeviceId": "DeviceId",
            "ErrorCode": "ErrorCode",
            "ErrorName": "ErrorName",
            "station": "station",
            "Site": "Site",
            "Project": "Project",
        }
        sort_field = field_mapping.get(request.sort_by, request.sort_by)

        try:
            compact_records = sorted(
                compact_records,
                key=lambda r: r.get(sort_field, "") or "",
                reverse=request.sort_desc,
            )
        except Exception as e:
            logger.warning(f"Sorting by {sort_field} failed: {e}")

    # Apply pagination
    if request.offset:
        compact_records = compact_records[request.offset :]
    if request.limit:
        compact_records = compact_records[: request.limit]

    # Convert to compact format (dicts are validated by FastAPI response_model)
    compact_models = compact_records

    return CompactCsvTestItemResponse(
        data=compact_models,
        total_records=total_records,
        returned_records=len(compact_models),
        filtered=filtered,
        cached=cached,
        possibly_truncated=possibly_truncated,
        chunks_fetched=chunks_fetched,
        total_chunks=total_chunks,
        used_hybrid_strategy=used_hybrid_strategy,
    )


@router.post(
    "/record-test-items",
    response_model=IplasRecordTestItemsResponse,
    summary="Get test items for a specific record (lazy loading)",
    description="""
    Fetches the TestItem array for a specific record identified by ISN and test start time.
    
    Use this endpoint to lazy-load test item details when a user expands a record
    in the UI, instead of loading all test items upfront.
    
    This is part of the memory optimization strategy - load compact records first,
    then fetch test items only when needed.
    """,
)
async def get_record_test_items(
    request: IplasRecordTestItemsRequest,
) -> IplasRecordTestItemsResponse:
    """Get test items for a specific record."""
    redis = get_redis_client()

    # Use a date range around the test start time to minimize API calls
    # Parse the test start time and query a 1-hour window
    try:
        # Parse ISO format or common formats
        test_time = datetime.fromisoformat(request.test_start_time.replace("Z", "+00:00"))
    except ValueError:
        # Try parsing other formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
            try:
                test_time = datetime.strptime(request.test_start_time, fmt)
                break
            except ValueError:
                continue
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid test_start_time format: {request.test_start_time}",
            )

    # Query a 2-hour window around the test time
    begin_time = test_time - timedelta(hours=1)
    end_time = test_time + timedelta(hours=1)

    cache_key = _generate_cache_key(
        request.site,
        request.project,
        request.station,
        request.device_id,
        begin_time,
        end_time,
        request.test_status,
    )

    cached = False
    records: list[dict[str, Any]] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                records = json_loads(cached_data)
                cached = True
                logger.debug(f"Cache HIT (record-test-items): {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS (record-test-items): {cache_key}")
        records = await _fetch_from_iplas(
            request.site,
            request.project,
            request.station,
            request.device_id,
            begin_time,
            end_time,
            request.test_status,
            request.token,
        )

        # Store in cache
        if redis:
            try:
                serialized = json_dumps(records)
                redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    # Find the specific record by ISN and test start time
    target_record = None
    for record in records:
        if record.get("ISN") == request.isn and record.get("Test Start Time") == request.test_start_time:
            target_record = record
            break

    if not target_record:
        # Record not found - return empty test items
        return IplasRecordTestItemsResponse(
            isn=request.isn,
            test_start_time=request.test_start_time,
            test_items=[],
            test_item_count=0,
            cached=cached,
        )

    test_items = target_record.get("TestItem", [])

    return IplasRecordTestItemsResponse(
        isn=request.isn,
        test_start_time=request.test_start_time,
        test_items=test_items,
        test_item_count=len(test_items),
        cached=cached,
    )


@router.post(
    "/test-item-names",
    response_model=IplasTestItemNamesResponse,
    summary="Get unique test item names from iPLAS",
    description="""
    Fetches unique test item names from iPLAS for the selection dialog.
    
    This is a lightweight endpoint that returns only test item names (not full data),
    reducing the initial load from ~500KB to ~5KB.
    
    Returns test items with type information (is_value: true for numeric values).
    """,
)
async def get_test_item_names(
    request: IplasTestItemNamesRequest,
) -> IplasTestItemNamesResponse:
    """Get unique test item names for the selection dialog."""
    redis = get_redis_client()
    cache_key = _generate_cache_key(
        request.site,
        request.project,
        request.station,
        request.device_id,
        request.begin_time,
        request.end_time,
        request.test_status,
    )

    records: list[dict[str, Any]] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                records = json_loads(cached_data)
                logger.debug(f"Cache HIT (names): {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss (using chunked fetching for large date ranges)
    if not records:
        logger.debug(f"Cache MISS (names): {cache_key}")
        records, _, _, _, _ = await _fetch_chunked_from_iplas(
            request.site,
            request.project,
            request.station,
            request.device_id,
            request.begin_time,
            request.end_time,
            request.test_status,
            request.token,  # Pass user token if provided
        )

        # Store in cache (benefits both endpoints)
        if redis:
            try:
                serialized = json_dumps(records)
                redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    # Extract unique test item names
    test_items = _extract_unique_test_items(records)

    # UPDATED: Filter out BIN items if requested (for scoring dialogs)
    if request.exclude_bin:
        test_items = [item for item in test_items if not item.is_bin]
        logger.debug(f"Filtered out BIN items, remaining: {len(test_items)} test items")

    return IplasTestItemNamesResponse(
        test_items=test_items,
        total_count=len(test_items),
    )


# Cache TTL for database-backed test item names (7 days in seconds)
CACHED_TEST_ITEMS_TTL_HOURS = 168  # 7 days


@router.post(
    "/test-item-names-cached",
    response_model=IplasCachedTestItemNamesResponse,
    summary="Get cached test item names from database",
    description="""
    Fetches test item names for a station from the database cache.
    
    This is optimized for the "Configure Station" dialog where loading test items
    for long date ranges (30+ days) often times out. The cache key is:
    site + project + station (date range NOT included since test items rarely change).
    
    **Cache TTL**: 7 days in database
    
    On cache miss, fetches from iPLAS using a 7-day window and saves to database.
    Use `force_refresh=true` to manually refresh the cache.
    """,
)
async def get_test_item_names_cached(
    request: IplasCachedTestItemNamesRequest,
    db: Session = Depends(get_db),
) -> IplasCachedTestItemNamesResponse:
    """Get cached test item names from database."""
    from datetime import timezone

    site = request.site.upper()
    project = request.project
    station = request.station

    # Check database cache first (unless force_refresh)
    if not request.force_refresh:
        cached_items = (
            db.query(CachedTestItem)
            .filter(
                CachedTestItem.site == site,
                CachedTestItem.project == project,
                CachedTestItem.station == station,
            )
            .all()
        )

        if cached_items:
            # Check cache age
            oldest_item = min(cached_items, key=lambda x: x.created_at)
            cache_age = datetime.now(timezone.utc) - oldest_item.created_at.replace(tzinfo=timezone.utc)
            cache_age_hours = cache_age.total_seconds() / 3600

            # If cache is still valid (within TTL)
            if cache_age_hours < CACHED_TEST_ITEMS_TTL_HOURS:
                logger.info(f"DB cache HIT: {site}/{project}/{station} ({len(cached_items)} items, {cache_age_hours:.1f}h old)")

                # Convert to response format
                test_items = [
                    IplasTestItemInfo(
                        name=item.test_item_name,
                        is_value=item.is_value,
                        is_bin=item.is_bin,
                        has_ucl=item.has_ucl,
                        has_lcl=item.has_lcl,
                    )
                    for item in cached_items
                ]

                # Filter out BIN items if requested
                if request.exclude_bin:
                    test_items = [item for item in test_items if not item.is_bin]

                return IplasCachedTestItemNamesResponse(
                    test_items=test_items,
                    total_count=len(test_items),
                    cached=True,
                    cache_age_hours=round(cache_age_hours, 1),
                )
            else:
                logger.info(f"DB cache EXPIRED: {site}/{project}/{station} ({cache_age_hours:.1f}h > {CACHED_TEST_ITEMS_TTL_HOURS}h TTL)")

    # Cache miss or force refresh - fetch from iPLAS
    logger.info(f"DB cache MISS: {site}/{project}/{station} - fetching from iPLAS")

    records = []
    last_error = None

    # UPDATED: If user provides time range, use it directly (single attempt)
    if request.begin_time and request.end_time:
        logger.info(f"Using user-provided time range for {site}/{project}/{station}")
        try:
            records, _, _, _, _ = await _fetch_chunked_from_iplas(
                site=site,
                project=project,
                station=station,
                device_id="ALL",
                begin_time=request.begin_time,
                end_time=request.end_time,
                test_status="ALL",
                user_token=request.token,
            )
            if records:
                logger.info(f"Found {len(records)} records with user-provided time range")
        except HTTPException as e:
            last_error = e
            logger.warning(f"Failed with user-provided time range: {e.detail}")
        except Exception as e:
            last_error = HTTPException(
                status_code=502,
                detail=f"Failed to fetch test items from iPLAS: {str(e)}",
            )
            logger.error(f"Failed to fetch from iPLAS: {e}")

    # Fallback: Try progressively larger time windows if no user-provided range
    # or if user-provided range failed
    if not records:
        time_windows_days = [3, 5, 7]
        for days in time_windows_days:
            end_time = datetime.now()
            begin_time = end_time - timedelta(days=days)

            logger.info(f"Trying {days}-day window for {site}/{project}/{station}")

            try:
                records, _, _, _, _ = await _fetch_chunked_from_iplas(
                    site=site,
                    project=project,
                    station=station,
                    device_id="ALL",
                    begin_time=begin_time,
                    end_time=end_time,
                    test_status="ALL",
                    user_token=request.token,
                )

                if records:
                    logger.info(f"Found {len(records)} records with {days}-day window")
                    break
                else:
                    logger.info(f"No records found with {days}-day window, trying larger")

            except HTTPException as e:
                last_error = e
                # If this is a time interval error, try smaller window
                if "time interval" in str(e.detail).lower():
                    logger.warning(f"Time interval error with {days}-day window, trying smaller")
                    continue
                # For other errors, try next window
                logger.warning(f"Error with {days}-day window: {e.detail}")
                continue
            except Exception as e:
                last_error = HTTPException(
                    status_code=502,
                    detail=f"Failed to fetch test items from iPLAS: {str(e)}",
                )
                logger.error(f"Failed to fetch from iPLAS: {e}")
                continue

    if not records and last_error:
        raise last_error

    if not records:
        logger.warning(f"No records found from iPLAS for {site}/{project}/{station}")
        return IplasCachedTestItemNamesResponse(
            test_items=[],
            total_count=0,
            cached=False,
            cache_age_hours=None,
        )

    # Extract unique test item names
    test_items = _extract_unique_test_items(records)

    # Clear old cache entries for this station
    db.query(CachedTestItem).filter(
        CachedTestItem.site == site,
        CachedTestItem.project == project,
        CachedTestItem.station == station,
    ).delete()

    # Save new entries to database
    now = datetime.now(timezone.utc)
    for item in test_items:
        cached_item = CachedTestItem(
            site=site,
            project=project,
            station=station,
            test_item_name=item.name,
            is_value=item.is_value,
            is_bin=item.is_bin,
            has_ucl=item.has_ucl,
            has_lcl=item.has_lcl,
            created_at=now,
            updated_at=now,
        )
        db.add(cached_item)

    db.commit()
    logger.info(f"DB cache STORED: {site}/{project}/{station} ({len(test_items)} items)")

    # Filter out BIN items if requested
    if request.exclude_bin:
        test_items = [item for item in test_items if not item.is_bin]

    return IplasCachedTestItemNamesResponse(
        test_items=test_items,
        total_count=len(test_items),
        cached=False,
        cache_age_hours=None,
    )


@router.get(
    "/health",
    summary="Check iPLAS proxy health",
    description="Verify iPLAS API connectivity and cache status.",
)
async def health_check():
    """Check iPLAS proxy health and cache status."""
    redis = get_redis_client()
    redis_status = "connected" if redis else "unavailable"

    # Check configured sites
    configured_sites = [site for site, config in IPLAS_SITES.items() if config["token"]]

    return {
        "status": "ok",
        "redis_cache": redis_status,
        "configured_sites": configured_sites,
        "cache_ttl": {
            "test_items": IPLAS_CACHE_TTL,
            "site_projects": IPLAS_CACHE_TTL_SITE_PROJECTS,
            "stations": IPLAS_CACHE_TTL_STATIONS,
            "devices": IPLAS_CACHE_TTL_DEVICES,
            "isn_search": IPLAS_CACHE_TTL_ISN,
        },
    }


# ============================================================================
# iPLAS v2 API Proxy Endpoints
# ============================================================================


@router.get(
    "/site-projects",
    response_model=IplasSiteProjectListResponse,
    summary="Get site/project list from iPLAS",
    description="""
    Fetches the list of all available site/project pairs from iPLAS.
    
    **Cache TTL**: 24 hours (data rarely changes)
    
    Note: This endpoint queries all configured sites and aggregates results.
    """,
)
async def get_site_projects(
    data_type: Literal["simple", "strict"] = Query(default="simple", description="Data type filter"),
) -> IplasSiteProjectListResponse:
    """Get all site/project pairs with caching."""
    redis = get_redis_client()
    cache_key = f"iplas:v2:site-projects:{data_type}"

    cached = False
    all_projects: list[SiteProject] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                data = json_loads(cached_data)
                all_projects = [SiteProject(**item) for item in data]
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")

        # Query all configured sites
        for site_name, site_config in IPLAS_SITES.items():
            if not site_config["token"]:
                continue

            try:
                v2_url = f"{site_config['base_url']}:{IPLAS_PORT}{IPLAS_V2_VERSION}"
                url = f"{v2_url}/site_project_list"

                async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                    response = await client.get(
                        url,
                        params={"data_type": data_type},
                        headers={"Authorization": f"Bearer {site_config['token']}"},
                    )

                    if response.status_code == 200:
                        data = response.json()
                        for item in data:
                            all_projects.append(SiteProject(**item))
                    else:
                        logger.warning(f"iPLAS v2 error for {site_name}: {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to fetch site/projects from {site_name}: {e}")

        # Store in cache
        if redis and all_projects:
            try:
                serialized = json_dumps([p.model_dump() for p in all_projects])
                redis.setex(cache_key, IPLAS_CACHE_TTL_SITE_PROJECTS, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_SITE_PROJECTS}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    return IplasSiteProjectListResponse(
        data=all_projects,
        total_count=len(all_projects),
        cached=cached,
    )


@router.post(
    "/stations",
    response_model=IplasStationListResponse,
    summary="Get station list for a project from iPLAS",
    description="""
    Fetches the list of stations for a specific site/project from iPLAS.
    
    **Cache TTL**: 1 hour
    """,
)
async def get_stations(
    request: IplasStationListRequest,
) -> IplasStationListResponse:
    """Get station list for a project with caching."""
    redis = get_redis_client()
    cache_key = f"iplas:v2:stations:{request.site}:{request.project}"

    cached = False
    stations: list[IplasStation] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                data = json_loads(cached_data)
                stations = [IplasStation(**item) for item in data]
                stations.sort(key=lambda s: (s.order is None, s.order))
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")
        site_config = _get_site_config(request.site, request.token)
        url = f"{site_config['v2_url']}/{request.site}/{request.project}/station_list"

        try:
            async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {site_config['token']}"},
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"iPLAS v2 API error: {response.text}",
                    )

                data = response.json()
                stations = [IplasStation(**item) for item in data]
                stations.sort(key=lambda s: (s.order is None, s.order))

        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"iPLAS v2 API unavailable: {e}") from None

        # Store in cache
        if redis and stations:
            try:
                serialized = json_dumps([s.model_dump() for s in stations])
                redis.setex(cache_key, IPLAS_CACHE_TTL_STATIONS, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_STATIONS}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    return IplasStationListResponse(
        data=stations,
        total_count=len(stations),
        cached=cached,
    )


@router.post(
    "/devices",
    response_model=IplasDeviceListResponse,
    summary="Get device list for a station from iPLAS",
    description="""
    Fetches the list of device IDs for a specific station within a time range.
    
    **Cache TTL**: 5 minutes
    """,
)
async def get_devices(
    request: IplasDeviceListRequest,
) -> IplasDeviceListResponse:
    """Get device list for a station with caching."""
    redis = get_redis_client()
    start_str = request.start_time.strftime("%Y%m%d%H%M%S")
    end_str = request.end_time.strftime("%Y%m%d%H%M%S")
    cache_key = f"iplas:v2:devices:{request.site}:{request.project}:{request.station}:{start_str}:{end_str}"

    cached = False
    devices: list[str] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                devices = json_loads(cached_data)
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")
        site_config = _get_site_config(request.site, request.token)
        url = f"{site_config['v2_url']}/{request.site}/{request.project}/{request.station}/test_station_device_list"

        try:
            async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                response = await client.get(
                    url,
                    params={
                        "start_time": request.start_time.isoformat(),
                        "end_time": request.end_time.isoformat(),
                    },
                    headers={"Authorization": f"Bearer {site_config['token']}"},
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"iPLAS v2 API error: {response.text}",
                    )

                devices = response.json()

        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"iPLAS v2 API unavailable: {e}") from None

        # Store in cache
        if redis and devices:
            try:
                serialized = json_dumps(devices)
                redis.setex(cache_key, IPLAS_CACHE_TTL_DEVICES, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_DEVICES}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    return IplasDeviceListResponse(
        data=devices,
        total_count=len(devices),
        cached=cached,
    )


@router.post(
    "/isn-search",
    response_model=IplasIsnSearchResponse,
    summary="Search DUT by ISN from iPLAS",
    description="""
    Searches for DUT test data by ISN across all stations.
    
    **Cache TTL**: 5 minutes
    
    Note: This queries all configured sites to find the ISN.
    """,
)
async def search_by_isn(
    request: IplasIsnSearchRequest,
) -> IplasIsnSearchResponse:
    """Search for DUT data by ISN with caching."""
    redis = get_redis_client()
    cache_key = f"iplas:v2:isn-search:{request.isn}"

    cached = False
    results: list[dict] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                results = json_loads(cached_data)
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")

        # If user provided a token, use it with PTB as default site
        if request.token:
            site_config = _get_site_config("PTB", request.token)
            try:
                url = f"{site_config['v2_url']}/isn_search"
                async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                    response = await client.get(
                        url,
                        params={"isn": request.isn},
                        headers={"Authorization": f"Bearer {site_config['token']}"},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status_code") == 200 and data.get("data"):
                            results = data["data"]
            except Exception as e:
                logger.warning(f"Failed to search ISN with user token: {e}")
        else:
            # Query all configured sites until we find results
            for site_name, site_config in IPLAS_SITES.items():
                if not site_config["token"]:
                    continue

                if results:
                    break  # Found results, stop querying

                try:
                    v2_url = f"{site_config['base_url']}:{IPLAS_PORT}{IPLAS_V2_VERSION}"
                    url = f"{v2_url}/isn_search"

                    async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                        response = await client.get(
                            url,
                            params={"isn": request.isn},
                            headers={"Authorization": f"Bearer {site_config['token']}"},
                        )

                        if response.status_code == 200:
                            data = response.json()
                            if data.get("status_code") == 200 and data.get("data"):
                                results = data["data"]
                                logger.info(f"Found ISN {request.isn} in {site_name}")
                        else:
                            logger.debug(f"ISN not found in {site_name}: {response.status_code}")

                except Exception as e:
                    logger.warning(f"Failed to search ISN in {site_name}: {e}")

        # Store in cache
        if redis and results:
            try:
                serialized = json_dumps(results)
                redis.setex(cache_key, IPLAS_CACHE_TTL_ISN, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_ISN}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    return IplasIsnSearchResponse(
        data=results,
        total_count=len(results),
        cached=cached,
    )


# ============================================================================
# iPLAS v2 Stations From ISN Endpoints
# ============================================================================


async def _search_isn_for_project(
    isn: str,
    token: str | None = None,
) -> tuple[str | None, str | None, bool]:
    """
    Search for an ISN and return its site/project.

    Returns:
        Tuple of (site, project, cached) or (None, None, False) if not found.
    """
    redis = get_redis_client()
    cache_key = f"iplas:v2:isn-project:{isn}"

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                data = json_loads(cached_data)
                logger.debug(f"Cache HIT: {cache_key}")
                return data.get("site"), data.get("project"), True
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    logger.debug(f"Cache MISS: {cache_key}")

    site: str | None = None
    project: str | None = None

    # If user provided a token, use it with PTB as default site
    if token:
        site_config = _get_site_config("PTB", token)
        try:
            url = f"{site_config['v2_url']}/isn_search"
            async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                response = await client.get(
                    url,
                    params={"isn": isn},
                    headers={"Authorization": f"Bearer {site_config['token']}"},
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status_code") == 200 and data.get("data"):
                        first_record = data["data"][0]
                        site = first_record.get("site")
                        project = first_record.get("project")
        except Exception as e:
            logger.warning(f"Failed to search ISN with user token: {e}")
    else:
        # Query all configured sites until we find results
        for site_name, site_config in IPLAS_SITES.items():
            if not site_config["token"]:
                continue

            if site and project:
                break  # Found results, stop querying

            try:
                v2_url = f"{site_config['base_url']}:{IPLAS_PORT}{IPLAS_V2_VERSION}"
                url = f"{v2_url}/isn_search"

                async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                    response = await client.get(
                        url,
                        params={"isn": isn},
                        headers={"Authorization": f"Bearer {site_config['token']}"},
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status_code") == 200 and data.get("data"):
                            first_record = data["data"][0]
                            site = first_record.get("site")
                            project = first_record.get("project")
                            logger.info(f"Found ISN {isn} in {site_name}: {site}/{project}")
                    else:
                        logger.debug(f"ISN not found in {site_name}: {response.status_code}")

            except Exception as e:
                logger.warning(f"Failed to search ISN in {site_name}: {e}")

    # Cache the result
    if redis and site and project:
        try:
            serialized = json_dumps({"site": site, "project": project})
            redis.setex(cache_key, IPLAS_CACHE_TTL_ISN, serialized)
            logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_ISN}s)")
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")

    return site, project, False


async def _get_stations_for_project(
    site: str,
    project: str,
    token: str | None = None,
) -> tuple[list[IplasStation], bool]:
    """
    Get station list for a site/project.

    Returns:
        Tuple of (stations, cached)
    """
    redis = get_redis_client()
    cache_key = f"iplas:v2:stations:{site}:{project}"

    stations: list[IplasStation] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                data = json_loads(cached_data)
                stations = [IplasStation(**item) for item in data]
                stations.sort(key=lambda s: (s.order is None, s.order))
                logger.debug(f"Cache HIT: {cache_key}")
                return stations, True
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    logger.debug(f"Cache MISS: {cache_key}")
    site_config = _get_site_config(site, token)
    url = f"{site_config['v2_url']}/{site}/{project}/station_list"

    try:
        async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {site_config['token']}"},
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get stations for {site}/{project}: {response.status_code}")
                return [], False

            data = response.json()
            stations = [IplasStation(**item) for item in data]
            stations.sort(key=lambda s: (s.order is None, s.order))

    except httpx.RequestError as e:
        logger.warning(f"iPLAS v2 API unavailable for station list: {e}")
        return [], False

    # Store in cache
    if redis and stations:
        try:
            serialized = json_dumps([s.model_dump() for s in stations])
            redis.setex(cache_key, IPLAS_CACHE_TTL_STATIONS, serialized)
            logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_STATIONS}s)")
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")

    return stations, False


@router.post(
    "/isn/stations",
    response_model=IplasStationsFromIsnResponse,
    summary="Get station list from ISN",
    description="""
    Looks up an ISN to find its site/project, then returns all stations for that project.
    
    This is useful when you only have an ISN and want to know all available test stations.
    
    **Flow**:
    1. Search for the ISN across all configured sites
    2. Extract the site/project from the first matching record
    3. Fetch the complete station list for that project
    
    **Cache TTL**: 
    - ISN lookup: 5 minutes
    - Station list: 1 hour
    """,
)
async def get_stations_from_isn(
    request: IplasStationsFromIsnRequest,
) -> IplasStationsFromIsnResponse:
    """Get station list by looking up ISN's project."""

    # Step 1: Search for ISN to get site/project
    site, project, isn_cached = await _search_isn_for_project(request.isn, request.token)

    if not site or not project:
        # ISN not found - return empty result
        return IplasStationsFromIsnResponse(
            isn_info=IplasIsnProjectInfo(
                isn=request.isn,
                site="",
                project="",
                found=False,
            ),
            stations=[],
            total_stations=0,
            cached=False,
        )

    # Step 2: Get station list for the project
    stations, stations_cached = await _get_stations_for_project(site, project, request.token)

    return IplasStationsFromIsnResponse(
        isn_info=IplasIsnProjectInfo(
            isn=request.isn,
            site=site,
            project=project,
            found=True,
        ),
        stations=stations,
        total_stations=len(stations),
        cached=isn_cached and stations_cached,
    )


@router.post(
    "/isn-batch/stations",
    response_model=IplasStationsFromIsnBatchResponse,
    summary="Get station lists from multiple ISNs",
    description="""
    Looks up multiple ISNs to find their site/project pairs, then returns station lists.
    
    Results are deduplicated - if multiple ISNs belong to the same project, 
    the station list is only fetched once.
    
    **Flow**:
    1. Search for each ISN across all configured sites
    2. Group ISNs by unique site/project pairs
    3. Fetch station lists for each unique project
    
    **Limits**:
    - Maximum 50 ISNs per request
    
    **Cache TTL**: 
    - ISN lookup: 5 minutes
    - Station list: 1 hour
    """,
)
async def get_stations_from_isn_batch(
    request: IplasStationsFromIsnBatchRequest,
) -> IplasStationsFromIsnBatchResponse:
    """Get station lists for multiple ISNs."""

    results: list[IplasStationsFromIsnBatchItem] = []
    not_found_isns: list[str] = []
    project_cache: dict[str, list[IplasStation]] = {}  # site:project -> stations
    any_cached = False

    # Step 1: Look up each ISN and group by project
    isn_projects: dict[str, tuple[str, str]] = {}  # isn -> (site, project)

    for isn in request.isns:
        site, project, cached = await _search_isn_for_project(isn, request.token)
        if cached:
            any_cached = True

        if site and project:
            isn_projects[isn] = (site, project)
        else:
            not_found_isns.append(isn)

    # Step 2: Get unique projects and fetch their station lists
    unique_projects: set[tuple[str, str]] = set(isn_projects.values())

    for site, project in unique_projects:
        cache_key = f"{site}:{project}"
        if cache_key not in project_cache:
            stations, cached = await _get_stations_for_project(site, project, request.token)
            if cached:
                any_cached = True
            project_cache[cache_key] = stations

    # Step 3: Build results for each ISN
    for isn, (site, project) in isn_projects.items():
        cache_key = f"{site}:{project}"
        stations = project_cache.get(cache_key, [])

        results.append(
            IplasStationsFromIsnBatchItem(
                isn_info=IplasIsnProjectInfo(
                    isn=isn,
                    site=site,
                    project=project,
                    found=True,
                ),
                stations=stations,
                total_stations=len(stations),
            )
        )

    return IplasStationsFromIsnBatchResponse(
        results=results,
        total_isns=len(request.isns),
        unique_projects=len(unique_projects),
        not_found_isns=not_found_isns,
        cached=any_cached,
    )


# ============================================================================
# iPLAS v1 Download Attachment Endpoint
# ============================================================================


@router.post(
    "/download-attachment",
    response_model=IplasDownloadAttachmentResponse,
    summary="Download test log attachments from iPLAS",
    description="""
    Downloads test log attachments from iPLAS.
    
    Supports multiple downloads in a single request.
    Returns base64-encoded file content.
    """,
)
async def download_attachment(
    request: IplasDownloadAttachmentRequest,
) -> IplasDownloadAttachmentResponse:
    """Download test log attachments from iPLAS."""
    site_config = _get_site_config(request.site, request.token)
    url = f"{site_config['v1_url']}/file/{request.site}/{request.project}/download_attachment"

    # Build payload for iPLAS v1 API
    payload = {
        "info": [
            {
                "isn": item.isn,
                "time": item.time,
                "deviceid": item.deviceid,
                "station": item.station,
            }
            for item in request.info
        ],
        "token": site_config["token"],
    }

    logger.info(f"Downloading attachments: {request.site}/{request.project} ({len(request.info)} items)")

    try:
        async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
            response = await client.post(url, json=payload)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"iPLAS v1 API error: {response.text}",
                )

            data = response.json()

            if data.get("statuscode") != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"iPLAS API returned status {data.get('statuscode')}",
                )

            return IplasDownloadAttachmentResponse(
                content=data["data"]["content"],
                filename=data["data"].get("filename"),
            )

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"iPLAS v1 API unavailable: {e}") from None


# ============================================================================
# iPLAS v1 Download CSV Test Log Endpoint
# ============================================================================


@router.post(
    "/download-csv-log",
    response_model=IplasDownloadCsvLogResponse,
    summary="Download CSV test logs from iPLAS",
    description="""        
    **Important**: The test_end_time field MUST include milliseconds (e.g., '2026/01/22 18:57:05.000')
    
    Returns CSV file content as string with optional filename from response header.
    """,
)
async def download_csv_log(
    request: IplasDownloadCsvLogRequest,
) -> IplasDownloadCsvLogResponse:
    """Download CSV test logs from iPLAS."""
    if not request.query_list:
        raise HTTPException(status_code=400, detail="query_list cannot be empty")

    # Use the first item to determine site config
    first_item = request.query_list[0]
    site_config = _get_site_config(first_item.site, request.token)
    url = f"{site_config['v1_url']}/raw/get_test_log"

    # Build payload for iPLAS v1 API
    payload = {
        "query_list": [
            {
                "site": item.site,
                "project": item.project,
                "station": item.station,
                "line": item.line,
                "model": item.model,
                "deviceid": item.deviceid,
                "isn": item.isn,
                "test_end_time": item.test_end_time,
                "data_source": item.data_source,
            }
            for item in request.query_list
        ],
        "token": site_config["token"],
    }

    logger.info(f"Downloading CSV logs: {len(request.query_list)} items")

    try:
        async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
            response = await client.post(url, json=payload)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"iPLAS v1 API error: {response.text}",
                )

            # Get filename from Content-Disposition header if available
            filename = None
            content_disposition = response.headers.get("content-disposition", "")
            if "filename=" in content_disposition:
                # Extract filename from header
                import re

                match = re.search(r'filename="?([^";\n]+)"?', content_disposition)
                if match:
                    filename = match.group(1)

            # Return CSV content as string
            return IplasDownloadCsvLogResponse(
                content=response.text,
                filename=filename,
            )

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"iPLAS v1 API unavailable: {e}") from None


# ============================================================================
# iPLAS Batch Download Endpoint (TXT + CSV combined)
# ============================================================================


async def _download_single_csv_log(
    client: httpx.AsyncClient,
    site_config: dict,
    item: dict,
) -> tuple[str, str | None]:
    """Download a single CSV log file from iPLAS.

    Returns (csv_content, filename) or raises exception.
    """
    url = f"{site_config['v1_url']}/raw/get_test_log"
    payload = {
        "query_list": [item],
        "token": site_config["token"],
    }

    response = await client.post(url, json=payload)
    if response.status_code != 200:
        logger.warning(f"CSV download failed for {item.get('isn')}: {response.status_code}")
        return "", None

    # Get filename from header
    filename = None
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition:
        import re

        match = re.search(r'filename="?([^";\n]+)"?', content_disposition)
        if match:
            filename = match.group(1)

    return response.text, filename


async def _download_single_attachment(
    client: httpx.AsyncClient,
    site_config: dict,
    site: str,
    project: str,
    item: dict,
) -> tuple[bytes, str | None]:
    """Download a single TXT attachment from iPLAS.

    Returns (file_bytes, filename) or raises exception.
    """
    url = f"{site_config['v1_url']}/file/{site}/{project}/download_attachment"
    payload = {
        "info": [item],
        "token": site_config["token"],
    }

    response = await client.post(url, json=payload)
    if response.status_code != 200:
        logger.warning(f"TXT download failed for {item.get('isn')}: {response.status_code}")
        return b"", None

    data = response.json()
    if data.get("statuscode") != 200:
        logger.warning(f"TXT download failed for {item.get('isn')}: status {data.get('statuscode')}")
        return b"", None

    import base64

    content = base64.b64decode(data["data"]["content"])
    filename = data["data"].get("filename")

    return content, filename


@router.post(
    "/batch-download",
    response_model=IplasBatchDownloadResponse,
    summary="Batch download test logs (TXT, CSV, or both)",
    description="""
    Downloads multiple test logs in a single request, packaging them into a zip archive.
    
    **Download Types**:
    - `txt`: Download TXT attachments only
    - `csv`: Download CSV test logs only  
    - `all`: Download both TXT and CSV logs (default)
    
    **Performance**: Uses parallel requests for faster downloads.
    
    **Returns**: Base64-encoded zip file containing all requested logs.
    """,
)
async def batch_download(
    request: IplasBatchDownloadRequest,
) -> IplasBatchDownloadResponse:
    """Batch download test logs with proper zip packaging."""
    import base64
    import io
    import zipfile

    if not request.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    site_config = _get_site_config(request.site, request.token)
    download_type = request.download_type.lower()

    if download_type not in ("txt", "csv", "all"):
        raise HTTPException(status_code=400, detail="download_type must be 'txt', 'csv', or 'all'")

    logger.info(f"Batch download: {len(request.items)} items, type={download_type}")

    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    txt_count = 0
    csv_count = 0

    async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Process each item
            for item in request.items:
                item_dict = {
                    "site": item.site,
                    "project": item.project,
                    "station": item.station,
                    "line": item.line,
                    "model": item.model,
                    "deviceid": item.deviceid,
                    "isn": item.isn,
                    "test_end_time": item.test_end_time,
                    "data_source": item.data_source,
                }

                # Base filename for this item
                safe_time = item.test_end_time.replace("/", "_").replace(":", "_").replace(" ", "_").replace(".", "_")
                base_filename = f"{item.isn}_{safe_time}"

                # Download TXT attachment if requested
                if download_type in ("txt", "all"):
                    try:
                        # Convert test_end_time format for attachment API
                        # From "2026/01/22 18:57:05.000" to "2026/01/22 18:57:05"
                        attachment_time = item.test_end_time.split(".")[0]

                        attachment_item = {
                            "isn": item.isn,
                            "time": attachment_time,
                            "deviceid": item.deviceid,
                            "station": item.station,
                        }

                        content, filename = await _download_single_attachment(client, site_config, request.site, request.project, attachment_item)

                        if content:
                            # Use original filename or generate one
                            zip_filename = filename or f"{base_filename}.zip"
                            # Add to zip - the content is already a zip, so add its contents
                            try:
                                inner_zip = zipfile.ZipFile(io.BytesIO(content))
                                for inner_name in inner_zip.namelist():
                                    inner_data = inner_zip.read(inner_name)
                                    # Prefix with txt/ folder
                                    zip_file.writestr(f"txt/{inner_name}", inner_data)
                                txt_count += 1
                            except zipfile.BadZipFile:
                                # Not a zip, add as-is
                                zip_file.writestr(f"txt/{zip_filename}", content)
                                txt_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to download TXT for {item.isn}: {e}")

                # Download CSV log if requested
                if download_type in ("csv", "all"):
                    try:
                        csv_content, csv_filename = await _download_single_csv_log(client, site_config, item_dict)

                        if csv_content:
                            filename = csv_filename or f"{base_filename}.csv"
                            # Prefix with csv/ folder
                            zip_file.writestr(f"csv/{filename}", csv_content.encode("utf-8"))
                            csv_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to download CSV for {item.isn}: {e}")

    # Get zip content
    zip_buffer.seek(0)
    zip_content = zip_buffer.read()

    if not zip_content or (txt_count == 0 and csv_count == 0):
        raise HTTPException(status_code=404, detail="No files were downloaded successfully")

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if download_type == "txt":
        filename = f"test_logs_txt_{timestamp}.zip"
    elif download_type == "csv":
        filename = f"test_logs_csv_{timestamp}.zip"
    else:
        filename = f"test_logs_all_{timestamp}.zip"

    logger.info(f"Batch download complete: {txt_count} TXT + {csv_count} CSV files")

    return IplasBatchDownloadResponse(
        content=base64.b64encode(zip_content).decode("utf-8"),
        filename=filename,
        file_count=txt_count + csv_count,
        txt_count=txt_count,
        csv_count=csv_count,
    )


# ============================================================================
# iPLAS v2 Verify Endpoint
# ============================================================================


@router.post(
    "/verify",
    response_model=IplasVerifyResponse,
    summary="Verify token access for site/project",
    description="""
    Verifies that the provided token has access to the specified site/project.
    
    This is useful for validating user-provided tokens before making other requests.
    """,
)
async def verify_access(
    request: IplasVerifyRequest,
) -> IplasVerifyResponse:
    """Verify token access for a site/project."""
    site_config = _get_site_config(request.site, request.token)
    url = f"{site_config['v2_url']}/verify"

    try:
        async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
            response = await client.get(
                url,
                params={"site": request.site, "project": request.project},
                headers={"Authorization": f"Bearer {site_config['token']}"},
            )

            if response.status_code == 200:
                data = response.json()
                return IplasVerifyResponse(
                    success=data.get("message") == "success",
                    message=data.get("message", "Unknown response"),
                )
            else:
                return IplasVerifyResponse(
                    success=False,
                    message=f"HTTP {response.status_code}: {response.text}",
                )

    except httpx.RequestError as e:
        return IplasVerifyResponse(
            success=False,
            message=f"Connection error: {e}",
        )


# ============================================================================
# iPLAS v1 Get Test Item By ISN Endpoint
# ============================================================================


@router.post(
    "/test-item-by-isn",
    response_model=IplasTestItemByIsnResponse,
    summary="Get test items by ISN from iPLAS (cross-station search)",
    description="""
    Searches for an ISN across all related test stations within a date range.
    
    This is more flexible than the /isn_search` endpoint because it:
    - Supports date range filtering (begin_time/end_time)
    - Can filter by specific station or device
    - Returns test items from ALL stations that processed this ISN
    
    **Use Cases:**
    - Track a DUT through the entire production flow
    - Find all test records for a specific serial number
    - Compare test results across different stations for the same ISN
    
    **Cache TTL**: 5 minutes
    """,
)
async def get_test_item_by_isn(
    request: IplasTestItemByIsnRequest,
) -> IplasTestItemByIsnResponse:
    """Get test items by ISN with cross-station search capability."""
    redis = get_redis_client()

    # Build cache key including all parameters
    begin_str = request.begin_time.strftime("%Y%m%d%H%M%S")
    end_str = request.end_time.strftime("%Y%m%d%H%M%S")
    cache_key = f"iplas:v1:test-item-by-isn:{request.site}:{request.project}:{request.isn}:{request.station}:{request.device}:{begin_str}:{end_str}"

    cached = False
    results: list[IplasTestItemByIsnRecord] = []

    # Try cache first
    if redis:
        try:
            cached_data = redis.get(cache_key)
            if cached_data:
                data = json_loads(cached_data)
                results = [IplasTestItemByIsnRecord(**item) for item in data]
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")
        site_config = _get_site_config(request.site, request.token)
        url = f"{site_config['v1_url']}/{request.site}/{request.project}/dut/get_test_item_by_isn"

        # Build payload for iPLAS v1 API
        payload = {
            "ISN": request.isn,
            "station": request.station or "",
            "model": "",  # Leave empty as per API doc
            "line": "",  # Leave empty as per API doc
            "device": request.device or "",
            "begintime": request.begin_time.strftime("%Y/%m/%d %H:%M:%S"),
            "endtime": request.end_time.strftime("%Y/%m/%d %H:%M:%S"),
            "token": site_config["token"],
        }

        logger.info(f"Fetching test items by ISN: {request.isn} from {request.site}/{request.project}")

        try:
            async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
                response = await client.post(url, json=payload)

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"iPLAS v1 API error: {response.text}",
                    )

                data = response.json()

                # Check for error response
                if "error_msg" in data:
                    raise HTTPException(
                        status_code=400,
                        detail=data["error_msg"],
                    )

                # Parse the response data
                raw_data = data.get("data", [])
                for item in raw_data:
                    # Convert test_item format from API to our schema
                    test_items = []
                    for ti in item.get("test_item", []):
                        test_items.append(
                            IplasTestItemByIsnTestItem(
                                name=ti.get("name", ""),
                                Status=ti.get("Status", ""),
                                LSL=ti.get("LSL", ""),
                                Value=ti.get("Value", ""),
                                USL=ti.get("USL", ""),
                                CYCLE=ti.get("CYCLE", ""),
                            )
                        )

                    results.append(
                        IplasTestItemByIsnRecord(
                            site=item.get("site", request.site),
                            project=item.get("project", request.project),
                            ISN=item.get("ISN", request.isn),
                            station=item.get("station", ""),
                            model=item.get("model", ""),
                            line=item.get("line", ""),
                            device=item.get("device", ""),
                            test_end_time=item.get("test_end_time", ""),
                            test_item=test_items,
                        )
                    )

                logger.info(f"Found {len(results)} records for ISN {request.isn}")

        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"iPLAS v1 API unavailable: {e}") from None

        # Store in cache
        if redis and results:
            try:
                serialized = json_dumps([r.model_dump() for r in results])
                redis.setex(cache_key, IPLAS_CACHE_TTL_ISN, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL_ISN}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    return IplasTestItemByIsnResponse(
        data=results,
        total_count=len(results),
        cached=cached,
    )


# ============================================================================
# Streaming Endpoint for Large Datasets
# ============================================================================


@router.post(
    "/csv-test-items/stream",
    summary="Stream CSV test items as NDJSON",
    description="""
    Streams test item data as newline-delimited JSON (NDJSON) for reduced memory usage.
    
    Benefits:
    - Frontend can process records as they arrive (progressive rendering)
    - Reduced peak memory usage on both backend and frontend
    - Better UX for large datasets (users see data immediately)
    
    Response format: One JSON object per line, each line is a complete record.
    Content-Type: application/x-ndjson
    
    Use this endpoint when fetching large datasets (1000+ records) to avoid
    loading everything into memory at once.
    """,
)
async def stream_csv_test_items(
    request: IplasCsvTestItemRequest,
) -> StreamingResponse:
    """Stream CSV test items as NDJSON for large datasets."""

    async def generate():
        """Generator that yields NDJSON records."""
        try:
            redis = get_redis_client()
            cache_key = _generate_cache_key(
                request.site,
                request.project,
                request.station,
                request.device_id,
                request.begin_time,
                request.end_time,
                request.test_status,
            )

            records: list[dict[str, Any]] = []
            cached = False

            # Try cache first
            if redis:
                try:
                    cached_data = redis.get(cache_key)
                    if cached_data:
                        records = json_loads(cached_data)
                        cached = True
                        logger.debug(f"Cache HIT (stream): {cache_key}")
                except Exception as e:
                    logger.warning(f"Redis GET error in stream: {e}")

            # Fetch from iPLAS if cache miss
            if not cached:
                logger.debug(f"Cache MISS (stream): {cache_key}")
                records, possibly_truncated, chunks_fetched, total_chunks, used_hybrid = await _fetch_chunked_from_iplas(
                    request.site,
                    request.project,
                    request.station,
                    request.device_id,
                    request.begin_time,
                    request.end_time,
                    request.test_status,
                    request.token,
                )

                # Store in cache for subsequent requests
                if redis:
                    try:
                        serialized = json_dumps(records)
                        redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                        logger.debug(f"Cached (stream): {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
                    except Exception as e:
                        logger.warning(f"Redis SET error in stream: {e}")
            else:
                possibly_truncated = False
                chunks_fetched = 1
                total_chunks = 1
                used_hybrid = False

            # Apply test item filtering if specified
            if request.test_item_filters:
                records = _filter_test_items(records, request.test_item_filters)

            total_records = len(records)

            # Apply sorting if specified
            if request.sort_by:
                field_mapping = {
                    "TestStartTime": "Test Start Time",
                    "TestEndTime": "Test end Time",
                    "TestStatus": "Test Status",
                    "ISN": "ISN",
                    "DeviceId": "DeviceId",
                    "ErrorCode": "ErrorCode",
                    "ErrorName": "ErrorName",
                    "station": "station",
                    "Site": "Site",
                    "Project": "Project",
                }
                sort_field = field_mapping.get(request.sort_by, request.sort_by)

                try:
                    records = sorted(
                        records,
                        key=lambda r: r.get(sort_field, "") or "",
                        reverse=request.sort_desc,
                    )
                except Exception as e:
                    logger.warning(f"Sorting by {sort_field} failed in stream: {e}")

            # Yield metadata header first
            metadata = {
                "_metadata": True,
                "total_records": total_records,
                "filtered": bool(request.test_item_filters),
                "cached": cached,
                "possibly_truncated": possibly_truncated,
                "chunks_fetched": chunks_fetched,
                "total_chunks": total_chunks,
                "used_hybrid_strategy": used_hybrid,
            }
            yield json_dumps(metadata) + b"\n"

            # Apply pagination offset
            start_idx = request.offset or 0
            end_idx = start_idx + request.limit if request.limit else len(records)

            # Yield records one by one
            for record in records[start_idx:end_idx]:
                # Convert to compact format for streaming (excludes TestItem array)
                compact = _convert_to_compact_record(record)
                yield json_dumps(compact.model_dump()) + b"\n"

        except HTTPException as e:
            # Handle expected HTTP errors (e.g., upstream connection failures)
            logger.error(f"HTTPException in stream generator: {e.status_code} - {e.detail}")
            error_metadata = {
                "_metadata": True,
                "_error": True,
                "error_code": e.status_code,
                "error_message": e.detail,
                "total_records": 0,
            }
            yield json_dumps(error_metadata) + b"\n"
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error in stream generator: {e}")
            error_metadata = {
                "_metadata": True,
                "_error": True,
                "error_code": 500,
                "error_message": f"Internal server error: {str(e)}",
                "total_records": 0,
            }
            yield json_dumps(error_metadata) + b"\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )
