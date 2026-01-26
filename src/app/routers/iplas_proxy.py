"""
iPLAS External API Proxy Router.

Provides cached proxy endpoints for iPLAS v1 and v2 APIs with:
- Multi-site support (PTB, PSZ, PXD, PVN, PTY)
- Redis caching for performance
- Server-side filtering to reduce browser memory usage
"""

import logging
import os
from datetime import datetime
from typing import Any, Literal

import httpx
import orjson
from fastapi import APIRouter, HTTPException, Query
from redis import Redis

from app.schemas.iplas_schemas import (
    IplasCsvTestItemRequest,
    IplasCsvTestItemResponse,
    IplasDeviceListRequest,
    IplasDeviceListResponse,
    IplasDownloadAttachmentRequest,
    IplasDownloadAttachmentResponse,
    IplasIsnSearchRequest,
    IplasIsnSearchResponse,
    IplasSiteProjectListResponse,
    IplasStation,
    IplasStationListRequest,
    IplasStationListResponse,
    IplasTestItemInfo,
    IplasTestItemNamesRequest,
    IplasTestItemNamesResponse,
    IplasVerifyRequest,
    IplasVerifyResponse,
    SiteProject,
)

logger = logging.getLogger(__name__)

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


async def _fetch_from_iplas(
    site: str,
    project: str,
    station: str,
    device_id: str,
    begin_time: datetime,
    end_time: datetime,
    test_status: str,
    user_token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch test item data from iPLAS v1 API with multi-site support."""
    site_config = _get_site_config(site, user_token)
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

    async with httpx.AsyncClient(timeout=IPLAS_TIMEOUT) as client:
        response = await client.post(url, json=payload)

        if response.status_code != 200:
            logger.error(f"iPLAS API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"iPLAS API error: {response.text}",
            )

        data = response.json()

        if data.get("statuscode") != 200:
            raise HTTPException(
                status_code=500,
                detail=f"iPLAS API returned status {data.get('statuscode')}",
            )

        return data.get("data", [])


def _filter_test_items(
    records: list[dict[str, Any]], test_item_filters: list[str] | None
) -> list[dict[str, Any]]:
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
        filtered_items = [
            item for item in test_items if item.get("NAME") in filter_set
        ]

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
        List of unique test item info with name and is_value flag
    """
    test_items_map: dict[str, IplasTestItemInfo] = {}

    for record in records:
        test_items = record.get("TestItem", [])
        for item in test_items:
            name = item.get("NAME")
            if name and name not in test_items_map:
                # Determine if it's a VALUE type (numeric), BIN type (PASS/FAIL), or Non-Value
                value = item.get("VALUE", "").upper().strip()
                is_value = False
                is_bin = False
                
                if value in ("PASS", "FAIL", "-999"):
                    # Binary data - PASS/FAIL only
                    is_bin = True
                elif value and value not in ("",):
                    # Try to parse as numeric value
                    try:
                        float(value)
                        is_value = True
                    except (ValueError, TypeError):
                        # Non-value: not numeric and not PASS/FAIL
                        is_value = False

                test_items_map[name] = IplasTestItemInfo(name=name, is_value=is_value, is_bin=is_bin)

    # Sort by name
    return sorted(test_items_map.values(), key=lambda x: x.name)


@router.post(
    "/csv-test-items",
    response_model=IplasCsvTestItemResponse,
    summary="Get filtered CSV test items from iPLAS",
    description="""
    Fetches test item data from iPLAS v1 API with server-side caching and filtering.
    
    This endpoint:
    1. Checks Redis cache for existing data
    2. Fetches from iPLAS API if cache miss
    3. Applies test_item_filters on server-side (reduces payload to frontend)
    4. Supports pagination with limit/offset
    
    **Cache TTL**: Data is cached for 3 minutes (configurable via IPLAS_CACHE_TTL env var)
    """,
)
async def get_csv_test_items(
    request: IplasCsvTestItemRequest,
) -> IplasCsvTestItemResponse:
    """Get filtered CSV test items with caching."""
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
                records = orjson.loads(cached_data)
                cached = True
                logger.debug(f"Cache HIT: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not cached:
        logger.debug(f"Cache MISS: {cache_key}")
        records = await _fetch_from_iplas(
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
                serialized = orjson.dumps(records)
                redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

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
                records = orjson.loads(cached_data)
                logger.debug(f"Cache HIT (names): {cache_key}")
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")

    # Fetch from iPLAS if cache miss
    if not records:
        logger.debug(f"Cache MISS (names): {cache_key}")
        records = await _fetch_from_iplas(
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
                serialized = orjson.dumps(records)
                redis.setex(cache_key, IPLAS_CACHE_TTL, serialized)
                logger.debug(f"Cached: {cache_key} (TTL={IPLAS_CACHE_TTL}s)")
            except Exception as e:
                logger.warning(f"Redis SET error: {e}")

    # Extract unique test item names
    test_items = _extract_unique_test_items(records)

    return IplasTestItemNamesResponse(
        test_items=test_items,
        total_count=len(test_items),
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
    "/v2/site-projects",
    response_model=IplasSiteProjectListResponse,
    summary="Get site/project list from iPLAS v2",
    description="""
    Fetches the list of all available site/project pairs from iPLAS v2 API.
    
    **Cache TTL**: 24 hours (data rarely changes)
    
    Note: This endpoint queries all configured sites and aggregates results.
    """,
)
async def get_site_projects(
    data_type: Literal["simple", "strict"] = Query(
        default="simple", description="Data type filter"
    ),
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
                data = orjson.loads(cached_data)
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
                serialized = orjson.dumps([p.model_dump() for p in all_projects])
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
    "/v2/stations",
    response_model=IplasStationListResponse,
    summary="Get station list for a project from iPLAS v2",
    description="""
    Fetches the list of stations for a specific site/project from iPLAS v2 API.
    
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
                data = orjson.loads(cached_data)
                stations = [IplasStation(**item) for item in data]
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

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"iPLAS v2 API unavailable: {e}"
            ) from None

        # Store in cache
        if redis and stations:
            try:
                serialized = orjson.dumps([s.model_dump() for s in stations])
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
    "/v2/devices",
    response_model=IplasDeviceListResponse,
    summary="Get device list for a station from iPLAS v2",
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
                devices = orjson.loads(cached_data)
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
            raise HTTPException(
                status_code=503, detail=f"iPLAS v2 API unavailable: {e}"
            ) from None

        # Store in cache
        if redis and devices:
            try:
                serialized = orjson.dumps(devices)
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
    "/v2/isn-search",
    response_model=IplasIsnSearchResponse,
    summary="Search DUT by ISN from iPLAS v2",
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
                results = orjson.loads(cached_data)
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
                serialized = orjson.dumps(results)
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
# iPLAS v1 Download Attachment Endpoint
# ============================================================================


@router.post(
    "/v1/download-attachment",
    response_model=IplasDownloadAttachmentResponse,
    summary="Download test log attachments from iPLAS v1",
    description="""
    Downloads test log attachments from iPLAS v1 API.
    
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

    logger.info(
        f"Downloading attachments: {request.site}/{request.project} "
        f"({len(request.info)} items)"
    )

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
        raise HTTPException(
            status_code=503, detail=f"iPLAS v1 API unavailable: {e}"
        ) from None


# ============================================================================
# iPLAS v2 Verify Endpoint
# ============================================================================


@router.post(
    "/v2/verify",
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

