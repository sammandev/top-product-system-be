"""
SFISTSP API Router for ISN Reference Lookup.

This router exposes endpoints to look up ISN references (SSN, MAC address)
from the SFISTSP Web Service.

Endpoints:
- GET /api/sfistsp/isn/{isn} - Look up a single ISN
- POST /api/sfistsp/isn/batch - Look up multiple ISNs
- GET /api/sfistsp/config - Get SFISTSP configuration info
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from ..external_services.sfistsp_api_client import (
    SFISTSP_DEFAULT_BASE_URL,
    SFISTSP_ENDPOINT_PATH,
    SfistspConfig,
    create_sfistsp_client,
)
from ..schemas.sfistsp_schemas import (
    SfistspConfigResponse,
    SfistspIsnBatchLookupRequest,
    SfistspIsnBatchLookupResponse,
    SfistspIsnReferenceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sfistsp", tags=["SFISTSP"])


# ============================================================================
# Configuration from Environment
# ============================================================================


def get_sfistsp_config() -> SfistspConfig:
    """
    Get SFISTSP configuration from environment variables.

    Environment variables (from .env.staging):
    - SFISTSP_API_BASE_URL: Base URL for SFISTSP server
    - SFIS_DEFAULT_PROGRAM_ID: Program ID for authentication
    - SFIS_DEFAULT_PROGRAM_PASSWORD: Program password
    - SFISTSP_API_TIMEOUT: Request timeout in seconds
    """
    return SfistspConfig(
        base_url=os.environ.get("SFISTSP_API_BASE_URL", SFISTSP_DEFAULT_BASE_URL),
        program_id=os.environ.get("SFIS_DEFAULT_PROGRAM_ID", "TSP_TDSTB"),
        program_password=os.environ.get("SFIS_DEFAULT_PROGRAM_PASSWORD", "ap_tbsus"),
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/isn/{isn}",
    response_model=SfistspIsnReferenceResponse,
    summary="Look up ISN references",
    description="Look up SSN and MAC address references for a given ISN from SFISTSP.",
)
async def lookup_isn(isn: str) -> SfistspIsnReferenceResponse:
    """
    Look up ISN references from SFISTSP.

    Args:
        isn: The ISN to look up (e.g., "264118730000123")

    Returns:
        SfistspIsnReferenceResponse with SSN, MAC, and other references
    """
    if not isn or not isn.strip():
        raise HTTPException(status_code=400, detail="ISN is required")

    isn = isn.strip()
    logger.info(f"SFISTSP ISN lookup request: {isn}")

    config = get_sfistsp_config()
    async with create_sfistsp_client(
        base_url=config.base_url,
        program_id=config.program_id,
        program_password=config.program_password,
    ) as client:
        result = await client.lookup_isn(isn)

    # UPDATED: isn_searched is the original search term, isn is the first reference (primary ISN)
    primary_isn = result.isn_references[0] if result.isn_references else result.isn
    return SfistspIsnReferenceResponse(
        isn_searched=result.isn,  # Original search term
        isn=primary_isn,  # Primary ISN from references
        ssn=result.ssn,
        mac=result.mac,
        success=result.success,
        error_message=result.error_message,
        isn_references=result.isn_references,
    )


@router.post(
    "/isn/batch",
    response_model=SfistspIsnBatchLookupResponse,
    summary="Batch ISN lookup",
    description="Look up multiple ISNs at once from SFISTSP.",
)
async def lookup_isns_batch(
    request: SfistspIsnBatchLookupRequest,
) -> SfistspIsnBatchLookupResponse:
    """
    Look up multiple ISNs from SFISTSP.

    Args:
        request: Batch lookup request with list of ISNs

    Returns:
        SfistspIsnBatchLookupResponse with results for each ISN
    """
    if not request.isns:
        raise HTTPException(status_code=400, detail="At least one ISN is required")

    # Clean ISNs
    isns = [isn.strip() for isn in request.isns if isn.strip()]
    if not isns:
        raise HTTPException(status_code=400, detail="No valid ISNs provided")

    logger.info(f"SFISTSP batch ISN lookup request: {len(isns)} ISNs")

    config = get_sfistsp_config()
    async with create_sfistsp_client(
        base_url=config.base_url,
        program_id=config.program_id,
        program_password=config.program_password,
    ) as client:
        results = await client.lookup_isns_batch(isns)

    # Convert to response models
    # UPDATED: isn_searched is the original search term, isn is the first reference (primary ISN)
    response_results = [
        SfistspIsnReferenceResponse(
            isn_searched=r.isn,  # Original search term
            isn=r.isn_references[0] if r.isn_references else r.isn,  # Primary ISN
            ssn=r.ssn,
            mac=r.mac,
            success=r.success,
            error_message=r.error_message,
            isn_references=r.isn_references,
        )
        for r in results
    ]

    success_count = sum(1 for r in results if r.success)

    return SfistspIsnBatchLookupResponse(
        results=response_results,
        total_count=len(results),
        success_count=success_count,
        failed_count=len(results) - success_count,
    )


@router.get(
    "/config",
    response_model=SfistspConfigResponse,
    summary="Get SFISTSP configuration",
    description="Get current SFISTSP configuration and availability status.",
)
async def get_config() -> SfistspConfigResponse:
    """
    Get SFISTSP configuration info.

    Returns:
        SfistspConfigResponse with configuration details
    """
    config = get_sfistsp_config()

    # Check availability by doing a simple test (could be cached)
    available = True  # Assume available, could do a health check here

    return SfistspConfigResponse(
        base_url=config.base_url,
        endpoint=SFISTSP_ENDPOINT_PATH,
        available=available,
    )
