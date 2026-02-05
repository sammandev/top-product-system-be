"""
Pydantic schemas for iPLAS proxy API endpoints.

These schemas handle request/response validation for the iPLAS external API proxy.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IplasCsvTestItemRequest(BaseModel):
    """Request schema for fetching filtered CSV test items from iPLAS."""

    site: str = Field(..., description="Site identifier (e.g., 'PTB')")
    project: str = Field(..., description="Project identifier (e.g., 'HH5K')")
    station: str = Field(..., description="Station display name (e.g., 'Wireless_Test_2_5G')")
    device_id: str = Field(..., description="Device ID or 'ALL' for all devices")
    begin_time: datetime = Field(..., description="Start time for the query")
    end_time: datetime = Field(..., description="End time for the query")
    test_status: Literal["ALL", "PASS", "FAIL"] = Field(
        default="ALL", description="Filter by test status"
    )
    test_item_filters: list[str] | None = Field(
        default=None,
        description="List of test item names to filter. If empty or None, returns all test items.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Maximum number of records to return (for pagination)",
    )
    offset: int | None = Field(
        default=None,
        ge=0,
        description="Number of records to skip (for pagination)",
    )
    sort_by: str | None = Field(
        default=None,
        description="Field name to sort by (e.g., 'TestStartTime', 'ISN', 'TestStatus')",
    )
    sort_desc: bool = Field(
        default=True,
        description="Sort in descending order (default True for newest first)",
    )
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasTestItemNamesRequest(BaseModel):
    """Request schema for fetching unique test item names from iPLAS."""

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    station: str = Field(..., description="Station display name")
    device_id: str = Field(default="ALL", description="Device ID or 'ALL' for all devices")
    begin_time: datetime = Field(..., description="Start time for the query")
    end_time: datetime = Field(..., description="End time for the query")
    test_status: Literal["ALL", "PASS", "FAIL"] = Field(
        default="PASS", description="Filter by test status (defaults to PASS for test item discovery)"
    )
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )
    # UPDATED: Add option to exclude BIN test items for scoring dialogs
    exclude_bin: bool = Field(
        default=False,
        description="If true, excludes BIN/PASS-FAIL test items (is_bin=True) from results. "
                    "Useful for scoring dialogs that only need CRITERIA and NON-CRITERIA items.",
    )


class IplasTestItemInfo(BaseModel):
    """Test item information with type indicator."""

    name: str = Field(..., description="Test item name")
    is_value: bool = Field(
        ..., description="True if the test item has numeric VALUE (not bin/pass-fail)"
    )
    is_bin: bool = Field(
        default=False, description="True if the test item is binary (PASS/FAIL only)"
    )
    has_ucl: bool = Field(
        default=False, description="True if the test item has a UCL (upper control limit)"
    )
    has_lcl: bool = Field(
        default=False, description="True if the test item has a LCL (lower control limit)"
    )


class IplasTestItemNamesResponse(BaseModel):
    """Response schema for unique test item names."""

    test_items: list[IplasTestItemInfo] = Field(
        ..., description="List of unique test item names with type info"
    )
    total_count: int = Field(..., description="Total number of unique test items")


class IplasRecordTestItemsRequest(BaseModel):
    """Request schema for fetching test items for a specific record."""

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    station: str = Field(..., description="Station name")
    isn: str = Field(..., description="ISN of the record")
    test_start_time: str = Field(..., description="Test start time (for unique identification)")
    device_id: str = Field(default="ALL", description="Device ID filter")
    test_status: Literal["ALL", "PASS", "FAIL"] = Field(default="ALL")
    token: str | None = Field(default=None, description="Optional user-provided token")


class IplasRecordTestItemsResponse(BaseModel):
    """Response schema for test items of a specific record."""

    isn: str = Field(..., description="ISN of the record")
    test_start_time: str = Field(..., description="Test start time")
    test_items: list[dict] = Field(..., description="Test items for this record")
    test_item_count: int = Field(..., description="Number of test items")
    cached: bool = Field(default=False, description="True if data was served from cache")


class IplasTestItem(BaseModel):
    """Individual test item data from iPLAS."""

    NAME: str
    STATUS: str
    VALUE: str
    UCL: str = ""
    LCL: str = ""
    CYCLE: str = ""


class IplasCsvTestItemRecord(BaseModel):
    """Single test record from iPLAS CSV test item endpoint."""

    Site: str
    Project: str
    station: str
    TSP: str
    Model: str = ""
    MO: str = ""
    Line: str
    ISN: str
    DeviceId: str
    TestStatus: str = Field(..., alias="Test Status")
    TestStartTime: str = Field(..., alias="Test Start Time")
    TestEndTime: str = Field(..., alias="Test end Time")
    ErrorCode: str
    ErrorName: str
    TestItem: list[IplasTestItem] = []

    class Config:
        populate_by_name = True


class IplasCsvTestItemResponse(BaseModel):
    """Response schema for filtered CSV test items."""

    data: list[dict] = Field(..., description="Filtered test item records")
    total_records: int = Field(..., description="Total records before pagination")
    returned_records: int = Field(..., description="Number of records returned")
    filtered: bool = Field(
        ..., description="True if test item filtering was applied"
    )
    cached: bool = Field(..., description="True if data was served from cache")
    possibly_truncated: bool = Field(
        default=False,
        description="True if any chunk hit the 5000 record limit (data may be incomplete)"
    )
    # Chunking metadata for progress indicators
    chunks_fetched: int = Field(
        default=1,
        description="Number of API chunks fetched (for queries >6 days)"
    )
    total_chunks: int = Field(
        default=1,
        description="Total number of chunks (for queries >6 days)"
    )
    # Hybrid V1/V2 strategy metadata
    used_hybrid_strategy: bool = Field(
        default=False,
        description="True if hybrid V1/V2 strategy was used (per-device fetching for high-density stations)"
    )


class CompactCsvTestItemRecord(BaseModel):
    """
    Compact record without TestItem array for memory-efficient list views.
    
    Use this for displaying record lists. TestItems can be loaded on-demand
    via a separate endpoint when the user expands a record.
    """
    Site: str
    Project: str
    station: str
    TSP: str
    Model: str = ""
    MO: str = ""
    Line: str
    ISN: str
    DeviceId: str
    TestStatus: str = Field(..., alias="Test Status")
    TestStartTime: str = Field(..., alias="Test Start Time")
    TestEndTime: str = Field(..., alias="Test end Time")
    ErrorCode: str
    ErrorName: str
    TestItemCount: int = Field(default=0, description="Number of test items in record")

    class Config:
        populate_by_name = True


class CompactCsvTestItemResponse(BaseModel):
    """Response schema for compact CSV test items (without TestItem arrays)."""

    data: list[CompactCsvTestItemRecord] = Field(..., description="Compact test item records")
    total_records: int = Field(..., description="Total records before pagination")
    returned_records: int = Field(..., description="Number of records returned")
    filtered: bool = Field(
        ..., description="True if test item filtering was applied"
    )
    cached: bool = Field(..., description="True if data was served from cache")
    possibly_truncated: bool = Field(
        default=False,
        description="True if any chunk hit the 5000 record limit (data may be incomplete)"
    )
    # Chunking metadata for progress indicators
    chunks_fetched: int = Field(
        default=1,
        description="Number of API chunks fetched (for queries >6 days)"
    )
    total_chunks: int = Field(
        default=1,
        description="Total number of chunks (for queries >6 days)"
    )
    # Hybrid V1/V2 strategy metadata
    used_hybrid_strategy: bool = Field(
        default=False,
        description="True if hybrid V1/V2 strategy was used (per-device fetching for high-density stations)"
    )


# ============================================================================
# iPLAS v2 API Schemas
# ============================================================================


class SiteProject(BaseModel):
    """Site and project pair from iPLAS v2 API."""

    site: str = Field(..., description="Site identifier (e.g., 'PTB')")
    project: str = Field(..., description="Project identifier (e.g., 'HH5K')")


class IplasSiteProjectListResponse(BaseModel):
    """Response schema for site/project list."""

    data: list[SiteProject] = Field(..., description="List of site/project pairs")
    total_count: int = Field(..., description="Total number of site/project pairs")
    cached: bool = Field(..., description="True if data was served from cache")


class IplasStation(BaseModel):
    """Station information from iPLAS v2 API."""

    display_station_name: str = Field(..., description="Display name of the station")
    station_name: str = Field(..., description="Internal station name")
    order: int = Field(default=0, description="Station order")
    data_source: str = Field(default="Test Station", description="Data source type")


class IplasStationListRequest(BaseModel):
    """Request schema for fetching station list."""

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasStationListResponse(BaseModel):
    """Response schema for station list."""

    data: list[IplasStation] = Field(..., description="List of stations")
    total_count: int = Field(..., description="Total number of stations")
    cached: bool = Field(..., description="True if data was served from cache")


class IplasDeviceListRequest(BaseModel):
    """Request schema for fetching device list."""

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    station: str = Field(..., description="Display station name")
    start_time: datetime = Field(..., description="Start time for the query")
    end_time: datetime = Field(..., description="End time for the query")
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasDeviceListResponse(BaseModel):
    """Response schema for device list."""

    data: list[str] = Field(..., description="List of device IDs")
    total_count: int = Field(..., description="Total number of devices")
    cached: bool = Field(..., description="True if data was served from cache")


class IplasIsnSearchRequest(BaseModel):
    """Request schema for ISN search."""

    isn: str = Field(..., description="ISN to search for")
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasIsnTestItem(BaseModel):
    """Test item from ISN search result."""

    NAME: str
    STATUS: str
    VALUE: str
    UCL: str = ""
    LCL: str = ""
    CYCLE: str = ""


class IplasIsnSearchRecord(BaseModel):
    """Single record from ISN search result."""

    site: str
    project: str
    isn: str
    error_name: str = ""
    station_name: str
    slot: str = ""
    error_code: str
    error_message: str = ""
    test_status: str
    line: str
    test_start_time: str
    total_testing_time: str = ""
    test_item: list[IplasIsnTestItem] = []
    mo: str = ""
    test_end_time: str
    device_id: str
    file_token: str = ""
    project_token: str = ""
    display_station_name: str


class IplasIsnSearchResponse(BaseModel):
    """Response schema for ISN search."""

    data: list[IplasIsnSearchRecord] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    cached: bool = Field(..., description="True if data was served from cache")


# ============================================================================
# iPLAS v1 Download Attachment Schemas
# ============================================================================


class IplasDownloadAttachmentInfo(BaseModel):
    """Single attachment info for download request."""

    isn: str = Field(..., description="Device serial number")
    time: str = Field(..., description="Test time in format 'YYYY/MM/DD HH:mm:ss'")
    deviceid: str = Field(..., description="Device ID")
    station: str = Field(..., description="Station name")


class IplasDownloadAttachmentRequest(BaseModel):
    """Request schema for downloading test log attachments."""

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    info: list[IplasDownloadAttachmentInfo] = Field(
        ..., description="List of attachment info to download"
    )
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasDownloadAttachmentResponse(BaseModel):
    """Response schema for download attachment."""

    content: str = Field(..., description="Base64 encoded file content")
    filename: str | None = Field(
        default=None, description="Filename (only when single file)"
    )


# ============================================================================
# iPLAS v1 Download CSV Test Log Schemas
# ============================================================================


class IplasDownloadCsvLogInfo(BaseModel):
    """Single CSV log info for download request.
    
    Based on iPLAS v1 API: POST /raw/get_test_log
    """

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    station: str = Field(..., description="Station display name")
    line: str = Field(..., description="Production line")
    model: str = Field(default="ALL", description="Model (usually 'ALL')")
    deviceid: str = Field(..., description="Device ID")
    isn: str = Field(..., description="Device serial number")
    test_end_time: str = Field(
        ..., 
        description="Test end time in format 'YYYY/MM/DD HH:mm:ss.000' (MUST include .000)"
    )
    data_source: int = Field(default=0, description="Data source (usually 0)")


class IplasDownloadCsvLogRequest(BaseModel):
    """Request schema for downloading CSV test logs."""

    query_list: list[IplasDownloadCsvLogInfo] = Field(
        ..., description="List of test log queries to download"
    )
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasDownloadCsvLogResponse(BaseModel):
    """Response schema for download CSV log."""

    content: str = Field(..., description="CSV file content as string")
    filename: str | None = Field(
        default=None, description="Filename from response header"
    )


# ============================================================================
# iPLAS v2 Verify Endpoint Schemas
# ============================================================================


class IplasVerifyRequest(BaseModel):
    """Request schema for verifying token access."""

    site: str = Field(..., description="Site identifier")
    project: str = Field(..., description="Project identifier")
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasVerifyResponse(BaseModel):
    """Response schema for verify endpoint."""

    success: bool = Field(..., description="True if token has access")
    message: str = Field(..., description="Verification message")


# ============================================================================
# iPLAS v1 Get Test Item By ISN Schemas
# ============================================================================


class IplasTestItemByIsnRequest(BaseModel):
    """Request schema for fetching test items by ISN from iPLAS v1 API.
    
    This endpoint searches for an ISN across all related stations within a date range.
    More flexible than the V2 isn_search as it supports date filtering.
    
    Based on: POST /{site}/{project}/dut/get_test_item_by_isn
    """

    site: str = Field(..., description="Site identifier (e.g., 'PTB')")
    project: str = Field(..., description="Project identifier (e.g., 'HH5K')")
    isn: str = Field(..., description="Device serial number to search for")
    station: str = Field(
        default="",
        description="Station filter (leave empty to search all related stations)"
    )
    device: str = Field(
        default="",
        description="Device ID filter (leave empty to search all device IDs)"
    )
    begin_time: datetime = Field(..., description="Start time for the search range")
    end_time: datetime = Field(..., description="End time for the search range")
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasTestItemByIsnTestItem(BaseModel):
    """Test item from get_test_item_by_isn response."""

    name: str
    Status: str
    LSL: str = ""  # Lower Spec Limit (same as LCL)
    Value: str = ""
    USL: str = ""  # Upper Spec Limit (same as UCL)
    CYCLE: str = ""


class IplasTestItemByIsnRecord(BaseModel):
    """Single record from get_test_item_by_isn response."""

    site: str
    project: str
    ISN: str
    station: str
    model: str = ""
    line: str
    device: str
    test_end_time: str
    test_item: list[IplasTestItemByIsnTestItem] = []


class IplasTestItemByIsnResponse(BaseModel):
    """Response schema for get_test_item_by_isn endpoint."""

    data: list[IplasTestItemByIsnRecord] = Field(
        ..., description="List of matching records across all related stations"
    )
    total_count: int = Field(..., description="Total number of records found")
    cached: bool = Field(default=False, description="True if data was served from cache")


# ============================================================================
# iPLAS v2 Stations From ISN Schemas
# ============================================================================


class IplasStationsFromIsnRequest(BaseModel):
    """Request schema for getting station list from ISN.
    
    This endpoint first looks up the ISN to find its site/project,
    then fetches the full station list for that project.
    """

    isn: str = Field(..., description="ISN to look up")
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasStationsFromIsnBatchRequest(BaseModel):
    """Request schema for getting station lists from multiple ISNs.
    
    For each unique site/project pair found, the station list is fetched.
    Results are deduplicated if multiple ISNs belong to the same project.
    """

    isns: list[str] = Field(
        ..., 
        description="List of ISNs to look up",
        min_length=1,
        max_length=50,
    )
    token: str | None = Field(
        default=None,
        description="Optional user-provided token. If not provided, uses backend default.",
    )


class IplasIsnProjectInfo(BaseModel):
    """Information about an ISN's project discovered from ISN search."""

    isn: str = Field(..., description="The ISN that was searched")
    site: str = Field(..., description="Site where ISN was found")
    project: str = Field(..., description="Project where ISN belongs")
    found: bool = Field(default=True, description="Whether the ISN was found")


class IplasStationsFromIsnResponse(BaseModel):
    """Response schema for stations from ISN lookup.
    
    Contains the ISN's site/project info and all stations for that project.
    """

    isn_info: IplasIsnProjectInfo = Field(
        ..., description="Information about the ISN's project"
    )
    stations: list[IplasStation] = Field(
        ..., description="List of all stations for the ISN's project"
    )
    total_stations: int = Field(..., description="Total number of stations")
    cached: bool = Field(default=False, description="True if data was served from cache")


class IplasStationsFromIsnBatchItem(BaseModel):
    """Single item in batch stations lookup response."""

    isn_info: IplasIsnProjectInfo = Field(
        ..., description="Information about the ISN's project"
    )
    stations: list[IplasStation] = Field(
        ..., description="List of all stations for the ISN's project"
    )
    total_stations: int = Field(..., description="Total number of stations")


class IplasStationsFromIsnBatchResponse(BaseModel):
    """Response schema for batch stations from ISN lookup.
    
    Returns station lists for each unique site/project found.
    ISNs sharing the same project will have the same station list.
    """

    results: list[IplasStationsFromIsnBatchItem] = Field(
        ..., description="Station lists for each ISN"
    )
    total_isns: int = Field(..., description="Total number of ISNs processed")
    unique_projects: int = Field(..., description="Number of unique site/project pairs")
    not_found_isns: list[str] = Field(
        default_factory=list,
        description="ISNs that were not found in any site"
    )
    cached: bool = Field(default=False, description="True if any data was served from cache")

