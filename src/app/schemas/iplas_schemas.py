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


class IplasTestItemInfo(BaseModel):
    """Test item information with type indicator."""

    name: str = Field(..., description="Test item name")
    is_value: bool = Field(
        ..., description="True if the test item has numeric VALUE (not bin/pass-fail)"
    )
    is_bin: bool = Field(
        default=False, description="True if the test item is binary (PASS/FAIL only)"
    )


class IplasTestItemNamesResponse(BaseModel):
    """Response schema for unique test item names."""

    test_items: list[IplasTestItemInfo] = Field(
        ..., description="List of unique test item names with type info"
    )
    total_count: int = Field(..., description="Total number of unique test items")


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

