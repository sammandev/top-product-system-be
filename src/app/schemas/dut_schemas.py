from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class SiteSchema(BaseModel):
    """Site information schema."""

    id: int
    name: str
    iplas_url: str | None = None
    ftp_url: str | None = None
    ftp_user: str | None = None
    ftp_password: str | None = None
    iplas_token: str | None = None
    sfis_url: str | None = None


class ModelSchema(BaseModel):
    """Model information schema."""

    id: int
    name: str
    site_id: int


class DeviceInfoSchema(BaseModel):
    """Device identifier assigned to a station."""

    id: int | None = None
    device_name: str | None = None
    line: str | None = None
    status: int | None = None

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalise_fields(cls, values: dict):
        if not isinstance(values, dict):
            return values
        data = dict(values)
        if data.get("device_name") in (None, "", 0):
            for candidate in ("name", "device", "device_id__name"):
                candidate_value = data.get(candidate)
                if candidate_value:
                    data["device_name"] = candidate_value
                    break
        if data.get("id") is None and data.get("device_id") is not None:
            data["id"] = data.get("device_id")
        return data

    @model_validator(mode="after")
    def _ensure_device_name(self):
        if self.device_name is None and self.id is not None:
            self.device_name = str(self.id)
        return self


class DevicePeriodEntrySchema(DeviceInfoSchema):
    """Device listing entry returned by the period endpoint."""

    model_config = ConfigDict(extra="allow")


class TestResultQuerySchema(BaseModel):
    """Payload item accepted by the device test result endpoint."""

    start_time: datetime
    end_time: datetime
    device_id: str
    test_result: str = Field(default="ALL")
    model: str | None = None
    model_id: str | None = None
    site_id: str | None = None

    model_config = ConfigDict(extra="allow")


class TestResultRecordSchema(BaseModel):
    """Individual record returned from the device test results endpoint."""

    id: int
    device: str | None = None
    device_id__name: str | None = None
    dut_id__isn: str | None = None
    dut_id: int | None = None
    test_date: datetime | None = None
    test_result: int | None = None
    station_id__name: str | None = None
    test_duration: float | None = None
    error_item: str | None = None
    model: str | None = None
    device_id__station_id__model_id__site_id__name: str | None = None
    site_name: str | None = None
    status: str | None = None

    model_config = ConfigDict(extra="allow")


class TestItemSchema(BaseModel):
    """Station test item meta-data."""

    id: int
    name: str
    upperlimit: float | None = None
    lowerlimit: float | None = None
    status: int | None = None

    model_config = ConfigDict(extra="ignore")


class _StationMetadataSchema(BaseModel):
    station_id: int | None = None
    station_name: str | None = None
    site_id: int | None = None
    site_name: str | None = None
    model_id: int | None = None
    model_name: str | None = None


class StationDeviceListSchema(_StationMetadataSchema):
    data: list[DeviceInfoSchema] = Field(default_factory=list)


class StationDevicePeriodListSchema(_StationMetadataSchema):
    data: list[DevicePeriodEntrySchema] = Field(default_factory=list)


class StationTestItemListSchema(_StationMetadataSchema):
    data: list[TestItemSchema] = Field(default_factory=list)


class BatchTestItemsRequestSchema(BaseModel):
    """Request to fetch test items for multiple stations at once."""

    station_identifiers: list[str] = Field(..., description="List of station identifiers (IDs or names)")
    site_identifier: str | None = Field(default=None, description="Optional site identifier to scope lookups")
    model_identifier: str | None = Field(default=None, description="Optional model identifier to scope lookups")
    status: str | None = Field(default=None, description="Optional status filter (ALL, Active, Online)")


class BatchTestItemsResponseSchema(BaseModel):
    """Response containing test items for multiple stations."""

    stations: list[StationTestItemListSchema] = Field(default_factory=list)


class BatchDevicesRequestSchema(BaseModel):
    """Request to fetch devices for multiple stations at once."""

    station_identifiers: list[str] = Field(..., description="List of station identifiers (IDs or names)")
    site_identifier: str | None = Field(default=None, description="Optional site identifier to scope lookups")
    model_identifier: str | None = Field(default=None, description="Optional model identifier to scope lookups")
    status: str | None = Field(default=None, description="Optional status filter (ALL, Active, Online)")


class BatchDevicesResponseSchema(BaseModel):
    """Response containing devices for multiple stations."""

    stations: list[StationDeviceListSchema] = Field(default_factory=list)


class StationSchema(BaseModel):
    """Station information schema."""

    id: int
    name: str
    status: int
    order: int
    model_id: int
    model_name: str | None = None
    site_id: int | None = None
    site_name: str | None = None
    svn_name: str | None = None
    svn_url: str | None = None


class TestDataSchema(BaseModel):
    """Individual test data schema."""

    id: int
    test_date: datetime
    test_duration: float
    test_result: int
    error_item: str
    device_id: int
    device_id__name: str
    device_id__station_id: int
    device_id__station_id__model_id__site_id__name: str
    dut_id: int
    dut_id__isn: str
    site_name: str


class DUTRecordSchema(BaseModel):
    """DUT record schema."""

    site_name: str
    model_name: str
    record_data: list[dict] = Field(default_factory=list)


class CompleteDUTInfoSchema(BaseModel):
    """Complete DUT information schema."""

    dut_id: str
    dut_records: DUTRecordSchema | None = None
    sites: list[SiteSchema] = Field(default_factory=list)
    models: list[ModelSchema] = Field(default_factory=list)
    stations: list[StationSchema] = Field(default_factory=list)


class SVNInfoSchema(BaseModel):
    """SVN information schema."""

    id: int
    category: str
    customer: str
    product: str
    model: str
    description: str | None = None
    svn_path: str
    svn_name: str | None = None
    svn_url: str | None = None
    created_at: datetime | None = None


class DeviceTestResultSchema(BaseModel):
    """Detailed result for a single device run."""

    test_date: datetime | None = None
    test_result: int | None = None
    status: str
    error_item: str | None = None
    test_duration: float | None = None


class DeviceSummarySchema(BaseModel):
    """Aggregated results for a device at a station."""

    device_id: int | None = None
    device_name: str | None = None
    total_runs: int = 0
    pass_runs: int = 0
    fail_runs: int = 0
    results: list[DeviceTestResultSchema] = Field(default_factory=list)


class StationTestSummarySchema(BaseModel):
    """Summary of test runs per station."""

    station_id: int | None = None
    station_name: str
    svn_name: str | None = None
    svn_url: str | None = None
    dut_id: int | None = None
    dut_isn: str | None = None
    test_runs: int = 0
    devices: list[DeviceSummarySchema] = Field(default_factory=list)


class DUTTestSummarySchema(BaseModel):
    """Aggregated DUT information derived from test records."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    station_count: int = 0
    stations: list[StationTestSummarySchema] = Field(default_factory=list)


class StationRecordRowSchema(BaseModel):
    """Individual record row returned by station-records endpoint."""

    test_date: datetime | None = None
    device: str | None = None
    station: str | None = None
    isn: str | None = None
    error_item: str | None = None
    site_name: str | None = None


class StationRecordResponseSchema(BaseModel):
    """Response payload for station record details."""

    record: list[StationRecordRowSchema] = Field(default_factory=list)
    data: list[list[str | int | float | None]] = Field(default_factory=list)


class PASROMDataItemSchema(BaseModel):
    """PA SROM test item with hex measurement and decimal value."""

    test_item: str = Field(..., description="Test item name (e.g., WiFi_PA1_SROM_OLD_6015_11AX_MCS9_B20)")
    measurement: str = Field(..., description="Hex measurement value (e.g., 0x2c89)")
    decimal_value: int = Field(..., description="Decimal conversion of hex measurement (e.g., 11190)")


class PAAdjustedPowerDataItemSchema(BaseModel):
    """PA Adjusted Power data item calculated from SROM_OLD and SROM_NEW pairs."""

    test_item: str = Field(..., description="Test item name (e.g., WiFi_PA1_ADJUSTED_POW_6015_11AX_MCS9_B20)")
    adjusted_value: float = Field(..., description="Adjusted power value: (SROM_NEW - SROM_OLD) / 256")


class GenericTestItemSchema(BaseModel):
    """Generic test item (for non-PA SROM items)."""

    test_item: str = Field(..., description="Test item name")
    value: str = Field(..., description="Test value (raw string)")


class PASROMEnhancedResponseSchema(BaseModel):
    """Enhanced response for PA SROM endpoint with sorted test items and calculated adjusted power."""

    record: list[StationRecordRowSchema] = Field(default_factory=list, description="Test record metadata (test_date, device, station, etc.)")
    data: list[PASROMDataItemSchema | PAAdjustedPowerDataItemSchema | GenericTestItemSchema] = Field(default_factory=list, description="Sorted PA SROM measurements and adjusted power values, or all test items if no PA SROM found")
    has_pa_srom: bool = Field(default=True, description="Whether the DUT has PA SROM test items")


class PAAdjustedPowerScoredItemSchema(BaseModel):
    """PA Adjusted Power item with score calculated from trend comparison."""

    test_item: str = Field(..., description="Test item name (e.g., WiFi_PA1_ADJUSTED_POW_6015_11AX_MCS9_B20)")
    current_value: float = Field(..., description="Current adjusted power value from latest test")
    trend_mean: float | None = Field(None, description="Trend mean value (24h average)")
    trend_mid: float | None = Field(None, description="Trend mid value (24h median)")
    deviation_from_mean: float | None = Field(None, description="Deviation from trend mean")
    deviation_from_mid: float | None = Field(None, description="Deviation from trend mid")
    score: float | None = Field(None, description="Score (0-10) based on deviation from trend mean")
    score_breakdown: dict = Field(default_factory=dict, description="Detailed scoring calculation")


class PAAdjustedPowerScoringResponseSchema(BaseModel):
    """Response for PA Adjusted Power scoring endpoint."""

    record: list[StationRecordRowSchema] = Field(default_factory=list, description="Test record metadata")
    data: list[PAAdjustedPowerScoredItemSchema] = Field(default_factory=list, description="Scored PA adjusted power items")
    trend_time_window: dict = Field(default_factory=dict, description="Time window used for trend calculation")
    avg_score: float | None = Field(None, description="Average score across all items")
    median_score: float | None = Field(None, description="Median score across all items")


class StationLatestRecordSchema(BaseModel):
    """Latest station record including the most recent run data."""

    station_id: int | None = None
    station_name: str
    test_date: datetime | None = None
    device_id: int | None = None
    device: str | None = None
    status: int | None = None
    order: int | None = None
    isn: str | None = None
    dut_id: int | None = None
    test_duration: float | None = None
    error_item: str | None = None
    data: list[list[str | int | float | None]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class LatestTestsResponseSchema(BaseModel):
    """Response payload for latest test data per station."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    station_count: int = 0
    record_data: list[StationLatestRecordSchema] = Field(default_factory=list)


class TopProductMeasurementSchema(BaseModel):
    """Measurement details used to rank top products."""

    test_item: str
    usl: float | None = None
    lsl: float | None = None
    target_value: float | None = None
    actual_value: float | None = None
    deviation: float | None = None


class TopProductStationResultSchema(StationLatestRecordSchema):
    """Station evaluation result for a top product."""

    measurements: list[TopProductMeasurementSchema] = Field(default_factory=list)
    score: float


class TopProductCandidateSchema(BaseModel):
    """Aggregated candidate score across selected stations."""

    dut_isn: str
    dut_id: int | None = None
    score: float
    stations: list[TopProductStationResultSchema] = Field(default_factory=list)


class TopProductAggregateResponseSchema(BaseModel):
    """Top product response across one or more stations."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    station_count: int = 0
    criteria_path: str | None = None
    is_requested_best: bool | None = None
    requested_product: TopProductCandidateSchema | None = None
    best_products: list[TopProductCandidateSchema] = Field(default_factory=list)


class StationTopProductDetailSchema(BaseModel):
    """Detailed entry for station-scoped top product ranking."""

    test_date: datetime | None = None
    station_id: int | None = None
    station_name: str
    station_status: int | None = None
    station_order: int | None = None
    device_id: int | None = None
    device: str | None = None
    device_status: int | None = None
    dut_id: int | None = None
    isn: str | None = None
    test_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    test_duration: float | None = None
    error_item: str | None = None
    latest_data: list[list[str | float | None] | dict[str, Any]] = Field(default_factory=list)
    overall_data_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class StationTopProductsResponseSchema(BaseModel):
    """Response contract for station-scoped top product lookup."""

    site_name: str
    model_name: str
    start_time: datetime
    end_time: datetime
    criteria_score: str
    requested_data: list[StationTopProductDetailSchema] = Field(default_factory=list)


class TopProductStationSummarySchema(BaseModel):
    """Flattened station evaluation summary for top product response."""

    station_id: int | None = None
    station_name: str
    dut_id: int | None = None
    isn: str
    device_id: int | None = None
    device: str | None = None
    test_date: datetime | None = None
    status: int | None = None
    order: int | None = None
    test_duration: float | None = None
    test_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    error_item: str | None = None
    data: list[dict[str, str | float | None | dict]] = Field(
        default_factory=list,
        description=(
            "Measurement data as list of objects with fields: test_item, usl, lsl, actual, score_breakdown. "
            "The score_breakdown is a dict containing: category, method, target_used, usl, lsl, "
            "actual, deviation, score, and formula_latex (LaTeX formula string for score calculation)."
        ),
    )
    overall_data_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    group_scores: dict[str, Any] | None = None
    overall_group_scores: dict[str, float] | None = None

    model_config = ConfigDict(extra="allow")


class TopProductResponseSchema(BaseModel):
    """Response payload for the top-product endpoint."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    criteria_path: str | None = None
    test_result: list[TopProductStationSummarySchema] = Field(default_factory=list)


class TopProductErrorSchema(BaseModel):
    """Error entry produced when a DUT ISN fails evaluation."""

    dut_isn: str
    detail: str


class TopProductBatchResponseSchema(BaseModel):
    """Batch response containing top-product evaluations for one or more DUTs."""

    results: list[TopProductResponseSchema] = Field(default_factory=list)
    errors: list[TopProductErrorSchema] = Field(default_factory=list)


class DUTIdentifierSchema(BaseModel):
    """Single DUT identifier entry found within records."""

    dut_id: int | None = None
    station_id: int | None = None
    station_name: str | None = None
    dut_isn: str | None = None


class DUTIdentifierListSchema(BaseModel):
    """List of DUT identifiers discovered for an ISN."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    identifiers: list[DUTIdentifierSchema] = Field(default_factory=list)


class DUTISNVariantListSchema(BaseModel):
    """Unique DUT ISN variants discovered for an identifier."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    isns: list[str] = Field(default_factory=list)


class StationProgressSchema(BaseModel):
    """Progress of a DUT at a particular station."""

    station_id: int | None = None
    station_name: str
    status: int | None = None
    tested: bool = False
    test_runs: int = 0


class DUTProgressSchema(BaseModel):
    """Progress report across stations for a DUT."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    stations: list[StationProgressSchema] = Field(default_factory=list)


class StationRunSummarySchema(BaseModel):
    """Summary of run outcomes per station."""

    station_id: int | None = None
    station_name: str
    test_runs: int = 0
    pass_runs: int = 0
    fail_runs: int = 0
    results: list[DeviceTestResultSchema] = Field(default_factory=list)


class DUTRunSummarySchema(BaseModel):
    """Overall run outcome summary for a DUT."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    stations: list[StationRunSummarySchema] = Field(default_factory=list)


class StationFilterConfigSchema(BaseModel):
    """Per-station filter configuration for top product analysis."""

    station_identifier: str = Field(..., description="Station ID or name to apply these filters to")
    device_identifiers: list[str] | None = Field(default=None, description="Optional list of device identifiers (IDs or names) specific to this station")
    test_item_filters: list[str] | None = Field(default=None, description="Optional list of regex patterns to include test items for this station")
    exclude_test_item_filters: list[str] | None = Field(default=None, description="Optional list of regex patterns to exclude test items for this station")


class ErrorItemSummarySchema(BaseModel):
    """Error item frequency representation used in model summaries."""

    error_item: str
    fail_count: int | None = None


class RetestDUTSchema(BaseModel):
    """Detailed retest DUT entry containing identifier information."""

    dut_id: int | None = None
    dut_id__isn: str | None = None


class StationDeviceAggregateSchema(BaseModel):
    """Aggregated metrics for a device within a station summary."""

    device: str | None = None
    device_id: int | None = None
    pass_count: int | None = None
    fail_count: int | None = None
    line: str | None = None
    error_items: list[ErrorItemSummarySchema] = Field(default_factory=list)
    retest_dut: list[str] = Field(default_factory=list)
    retest_dut2: list[RetestDUTSchema] = Field(default_factory=list)
    retest_ratio: float | None = None
    retest_count: int | None = None
    retest_count_unique: int | None = None

    model_config = ConfigDict(extra="allow")


class ModelStationSummarySchema(BaseModel):
    """Per-station metrics returned by the model summary endpoint."""

    station_id: int | None = None
    station_order: int | None = None
    station_name: str | None = None
    test_times: int | None = None
    pass_count: int | None = None
    fail_count: int | None = None
    retest_count: int | None = None
    yield_rate: float | None = None
    retest_rate: float | None = None
    device_summary: list[StationDeviceAggregateSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ModelSummaryRequestSchema(BaseModel):
    """Request payload accepted by the model summary endpoint."""

    model_id: int
    start_time: datetime
    end_time: datetime
    model: str | None = None

    model_config = ConfigDict(extra="allow")


class ModelSummarySchema(BaseModel):
    """Overall summary returned from the model summary endpoint."""

    model_id: int | None = None
    model_name: str | None = None
    site_name: str | None = None
    total_yield_rate: float | None = None
    total_retest_rate: float | None = None
    total_UPH: float | None = None
    station_info: list[ModelStationSummarySchema] = Field(default_factory=list)
    model_type_list: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class BatchLatestTestItemsRequestSchema(BaseModel):
    """Request to fetch latest test items for multiple stations based on a DUT ISN."""

    dut_isn: str = Field(..., description="DUT ISN to get latest test items for")
    station_identifiers: list[str] = Field(..., description="List of station identifiers (IDs or names)")
    site_identifier: str | None = Field(default=None, description="Optional site identifier to scope lookups")
    model_identifier: str | None = Field(default=None, description="Optional model identifier to scope lookups")


class LatestTestItemsRequestSchema(BaseModel):
    """Request payload to fetch latest test items for a DUT ISN (auto-deriving station list when omitted)."""

    dut_isn: str = Field(..., description="DUT ISN to collect test items for")
    station_identifiers: list[str] | None = Field(default=None, description="Optional station identifiers to restrict the response")
    site_identifier: str | None = Field(default=None, description="Optional site identifier to scope lookups")
    model_identifier: str | None = Field(default=None, description="Optional model identifier to scope lookups")


class TestItemDefinitionSchema(BaseModel):
    """Test item metadata extracted from latest station records."""

    name: str
    usl: float | None = Field(default=None, description="Upper specification limit when available")
    lsl: float | None = Field(default=None, description="Lower specification limit when available")


class StationLatestTestItemsSchema(BaseModel):
    """Test items from latest run for a specific station and DUT."""

    station_id: int
    station_name: str
    station_dut_id: int | None = Field(default=None, description="Resolved DUT ID for this station")
    station_dut_isn: str | None = Field(default=None, description="Resolved DUT ISN for this station")
    value_test_items: list[TestItemDefinitionSchema] = Field(default_factory=list, description="Value-based test items (with USL/LSL metadata)")
    nonvalue_test_items: list[TestItemDefinitionSchema] = Field(default_factory=list, description="Non-value test items (hex/config rows)")
    nonvalue_bin_test_items: list[TestItemDefinitionSchema] = Field(default_factory=list, description="Non-value BIN test items (e.g., PASS/FAIL logs)")
    error: str | None = Field(default=None, description="Error message if test items could not be retrieved")


class BatchLatestTestItemsResponseSchema(BaseModel):
    """Response containing latest test items for multiple stations."""

    dut_isn: str
    site_name: str | None = None
    model_name: str | None = None
    stations: list[StationLatestTestItemsSchema] = Field(default_factory=list)


# ==================== PA Test Items Trend Schemas ====================


class PATrendRequestSchema(BaseModel):
    """Request schema for PA test items trend data from external DUT API.

    Calls POST /api/api/testitems/PA/trend to get mean and mid values
    for PA SROM OLD and NEW test items.
    """

    start_time: datetime = Field(..., description="Start of date range (max 7 days from end_time)")
    end_time: datetime = Field(..., description="End of date range")
    station_id: int = Field(..., description="Station ID to query")
    test_items: list[str] = Field(..., description="List of PA test item names (e.g., WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80)")
    model: str | None = Field(default="", description="Model filter (empty string or 'ALL')")

    model_config = ConfigDict(json_schema_extra={"example": {"start_time": "2025-11-15T08:22:21.00Z", "end_time": "2025-11-17T08:22:21.00Z", "model": "", "station_id": 145, "test_items": ["WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80"]}})


class PATrendItemSchema(BaseModel):
    """Trend data (mean/mid values) for a single PA test item."""

    mid: float | None = Field(default=None, description="Median value from external API")
    mean: float | None = Field(default=None, description="Mean/average value from external API")


class PATrendResponseSchema(RootModel[dict[str, PATrendItemSchema]]):
    """Response from external DUT API containing mean/mid values per test item.

    Keys are test item names, values contain mid and mean statistics.
    """

    root: dict[str, PATrendItemSchema]

    model_config = ConfigDict(json_schema_extra={"example": {"WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80": {"mid": 11219.0, "mean": 11227}, "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80": {"mid": 11313.0, "mean": 11308}}})


class AdjustedPowerValueSchema(BaseModel):
    """Adjusted power values calculated from SROM_OLD and SROM_NEW pairs."""

    adjusted_mid: float | None = Field(default=None, description="Adjusted median power: (NEW_mid - OLD_mid) / 256, rounded to 2 decimals")
    adjusted_mean: float | None = Field(default=None, description="Adjusted mean power: (NEW_mean - OLD_mean) / 256, rounded to 2 decimals")
    raw_mid_difference: float | None = Field(default=None, description="Raw difference: NEW_mid - OLD_mid")
    raw_mean_difference: float | None = Field(default=None, description="Raw difference: NEW_mean - OLD_mean")


class AdjustedPowerItemSchema(BaseModel):
    """Adjusted power calculation result for a single test item base name."""

    test_item_base_name: str = Field(..., description="Base test item name without SROM_OLD/NEW suffix (e.g., WiFi_PA1_5985_11AX_MCS9_B80)")
    old_item_name: str | None = Field(default=None, description="Full SROM_OLD test item name")
    new_item_name: str | None = Field(default=None, description="Full SROM_NEW test item name")
    old_values: PATrendItemSchema | None = Field(default=None, description="Original OLD trend values")
    new_values: PATrendItemSchema | None = Field(default=None, description="Original NEW trend values")
    adjusted_power: AdjustedPowerValueSchema = Field(..., description="Calculated adjusted power values")
    error: str | None = Field(default=None, description="Error message if calculation failed (e.g., missing pair)")


class AdjustedPowerRequestSchema(BaseModel):
    """Request schema for calculating adjusted power from PA trend data."""

    start_time: datetime = Field(..., description="Start of date range (max 7 days from end_time)")
    end_time: datetime = Field(..., description="End of date range")
    station_id: int = Field(..., description="Station ID to query")
    test_items: list[str] = Field(..., description="List of PA test item names (must include both OLD and NEW variants)")
    model: str | None = Field(default="", description="Model filter (empty string or 'ALL')")

    model_config = ConfigDict(json_schema_extra={"example": {"start_time": "2025-11-15T08:22:21.00Z", "end_time": "2025-11-17T08:22:21.00Z", "station_id": 145, "test_items": ["WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80"], "model": ""}})


class PATrendDataItemSchema(BaseModel):
    """PA trend data item with OLD/NEW SROM values and adjusted power."""

    model_config = ConfigDict(extra="allow")  # Allow dynamic SROM field names as keys


class AdjustedPowerResponseSchema(BaseModel):
    """Response containing adjusted power calculations for all matched PA test item pairs."""

    station_id: int
    start_time: datetime
    end_time: datetime
    data: list[dict[str, Any]] = Field(default_factory=list, description="PA trend data with OLD/NEW SROM values and adjusted power calculations")
    unpaired_items: list[str] = Field(default_factory=list, description="Test items that could not be paired (missing OLD or NEW variant)")


class PAStationTestItemSchema(BaseModel):
    """PA SROM test item with its latest value from station records."""

    test_item_name: str = Field(..., description="Full PA SROM test item name (e.g., WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80)")
    latest_value: str | None = Field(default=None, description="Latest test value (SROM hex value)")


class PAStationItemsSchema(BaseModel):
    """PA test items for a specific station."""

    station_id: int = Field(..., description="Resolved station ID")
    station_name: str = Field(..., description="Station name")
    test_date: datetime | None = Field(default=None, description="Test date of the latest record")
    device: str | None = Field(default=None, description="Device name/ID")
    isn: str | None = Field(default=None, description="DUT ISN from the test record")


class PAAdjustedPowerTrendItemSchema(BaseModel):
    """PA adjusted power trend item with test pattern and calculated values."""

    test_pattern: str = Field(..., description="Base test pattern name (e.g., WiFi_PA1_5985_11AX_MCS9_B80)")
    adjusted_power_test_items: dict[str, float] = Field(default_factory=dict, description="Adjusted power values with full test item names as keys (e.g., WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80: 0.37)")


class CombinedPAAdjustedPowerSchema(BaseModel):
    """Combined response with PA SROM test items and adjusted power calculations."""

    dut_isn: str = Field(..., description="DUT ISN identifier")
    site_name: str | None = Field(default=None, description="Site name")
    model_name: str | None = Field(default=None, description="Model name")
    stations: list[PAStationItemsSchema] = Field(default_factory=list, description="PA test items per station")
    adjusted_power_trend: list[PAAdjustedPowerTrendItemSchema] = Field(default_factory=list, description="PA trend data and calculated adjusted power values")
    time_window_start: datetime | None = Field(default=None, description="Start of the time window used for PA trend calculation")
    time_window_end: datetime | None = Field(default=None, description="End of the time window used for PA trend calculation")
    unpaired_items: list[str] = Field(default_factory=list, description="Test items without matching OLD/NEW pairs")


class PATrendStationItemSchema(BaseModel):
    """PA SROM trend values for a test item in a specific station."""

    test_item_name: str = Field(..., description="Full PA SROM test item name (e.g., WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80)")
    mid: float | None = Field(default=None, description="Mid value from trend calculation")
    mean: float | None = Field(default=None, description="Mean value from trend calculation")


class PATrendStationDataSchema(BaseModel):
    """PA trend data for a specific station."""

    isn: str | None = Field(default=None, description="DUT ISN from the test record")
    station_id: int = Field(..., description="Resolved station ID")
    station_name: str = Field(..., description="Station name")
    device: str | None = Field(default=None, description="Device name/ID")
    test_date: datetime | None = Field(default=None, description="Test date of the latest record")
    trend_items: list[PATrendStationItemSchema] = Field(default_factory=list, description="PA SROM trend values (mid/mean) for each test item")


class SimplePATrendResponseSchema(BaseModel):
    """Simplified response with PA SROM trend values (mid/mean) per station."""

    dut_isn: str = Field(..., description="DUT ISN identifier")
    site_name: str | None = Field(default=None, description="Site name")
    model_name: str | None = Field(default=None, description="Model name")
    stations: list[PATrendStationDataSchema] = Field(default_factory=list, description="PA trend data per station")
    time_window_start: str = Field(..., description="Start of the time window used for PA trend calculation")
    time_window_end: str = Field(..., description="End of the time window used for PA trend calculation")


class PADiffStationDataSchema(BaseModel):
    """PA diff data (SROM_NEW - SROM_OLD) for a specific station."""

    isn: str | None = Field(default=None, description="DUT ISN from the test record")
    station_id: int = Field(..., description="Resolved station ID")
    station_name: str = Field(..., description="Station name")
    device: str | None = Field(default=None, description="Device name/ID")
    test_date: datetime | None = Field(default=None, description="Test date of the latest record")
    trend_diff_items: list[PATrendStationItemSchema] = Field(default_factory=list, description="PA SROM difference values (mid/mean) for each test item")


class PADiffResponseSchema(BaseModel):
    """Response with PA SROM difference values (SROM_NEW - SROM_OLD) per station."""

    dut_isn: str = Field(..., description="DUT ISN identifier")
    site_name: str | None = Field(default=None, description="Site name")
    model_name: str | None = Field(default=None, description="Model name")
    stations: list[PADiffStationDataSchema] = Field(default_factory=list, description="PA diff data per station")
    time_window_start: str = Field(..., description="Start of the time window used for PA trend calculation")
    time_window_end: str = Field(..., description="End of the time window used for PA trend calculation")


class PAActualAdjustedStationDataSchema(BaseModel):
    """PA actual adjusted data (SROM_NEW trend - SROM_NEW latest) for a specific station."""

    isn: str | None = Field(default=None, description="DUT ISN from the test record")
    station_id: int = Field(..., description="Resolved station ID")
    station_name: str = Field(..., description="Station name")
    device: str | None = Field(default=None, description="Device name/ID")
    test_date: datetime | None = Field(default=None, description="Test date of the latest record")
    data: list[list[str | float]] = Field(default_factory=list, description="PA actual adjusted values [[test_item_name, adjusted_value], ...]")


class PAActualAdjustedResponseSchema(BaseModel):
    """Response with PA actual adjusted values (SROM_NEW trend - SROM_NEW latest) per station."""

    dut_isn: str = Field(..., description="DUT ISN identifier")
    site_name: str | None = Field(default=None, description="Site name")
    model_name: str | None = Field(default=None, description="Model name")
    stations: list[PAActualAdjustedStationDataSchema] = Field(default_factory=list, description="PA actual adjusted data per station")
    time_window_start: str = Field(..., description="Start of the time window used for PA trend calculation")
    time_window_end: str = Field(..., description="End of the time window used for PA trend calculation")


class PAItemValueSchema(BaseModel):
    """PA item with test name and value."""

    test_item_name: str = Field(..., description="Test item name")
    value: float = Field(..., description="Test item value")


class PAStationDataSchema(BaseModel):
    """Station-level data with ISN identifier."""

    isn: str = Field(..., description="ISN identifier")
    station_name: str = Field(..., description="Station name")
    station_id: int = Field(..., description="Station ID")
    device: str | None = Field(default=None, description="Device name/ID")
    test_date: datetime | None = Field(default=None, description="Test date")
    site_name: str | None = Field(default=None, description="Site name")
    model_name: str | None = Field(default=None, description="Model name")
    data: list[PAItemValueSchema] = Field(default_factory=list, description="PA test items")


class PAActualAdjustedStationDataSchema(BaseModel):
    """Station-level data for PA actual adjusted power (includes ISN and time window)."""

    isn: str = Field(..., description="ISN identifier")
    station_name: str = Field(..., description="Station name")
    station_id: int = Field(..., description="Station ID")
    device: str | None = Field(default=None, description="Device name/ID")
    test_date: datetime | None = Field(default=None, description="Test date")
    site_name: str | None = Field(default=None, description="Site name")
    model_name: str | None = Field(default=None, description="Model name")
    time_window_start: str = Field(..., description="Start of time window")
    time_window_end: str = Field(..., description="End of time window")
    data: list[PAItemValueSchema] = Field(default_factory=list, description="PA actual adjusted items")


class PAISNGroupSchema(BaseModel):
    """Group of stations by ISN identifier."""

    isn: str = Field(..., description="ISN identifier for this group")
    stations: list[PAStationDataSchema] = Field(default_factory=list, description="Stations for this ISN")


class PAActualAdjustedISNGroupSchema(BaseModel):
    """Group of stations by ISN for actual adjusted power endpoint."""

    isn: str = Field(..., description="ISN identifier for this group")
    stations: list[PAActualAdjustedStationDataSchema] = Field(default_factory=list, description="Stations for this ISN")


class TestLogDownloadInfo(BaseModel):
    """Single test log download info"""
    isn: str = Field(..., description="DUT ISN identifier")
    time: str = Field(..., description="Test time in format: YYYY/MM/DD HH:MM:SS")
    deviceid: str = Field(..., description="Device ID name")
    station: str = Field(..., description="Station name")


class TestLogDownloadRequest(BaseModel):
    """Request to download test log(s) from External2 API"""
    info_list: list[TestLogDownloadInfo] = Field(..., description="List of test log info to download")
    site: str = Field(..., description="Site name (e.g., PTB)")
    project: str = Field(..., description="Project/Model name (e.g., HH5K)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "info_list": [
                    {
                        "isn": "DM2527470036123",
                        "time": "2025/11/17 14:20:28",
                        "deviceid": "614640",
                        "station": "Wireless_Test_2_5G"
                    }
                ],
                "site": "PTB",
                "project": "HH5K"
            }
        }
    )
