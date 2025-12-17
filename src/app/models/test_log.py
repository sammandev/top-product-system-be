"""
Pydantic models for test log parsing and comparison (optimized for batch processing).
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ParsedTestItem(BaseModel):
    """A single parsed test item from a test log file."""

    test_item: str = Field(..., description="Name of the test item")
    usl: str | None = Field(None, description="Upper Spec Limit (USL)")
    lsl: str | None = Field(None, description="Lower Spec Limit (LSL)")
    value: str = Field(..., description="Actual measured value")


class TestLogParseResponse(BaseModel):
    """Response model for test log parsing (optimized)."""

    filename: str = Field(..., description="Name of the parsed file")
    isn: str | None = Field(None, description="DUT ISN extracted from filename")
    parsed_count: int = Field(..., description="Number of successfully parsed items")
    parsed_items: list[ParsedTestItem] = Field(..., description="List of parsed test items")


class TestItemValue(BaseModel):
    """Value of a test item from a specific file (with ISN)."""

    isn: str | None = Field(None, description="DUT ISN for this measurement")
    value: str = Field(..., description="Value from this file")


class ComparisonItem(BaseModel):
    """Comparison data for a single test item across multiple files (optimized)."""

    test_item: str = Field(..., description="Name of the test item")
    usl: str | None = Field(None, description="Upper Spec Limit")
    lsl: str | None = Field(None, description="Lower Spec Limit")
    is_common: bool = Field(..., description="Whether item appears in all files")
    values: list[TestItemValue] = Field(..., description="Values from each file")

    # Optional numeric analysis fields (only if multiple numeric values)
    min: float | None = Field(None, description="Minimum value")
    max: float | None = Field(None, description="Maximum value")
    range: float | None = Field(None, description="Range of values")
    avg: float | None = Field(None, description="Average value")


class FileSummary(BaseModel):
    """Summary of a single file in comparison (optimized)."""

    filename: str = Field(..., description="Name of the file")
    isn: str | None = Field(None, description="DUT ISN extracted from filename")
    parsed_count: int = Field(..., description="Number of parsed items")


class CompareResponse(BaseModel):
    """Response model for test log comparison (optimized for batch processing)."""

    total_files: int = Field(..., description="Number of files compared")
    total_items: int = Field(..., description="Total unique test items")
    common_items: int = Field(..., description="Items present in all files")
    file_summary: list[FileSummary] = Field(..., description="Summary for each file")
    comparison: list[ComparisonItem] = Field(..., description="Detailed comparison data")


# ============================================================================
# Enhanced schemas for BY UPLOAD LOG feature
# ============================================================================


class TestLogMetadata(BaseModel):
    """Metadata extracted from test log header."""

    test_date: datetime | None = Field(None, description="Test execution date and time")
    device: str | None = Field(None, description="Device ID from log header")
    station: str | None = Field(None, description="Station name from log header")
    script_version: str | None = Field(None, description="Script version")
    duration_seconds: int | None = Field(None, description="Test duration in seconds")
    sfis_status: str | None = Field(None, description="SFIS status (e.g., On-Line)")
    result: str | None = Field(None, description="Test result (PASS/FAIL)")
    counter: int | None = Field(None, description="Test counter")


class ScoreBreakdown(BaseModel):
    """Detailed score calculation breakdown."""

    category: str = Field(..., description="Measurement category (EVM, Frequency, PER, etc.)")
    method: str = Field(..., description="Scoring method used")
    usl: float | None = Field(None, description="Upper Spec Limit used in calculation")
    lsl: float | None = Field(None, description="Lower Spec Limit used in calculation")
    target_used: float | None = Field(None, description="Target value used in calculation")
    actual: float = Field(..., description="Actual measured value")
    deviation: float = Field(..., description="Deviation from target")
    raw_score: float = Field(..., description="Raw score before scaling")
    final_score: float = Field(..., description="Final score (0-10 scale)")
    formula_latex: str = Field(..., description="LaTeX formula used for calculation")


class ParsedTestItemEnhanced(BaseModel):
    """Enhanced parsed test item with classification and scoring."""

    test_item: str = Field(..., description="Name of the test item")
    usl: float | None = Field(None, description="Upper Spec Limit")
    lsl: float | None = Field(None, description="Lower Spec Limit")
    target: float | None = Field(None, description="Target value from criteria")
    value: str = Field(..., description="Raw value string")
    numeric_value: float | None = Field(None, description="Parsed numeric value")
    is_value_type: bool = Field(..., description="Whether item is numeric/value type")
    is_hex: bool = Field(..., description="Whether value is hexadecimal")
    hex_decimal: int | None = Field(None, description="Decimal conversion of hex value")
    score: float | None = Field(None, description="Score (0-10) for value items")
    score_breakdown: ScoreBreakdown | None = Field(None, description="Detailed score calculation")
    matched_criteria: bool = Field(..., description="Whether item matched criteria pattern")
    is_calculated: bool = Field(default=False, description="Whether item was auto-calculated (e.g., PA ADJUSTED_POW)")


class TestLogParseResponseEnhanced(BaseModel):
    """Enhanced response for test log parsing with metadata and classification."""

    filename: str = Field(..., description="Name of the parsed file")
    isn: str | None = Field(None, description="DUT ISN extracted from filename")
    station: str = Field(..., description="Station name (from filename pattern)")
    metadata: TestLogMetadata = Field(..., description="Metadata from log header")
    parsed_count: int = Field(..., description="Total items parsed")
    parsed_items_enhanced: list[ParsedTestItemEnhanced] = Field(..., description="Enhanced parsed items with classification and scoring")
    value_type_count: int = Field(..., description="Count of value-type items")
    non_value_type_count: int = Field(..., description="Count of non-value items")
    hex_value_count: int = Field(..., description="Count of hex value items")
    avg_score: float | None = Field(None, description="Average score across value items")
    median_score: float | None = Field(None, description="Median score across value items")


class PerIsnData(BaseModel):
    """Per-ISN data for comparison items."""

    isn: str | None = Field(None, description="DUT ISN")
    value: str = Field(..., description="Raw value string")
    is_value_type: bool = Field(..., description="Whether value is numeric")
    numeric_value: float | None = Field(None, description="Parsed numeric value")
    is_hex: bool = Field(..., description="Whether value is hexadecimal")
    hex_decimal: int | None = Field(None, description="Decimal conversion of hex value")
    deviation: float | None = Field(None, description="Deviation from baseline (value items only)")
    is_calculated: bool = Field(default=False, description="Whether item was auto-calculated (e.g., PA ADJUSTED_POW)")
    score: float | None = Field(None, description="Score (value items only)")
    score_breakdown: ScoreBreakdown | None = Field(None, description="Score breakdown (value items only)")


class CompareItemEnhanced(BaseModel):
    """Enhanced comparison item with deviations and scores across ISNs."""

    test_item: str = Field(..., description="Name of the test item")
    usl: float | None = Field(None, description="Upper Spec Limit")
    lsl: float | None = Field(None, description="Lower Spec Limit")
    baseline: float | None = Field(None, description="Baseline value (median or criteria target, for value items)")
    per_isn_data: list[PerIsnData] = Field(..., description="Per-ISN measurements and deviations")
    avg_deviation: float | None = Field(None, description="Average absolute deviation (value items only)")
    avg_score: float | None = Field(None, description="Average score (value items only)")
    median_score: float | None = Field(None, description="Median score (value items only)")
    matched_criteria: bool = Field(..., description="Whether item matches criteria pattern")


class FileSummaryEnhanced(BaseModel):
    """Summary of a single file in comparison with metadata."""

    filename: str = Field(..., description="Name of the file")
    isn: str | None = Field(None, description="DUT ISN extracted from filename")
    metadata: TestLogMetadata = Field(..., description="Metadata from log header")
    parsed_count: int = Field(..., description="Number of parsed items")
    avg_score: float | None = Field(None, description="Average score for this file")


class CompareResponseEnhanced(BaseModel):
    """Enhanced comparison response with value/non-value separation and statistics."""

    total_files: int = Field(..., description="Number of files compared")
    total_value_items: int = Field(..., description="Total value-type test items")
    total_non_value_items: int = Field(..., description="Total non-value-type test items")
    file_summaries: list[FileSummaryEnhanced] = Field(..., description="Summary with metadata for each file")
    comparison_value_items: list[CompareItemEnhanced] = Field(..., description="Value-type comparison items")
    comparison_non_value_items: list[CompareItemEnhanced] = Field(..., description="Non-value comparison items")
