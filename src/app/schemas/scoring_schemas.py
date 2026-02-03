"""
Scoring System Schemas

Pydantic models for the Universal 0-10 Scoring System used in test item evaluation.

Score Range: 0.00 - 10.00 (stored internally as 0.00 - 1.00)
- Outside limits: 0.00
- At limit boundary (LCL/UCL): 1.00 (limit_score)
- At target: 10.00

Scoring Types:
- SYMMETRICAL: Target = midpoint (UCL + LCL) / 2
- ASYMMETRICAL: User-defined custom target with Policy options
- PER_MASK: UCL-only scoring for PER/MASK test items (lower is better, best=0)
- EVM: UCL-only scoring for EVM test items (lower is better, best=-35, gentle decay)
- BINARY: PASS = 10, FAIL = 0

Policy (for ASYMMETRICAL scoring):
- SYMMETRICAL: Peak score at target, decay both directions
- HIGHER: Perfect score at/above target, decay below
- LOWER: Perfect score at/below target, decay above
"""
from enum import Enum

from pydantic import BaseModel, Field


class ScoringType(str, Enum):
    """Available scoring algorithm types."""

    SYMMETRICAL = "symmetrical"       # Target centered between UCL and LCL
    ASYMMETRICAL = "asymmetrical"     # User-defined custom target with Policy
    PER_MASK = "per_mask"             # UCL-only, lower is better (best=0)
    EVM = "evm"                       # UCL-only, lower is better (best=-35, gentle decay)
    BINARY = "binary"                 # PASS/FAIL scoring
    # Legacy types kept for backwards compatibility (mapped internally)
    SYMMETRICAL_NL = "symmetrical_nl"
    THROUGHPUT = "throughput"


class ScoringPolicy(str, Enum):
    """Policy for asymmetrical scoring - determines how score decays from target."""

    SYMMETRICAL = "symmetrical"  # Peak at target, linear decay to both limits
    HIGHER = "higher"            # Perfect at/above target, decay below target to LCL
    LOWER = "lower"              # Perfect at/below target, decay above target to UCL


class ScoringConfig(BaseModel):
    """Configuration for scoring a specific test item."""

    test_item_name: str = Field(..., description="Name of the test item")
    scoring_type: ScoringType = Field(
        default=ScoringType.SYMMETRICAL,
        description="Scoring algorithm to use"
    )
    enabled: bool = Field(default=True, description="Whether scoring is enabled for this item")
    weight: float = Field(default=1.0, ge=0, le=10, description="Weight for aggregate scoring")

    # Type-specific parameters
    target: float | None = Field(
        default=None,
        description="User-defined target value (required for asymmetrical scoring)"
    )
    policy: ScoringPolicy = Field(
        default=ScoringPolicy.SYMMETRICAL,
        description="Scoring policy for asymmetrical (symmetrical/higher/lower)"
    )
    limit_score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Score at limit boundary (default: 1.0 on 0-10 scale)"
    )
    # Legacy parameters kept for backwards compatibility
    alpha: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Legacy: Min score at limit boundary (deprecated, use limit_score)"
    )
    target_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Legacy: Score at target deviation (deprecated)"
    )
    target_deviation: float | None = Field(
        default=None,
        ge=0,
        description="Legacy: Deviation value for target_score (deprecated)"
    )
    min_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Legacy: Minimum score at lower limit (deprecated)"
    )
    max_deviation: float | None = Field(
        default=None,
        ge=0,
        description="Legacy: Maximum deviation for PER/MASK scoring (deprecated)"
    )


class TestItemScoreResult(BaseModel):
    """Score result for a single test item."""

    test_item_name: str
    value: float | None = None
    ucl: float | None = None
    lcl: float | None = None
    status: str
    scoring_type: ScoringType
    policy: ScoringPolicy | None = None  # Only for asymmetrical scoring
    score: float = Field(ge=0, le=1)
    deviation: float | None = None


class RecordScoreResult(BaseModel):
    """Score result for a single test record (ISN/DUT)."""

    isn: str
    device_id: str
    station: str
    test_start_time: str
    test_status: str
    overall_score: float = Field(ge=0, le=1)
    value_items_score: float | None = None
    bin_items_score: float | None = None
    test_item_scores: list[TestItemScoreResult]
    total_items: int
    scored_items: int
    failed_items: int


class ScoreSummary(BaseModel):
    """Summary statistics for scored records."""

    average_score: float
    min_score: float
    max_score: float
    median_score: float
    std_deviation: float
    total_records: int
    pass_records: int
    fail_records: int


class CalculateScoresRequest(BaseModel):
    """Request schema for score calculation."""

    records: list[dict] = Field(..., description="Test records from iPLAS API")
    scoring_configs: list[ScoringConfig] = Field(
        default_factory=list,
        description="Per-test-item scoring configurations"
    )
    include_binary_in_overall: bool = Field(
        default=True,
        description="Include binary items in overall score"
    )


class CalculateScoresResponse(BaseModel):
    """Response schema for score calculation."""

    scored_records: list[RecordScoreResult]
    summary: ScoreSummary


class SaveScoringConfigRequest(BaseModel):
    """Request to save scoring configuration."""

    site: str
    project: str
    station: str | None = None
    configs: list[ScoringConfig]


class GetScoringConfigResponse(BaseModel):
    """Response for saved scoring configuration."""

    site: str
    project: str
    station: str | None = None
    configs: list[ScoringConfig]
    created_at: str
    updated_at: str


# Default parameters for each scoring type
SCORING_TYPE_DEFAULTS: dict[ScoringType, dict] = {
    ScoringType.SYMMETRICAL: {
        "limit_score": 1.0  # Score at UCL/LCL boundary (on 0-10 scale)
    },
    ScoringType.ASYMMETRICAL: {
        "limit_score": 1.0,  # Score at UCL/LCL boundary (on 0-10 scale)
        "policy": ScoringPolicy.SYMMETRICAL  # Default policy
    },
    ScoringType.PER_MASK: {
        "limit_score": 1.0  # Score at UCL boundary (on 0-10 scale)
    },
    ScoringType.EVM: {
        "limit_score": 1.0,  # Score at UCL boundary (on 0-10 scale)
        "reference_best": -35.0,  # Best possible EVM value
        "exponent": 0.25  # Gentle decay (lower = gentler)
    },
    ScoringType.BINARY: {},
    # Legacy defaults for backwards compatibility
    ScoringType.SYMMETRICAL_NL: {
        "limit_score": 1.0
    },
    ScoringType.THROUGHPUT: {
        "limit_score": 1.0
    }
}
