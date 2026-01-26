"""
Scoring System Schemas

Pydantic models for the scoring system used in test item evaluation.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScoringType(str, Enum):
    """Available scoring algorithm types."""
    
    SYMMETRICAL = "symmetrical"
    SYMMETRICAL_NL = "symmetrical_nl"
    EVM = "evm"
    THROUGHPUT = "throughput"
    ASYMMETRICAL = "asymmetrical"
    PER_MASK = "per_mask"
    BINARY = "binary"


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
    alpha: Optional[float] = Field(
        default=None, 
        ge=0, 
        le=1, 
        description="Min score at limit boundary (for symmetrical/asymmetrical)"
    )
    target: Optional[float] = Field(
        default=None, 
        description="User-defined target value (for asymmetrical)"
    )
    target_score: Optional[float] = Field(
        default=None, 
        ge=0, 
        le=1, 
        description="Score at target deviation (for non-linear types)"
    )
    target_deviation: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Deviation value for target_score (for Gaussian)"
    )
    min_score: Optional[float] = Field(
        default=None, 
        ge=0, 
        le=1, 
        description="Minimum score at lower limit (for throughput)"
    )
    max_deviation: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Maximum deviation for PER/MASK scoring"
    )


class TestItemScoreResult(BaseModel):
    """Score result for a single test item."""
    
    test_item_name: str
    value: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    status: str
    scoring_type: ScoringType
    score: float = Field(ge=0, le=1)
    deviation: Optional[float] = None


class RecordScoreResult(BaseModel):
    """Score result for a single test record (ISN/DUT)."""
    
    isn: str
    device_id: str
    station: str
    test_start_time: str
    test_status: str
    overall_score: float = Field(ge=0, le=1)
    value_items_score: Optional[float] = None
    bin_items_score: Optional[float] = None
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
    station: Optional[str] = None
    configs: list[ScoringConfig]


class GetScoringConfigResponse(BaseModel):
    """Response for saved scoring configuration."""
    
    site: str
    project: str
    station: Optional[str] = None
    configs: list[ScoringConfig]
    created_at: str
    updated_at: str


# Default parameters for each scoring type
SCORING_TYPE_DEFAULTS: dict[ScoringType, dict] = {
    ScoringType.SYMMETRICAL: {
        "alpha": 0.8
    },
    ScoringType.SYMMETRICAL_NL: {
        "target_score": 0.8,
        "target_deviation": 2.5
    },
    ScoringType.EVM: {
        "target_score": 0.9,
        "target": -30.0  # Target EVM value
    },
    ScoringType.THROUGHPUT: {
        "min_score": 0.4,
        "target_score": 0.9
    },
    ScoringType.ASYMMETRICAL: {
        "alpha": 0.4
    },
    ScoringType.PER_MASK: {
        "max_deviation": 10.0
    },
    ScoringType.BINARY: {}
}
