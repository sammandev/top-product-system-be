"""
Pydantic schemas for SFISTSP API endpoints.

These schemas define the request/response models for ISN reference lookup
using the SFISTSP Web Service.
"""

from pydantic import BaseModel, Field


class SfistspIsnLookupRequest(BaseModel):
    """Request model for ISN lookup."""

    isn: str = Field(..., description="The ISN to look up", examples=["264118730000123"])


class SfistspIsnBatchLookupRequest(BaseModel):
    """Request model for batch ISN lookup."""

    isns: list[str] = Field(
        ...,
        description="List of ISNs to look up",
        min_length=1,
        max_length=100,
        examples=[["264118730000123", "264118730000456"]],
    )


class SfistspIsnReferenceResponse(BaseModel):
    """Response model for a single ISN lookup."""

    isn: str = Field(..., description="The queried ISN")
    ssn: str | None = Field(None, description="The SSN (Serial Number) reference")
    mac: str | None = Field(None, description="The MAC address (12 hex characters)")
    success: bool = Field(..., description="Whether the lookup was successful")
    error_message: str | None = Field(None, description="Error message if lookup failed")
    isn_references: list[str] = Field(
        default_factory=list,
        description="All ISN references extracted from response",
    )


class SfistspIsnBatchLookupResponse(BaseModel):
    """Response model for batch ISN lookup."""

    results: list[SfistspIsnReferenceResponse] = Field(
        ..., description="List of lookup results"
    )
    total_count: int = Field(..., description="Total number of ISNs queried")
    success_count: int = Field(..., description="Number of successful lookups")
    failed_count: int = Field(..., description="Number of failed lookups")


class SfistspConfigResponse(BaseModel):
    """Response model for SFISTSP configuration info."""

    base_url: str = Field(..., description="SFISTSP server base URL")
    endpoint: str = Field(..., description="SFISTSP endpoint path")
    available: bool = Field(..., description="Whether the SFISTSP service is available")
