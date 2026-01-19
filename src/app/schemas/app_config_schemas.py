from pydantic import BaseModel, Field


class AppConfigResponse(BaseModel):
    name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    description: str | None = Field(None, description="Application description")
    updated_at: str | None = Field(None, description="Last update timestamp")
    updated_by: str | None = Field(None, description="Last updated by username")


class AppConfigUpdateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    version: str = Field(..., min_length=1, max_length=64)
    description: str | None = Field(None, max_length=500)
