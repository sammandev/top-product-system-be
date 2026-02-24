from pydantic import BaseModel, Field


# ============================================================================
# App Config (General)
# ============================================================================


class AppConfigResponse(BaseModel):
    name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    description: str | None = Field(None, description="Application description")
    tab_title: str | None = Field(None, description="Browser tab title")
    favicon_url: str | None = Field(None, description="Favicon URL path")
    updated_at: str | None = Field(None, description="Last update timestamp")
    updated_by: str | None = Field(None, description="Last updated by username")


class AppConfigUpdateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    version: str = Field(..., min_length=1, max_length=64)
    description: str | None = Field(None, max_length=500)
    tab_title: str | None = Field(None, max_length=200)


# ============================================================================
# IPLAS Tokens
# ============================================================================


class IplasTokenResponse(BaseModel):
    id: int
    site: str
    base_url: str
    token_masked: str = Field(..., description="Masked token value")
    label: str | None = None
    is_active: bool
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None


class IplasTokenCreateRequest(BaseModel):
    site: str = Field(..., min_length=2, max_length=10, description="Site identifier (PTB, PSZ, PXD, PVN, PTY)")
    base_url: str = Field(..., min_length=5, max_length=500, description="Base URL for the iPLAS API")
    token_value: str = Field(..., min_length=1, description="API token value")
    label: str | None = Field(None, max_length=200)
    is_active: bool = Field(True)


class IplasTokenUpdateRequest(BaseModel):
    site: str | None = Field(None, min_length=2, max_length=10)
    base_url: str | None = Field(None, min_length=5, max_length=500)
    token_value: str | None = Field(None, min_length=1, description="New token value (omit to keep existing)")
    label: str | None = Field(None, max_length=200)
    is_active: bool | None = None


class IplasTokenListResponse(BaseModel):
    tokens: list[IplasTokenResponse]
    total: int


# ============================================================================
# SFISTSP Config
# ============================================================================


class SfistspConfigResponse(BaseModel):
    id: int
    base_url: str
    program_id: str
    password_masked: str = Field(..., description="Masked program password")
    timeout: float
    label: str | None = None
    is_active: bool
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None


class SfistspConfigCreateRequest(BaseModel):
    base_url: str = Field(..., min_length=5, max_length=500)
    program_id: str = Field(..., min_length=1, max_length=200)
    program_password: str = Field(..., min_length=1)
    timeout: float = Field(30.0, ge=1, le=300)
    label: str | None = Field(None, max_length=200)
    is_active: bool = Field(True)


class SfistspConfigUpdateRequest(BaseModel):
    base_url: str | None = Field(None, min_length=5, max_length=500)
    program_id: str | None = Field(None, min_length=1, max_length=200)
    program_password: str | None = Field(None, min_length=1, description="New password (omit to keep existing)")
    timeout: float | None = Field(None, ge=1, le=300)
    label: str | None = Field(None, max_length=200)
    is_active: bool | None = None


class SfistspConfigListResponse(BaseModel):
    configs: list[SfistspConfigResponse]
    total: int


# ============================================================================
# Guest Credentials
# ============================================================================


class GuestCredentialResponse(BaseModel):
    id: int
    username_masked: str = Field(..., description="Masked username")
    label: str | None = None
    is_active: bool
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None


class GuestCredentialCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1)
    label: str | None = Field(None, max_length=200)
    is_active: bool = Field(True)


class GuestCredentialUpdateRequest(BaseModel):
    username: str | None = Field(None, min_length=1, max_length=200)
    password: str | None = Field(None, min_length=1, description="New password (omit to keep existing)")
    label: str | None = Field(None, max_length=200)
    is_active: bool | None = None


class GuestCredentialListResponse(BaseModel):
    credentials: list[GuestCredentialResponse]
    total: int
