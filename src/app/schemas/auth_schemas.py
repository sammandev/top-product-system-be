from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Login credentials schema with masked password in Swagger UI."""

    username: str = Field(..., description="Username for authentication", json_schema_extra={"example": "john_doe"})
    password: str = Field(
        ...,
        description="User password",
        json_schema_extra={"format": "password", "example": "secure_password123"},
    )


class ExternalLoginRequest(BaseModel):
    """External DUT API login credentials schema with masked password."""

    username: str = Field(..., description="DUT API username", json_schema_extra={"example": "john_doe"})
    password: str = Field(
        ...,
        description="DUT API password",
        json_schema_extra={"format": "password", "example": "secure_password123"},
    )


class CreateUserRequest(BaseModel):
    """User creation schema with masked password."""

    username: str = Field(..., description="New username", json_schema_extra={"example": "new_user"})
    password: str = Field(
        ...,
        description="User password",
        json_schema_extra={"format": "password", "example": "secure_password123"},
    )
    is_admin: bool = Field(False, description="Grant admin privileges")


class ChangePasswordRequest(BaseModel):
    """Password change request schema with masked new password."""

    username: str = Field(..., description="Username to change password for", json_schema_extra={"example": "john_doe"})
    new_password: str = Field(
        ...,
        description="New password",
        json_schema_extra={"format": "password", "example": "new_secure_password123"},
    )


class TokenRefreshRequest(BaseModel):
    """Refresh token request schema."""

    refresh_token: str = Field(
        ...,
        description="JWT refresh token",
        json_schema_extra={"example": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."},
    )


class RevokeTokenRequest(BaseModel):
    """Token revocation request schema."""

    username: str = Field(..., description="Username to revoke tokens for", json_schema_extra={"example": "john_doe"})


class ExternalTokenRequest(BaseModel):
    """External token request with optional credentials."""

    username: str | None = Field(
        None,
        description="Optional DUT API username",
        json_schema_extra={"example": "john_doe"},
    )
    password: str | None = Field(
        None,
        description="Optional DUT API password",
        json_schema_extra={"format": "password", "example": "secure_password123"},
    )


class TokenResponse(BaseModel):
    """JWT token response schema."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    user: "UserResponse | None" = Field(None, description="User information (optional)")


class UserResponse(BaseModel):
    """User profile response schema."""

    username: str = Field(..., description="Username")
    is_admin: bool = Field(..., description="Admin status (true if local admin OR external PTB admin)")
    is_ptb_admin: bool = Field(False, description="External PTB admin status (from DUT API)")
    worker_id: str | None = Field(None, description="External worker ID (from DUT API)")
    email: str | None = Field(None, description="User email address")
    role: str = Field("user", description="Access control role (developer/superadmin/user)")
    menu_permissions: dict[str, list[str]] | None = Field(None, description="Per-resource CRUD permissions")
    is_superuser: bool = Field(False, description="Superuser status (from external API)")
    is_staff: bool = Field(False, description="Staff status (from external API)")
    roles: list[str] = Field(default_factory=list, description="Assigned RBAC roles")


class UserCreatedResponse(BaseModel):
    """User creation response schema."""

    username: str = Field(..., description="Created username")
    is_admin: bool = Field(..., description="Admin status")


class PasswordChangedResponse(BaseModel):
    """Password change response schema."""

    username: str = Field(..., description="Username")
    changed: bool = Field(..., description="Password change status")
    revoked_tokens: bool = Field(..., description="Whether tokens were revoked")


class UserDeletedResponse(BaseModel):
    """User deletion response schema."""

    deleted: str = Field(..., description="Deleted username")


class TokenRevokedResponse(BaseModel):
    """Token revocation response schema."""

    revoked: str = Field(..., description="Username with revoked tokens")
    token_version: int = Field(..., description="New token version")
