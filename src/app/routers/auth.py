import os
from typing import Annotated
import logging

from fastapi import APIRouter, Depends, Form, Header, HTTPException
from fastapi.responses import JSONResponse
import httpx
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user, require_admin
from app.dependencies.external_api_client import get_dut_client, get_settings
from app.models.user import User as DBUser
from app.schemas.auth_schemas import (
    PasswordChangedResponse,
    TokenResponse,
    TokenRevokedResponse,
    UserCreatedResponse,
    UserDeletedResponse,
    UserResponse,
)
from app.services import dut_token_service
from app.utils import auth as auth_utils
from app.utils.admin_access import is_user_admin

router = APIRouter(prefix="/api/auth", tags=["Auth"])
# module-level dependency to avoid calling Depends() inside function defaults
db_dependency = Depends(get_db)
dut_client_dependency = Depends(get_dut_client)
admin_dependency = Depends(require_admin)
current_user_dependency = Depends(get_current_user)
settings_dependency = Depends(get_settings)


@router.post(
    "/login",
    summary="Login with username/password",
    description="Authenticate against the local user store and receive access and refresh JWTs.",
    response_model=TokenResponse,
)
def login(
    username: Annotated[str, Form(description="Username for authentication (e.g., jsmith)")],
    password: Annotated[
        str,
        Form(description="User password (e.g., SuperSecret123)", json_schema_extra={"format": "password"}),
    ],
    db: Session = db_dependency,
):
    user = auth_utils.authenticate_user(db, username, password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials", headers={"WWW-Authenticate": "Bearer"})

    # Update last_login timestamp
    from datetime import UTC, datetime

    user.last_login = datetime.now(UTC)
    db.commit()

    access_token = auth_utils.create_access_token(user)
    refresh_token = auth_utils.create_refresh_token(user)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "is_admin": is_user_admin(user),
            "is_ptb_admin": user.is_ptb_admin,  # External admin status
            "worker_id": user.worker_id,
            "roles": [r.name for r in user.roles],
        },
    }


@router.post(
    "/external-login",
    summary="Login via external DUT API system",
    description="Authenticate against the external DUT API, fetch user account info, sync admin status from is_ptb_admin, and issue local JWTs.",
    response_model=TokenResponse,
)
async def external_login(
    username: Annotated[str, Form(description="DUT API username (e.g., oa_username)")],
    password: Annotated[
        str,
        Form(description="DUT API password (e.g., oa_password)", json_schema_extra={"format": "password"}),
    ],
    db: Session = db_dependency,
    client=dut_client_dependency,
):
    """
    Authenticate via external DUT API, fetch user account info, and sync local user.

    Process:
    1. Authenticate with external DUT API using credentials
    2. Fetch user account info including is_ptb_admin, email, etc.
    3. Create/update local user with synced data
    4. Set is_admin based on is_ptb_admin from external API
    5. Issue local JWT tokens
    """
    from datetime import UTC, datetime

    logger = logging.getLogger(__name__)

    try:
        # Step 1: Authenticate with external DUT API
        try:
            auth_data = await client.authenticate(username=username, password=password)
        except httpx.HTTPStatusError as exc:
            normalized_username = auth_utils.normalize_username(username)
            if exc.response.status_code in {401, 403} and normalized_username and normalized_username != username:
                auth_data = await client.authenticate(username=normalized_username, password=password)
            else:
                raise HTTPException(status_code=401, detail="invalid external credentials") from exc
        access_token_external = auth_data.get("access")
        refresh_token_external = auth_data.get("refresh")

        if not access_token_external:
            raise HTTPException(status_code=401, detail="external auth failed - no access token returned")

        # Step 2: Fetch user account info from external API
        try:
            user_info = await client.get_user_account_info()
            # Extract is_ptb_admin/worker_id from nested employee_info object
            employee_info = user_info.get("employee_info", {})
            is_ptb_admin = employee_info.get("is_ptb_admin", False)
            worker_id = employee_info.get("worker_id")
            logger.info(f"Fetched external user info for {username}: email={user_info.get('email')}, is_ptb_admin={is_ptb_admin}")
        except Exception as e:
            logger.warning(f"Failed to fetch user account info for {username}: {e}")
            # Fallback if account info fetch fails - use basic info
            user_info = {
                "username": username,
                "email": None,
            }
            is_ptb_admin = False
            worker_id = None

        # Extract user details from external API
        external_email = user_info.get("email")

        # Step 3: Create or update local user
        user = auth_utils.get_user(db, username)
        if not user:
            # First time login - create new user
            # Keep is_admin=False (local admin), but set is_ptb_admin from external API
            logger.info(f"Creating new user {username} with is_ptb_admin={is_ptb_admin} from external API")
            user = auth_utils.create_user(db, username, password, is_admin=False)
            user.is_ptb_admin = is_ptb_admin
            user.worker_id = worker_id
            if external_email:
                user.email = external_email
            db.commit()
            db.refresh(user)
        else:
            # Existing user - sync PTB admin status and email from external API
            # Keep local is_admin unchanged (only update is_ptb_admin)
            logger.info(f"Updating existing user {username}: is_ptb_admin {user.is_ptb_admin} -> {is_ptb_admin}, is_admin={user.is_admin} (unchanged)")

            # Update PTB admin status from external API
            user.is_ptb_admin = is_ptb_admin

            # Update worker_id from external API
            if worker_id and user.worker_id != worker_id:
                user.worker_id = worker_id

            # Update email if provided and different
            if external_email and user.email != external_email:
                user.email = external_email

            # Update password hash in case it changed
            user.password_hash = auth_utils.hash_password(password)

            db.commit()
            db.refresh(user)

        # Step 4: Update last_login timestamp
        user.last_login = datetime.now(UTC)
        db.commit()

        # Step 5: Save external DUT tokens securely for later use
        dut_token_service.store_tokens(user.username, access_token_external, refresh_token_external)

        # Step 6: Issue our local JWT tokens
        access_token = auth_utils.create_access_token(user)
        refresh_token = auth_utils.create_refresh_token(user)
        # Admin access granted if EITHER is_admin OR is_ptb_admin is true
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "username": user.username,
                "is_admin": is_user_admin(user),
                "is_ptb_admin": user.is_ptb_admin,  # External admin status
                "worker_id": user.worker_id,
                "roles": [r.name for r in user.roles],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"External login failed for {username}: {e}")
        raise HTTPException(status_code=500, detail=f"external authentication error: {str(e)}") from e


@router.get(
    "/me",
    summary="Get current user info",
    description="Return the active user's profile, including admin flag and assigned roles.",
    response_model=UserResponse,
)
def me(user: DBUser = current_user_dependency):
    return {
        "username": user.username,
        "is_admin": is_user_admin(user),
        "is_ptb_admin": user.is_ptb_admin,  # External admin status
        "worker_id": user.worker_id,
        "roles": [r.name for r in user.roles],
    }


@router.post(
    "/token/refresh",
    summary="Refresh an access token using a REFRESH JWT",
    description="Validate a refresh token, enforce token versioning, and return rotated JWTs.",
    response_model=TokenResponse,
)
def token_refresh(
    refresh_token: Annotated[str, Form(description="JWT refresh token (e.g., eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9)")],
    db: Session = db_dependency,
):
    payload = auth_utils.decode_jwt(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(401, "invalid refresh token", headers={"WWW-Authenticate": "Bearer"})

    user = auth_utils.get_user(db, payload.get("sub"))
    if not user:
        raise HTTPException(401, "user not found or inactive")

    if not user.is_active:
        raise HTTPException(401, "user not found or inactive")

    # version check enforces stateless revocation for refresh tokens
    if payload.get("ver") != user.token_version:
        raise HTTPException(401, "refresh token no longer valid (revoked)")

    # Issue NEW access token; also rotate refresh token
    new_access = auth_utils.create_access_token(user)
    new_refresh = auth_utils.create_refresh_token(user)
    return {"access_token": new_access, "refresh_token": new_refresh, "token_type": "bearer"}


@router.post(
    "/external-token",
    summary="(admin) Obtain external API access token via external /api/auth/external-login endpoint",
    description="Call the external DUT authentication endpoint using either supplied credentials or the configured service account.",
    responses={
        200: {
            "description": "External access token",
            "content": {"application/json": {"example": {"access": "ey..."}}},
        }
    },
)
async def external_token(
    authorization: str | None = Header(None),
    username: Annotated[str | None, Form(description="Optional DUT API username (e.g., service_account)")] = None,
    password: Annotated[
        str | None,
        Form(description="Optional DUT API password (e.g., ServicePass789)", json_schema_extra={"format": "password"}),
    ] = None,
    client=dut_client_dependency,
):
    """Admin-only: call external authentication and return its access token.

    If `username`/`password` are provided they will be used for the external API call,
    otherwise the server-wide DUT API credentials from settings are used.
    """
    # verify admin
    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization header")
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="only bearer supported")
        uname, is_admin = auth_utils.decode_jwt_token(token)
        if not uname or not is_admin:
            raise HTTPException(status_code=403, detail="admin privileges required")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="invalid authorization header") from None

    # call external token endpoint via DUTAPIClient
    try:
        if username and password:
            data = await client.authenticate(username=username, password=password)
        else:
            data = await client.authenticate(
                username=os.environ.get("dut_api_username"),
                password=os.environ.get("dut_api_password"),
            )
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"external auth failed: {e}") from e
