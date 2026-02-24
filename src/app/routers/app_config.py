"""
Router for Application Configuration management.

Handles:
- General app config (name, version, description, tab title)
- Favicon upload
- iPLAS API token management (encrypted)
- SFISTSP configuration management (encrypted)
- Guest credential management (encrypted)
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user, require_admin
from app.dependencies.external_api_client import get_settings
from app.models.app_config import AppConfig, GuestCredential, IplasToken, SfistspConfig
from app.models.user import User
from app.schemas.app_config_schemas import (
    AppConfigResponse,
    AppConfigUpdateRequest,
    GuestCredentialCreateRequest,
    GuestCredentialListResponse,
    GuestCredentialResponse,
    GuestCredentialUpdateRequest,
    IplasTokenCreateRequest,
    IplasTokenListResponse,
    IplasTokenResponse,
    IplasTokenUpdateRequest,
    SfistspConfigCreateRequest,
    SfistspConfigListResponse,
    SfistspConfigResponse,
    SfistspConfigUpdateRequest,
)
from app.utils.encryption import decrypt_value, encrypt_value, mask_value

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/app-config", tags=["App_Config"])

db_dependency = Depends(get_db)
settings_dependency = Depends(get_settings)
admin_dependency = Depends(require_admin)

# Project root for static file storage
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_STATIC_DIR = _PROJECT_ROOT / "static"
_FAVICON_DIR = _STATIC_DIR / "branding"


# ============================================================================
# Helper Functions
# ============================================================================


def _default_config(settings) -> dict:
    return {
        "name": settings.app_name,
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "description": "Wireless Test Data Analysis Platform",
    }


def _format_config_response(config: AppConfig) -> AppConfigResponse:
    return AppConfigResponse(
        name=config.name,
        version=config.version,
        description=config.description,
        tab_title=config.tab_title,
        favicon_url=f"/static/branding/{os.path.basename(config.favicon_path)}" if config.favicon_path else None,
        updated_at=config.updated_at.isoformat() if config.updated_at else None,
        updated_by=config.updated_by,
    )


def _get_or_create_config(db: Session, settings) -> AppConfig:
    config = db.query(AppConfig).order_by(AppConfig.id).first()
    if config:
        return config

    defaults = _default_config(settings)
    config = AppConfig(
        name=defaults["name"],
        version=defaults["version"],
        description=defaults["description"],
        updated_at=datetime.now(UTC),
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def _format_iplas_token(token: IplasToken) -> IplasTokenResponse:
    decrypted = decrypt_value(token.token_value)
    return IplasTokenResponse(
        id=token.id,
        site=token.site,
        base_url=token.base_url,
        token_masked=mask_value(decrypted),
        label=token.label,
        is_active=token.is_active,
        created_at=token.created_at.isoformat() if token.created_at else None,
        updated_at=token.updated_at.isoformat() if token.updated_at else None,
        updated_by=token.updated_by,
    )


def _format_sfistsp_config(config: SfistspConfig) -> SfistspConfigResponse:
    decrypted_pw = decrypt_value(config.program_password)
    return SfistspConfigResponse(
        id=config.id,
        base_url=config.base_url,
        program_id=config.program_id,
        password_masked=mask_value(decrypted_pw),
        timeout=config.timeout,
        label=config.label,
        is_active=config.is_active,
        created_at=config.created_at.isoformat() if config.created_at else None,
        updated_at=config.updated_at.isoformat() if config.updated_at else None,
        updated_by=config.updated_by,
    )


def _format_guest_credential(cred: GuestCredential) -> GuestCredentialResponse:
    decrypted_username = decrypt_value(cred.username)
    return GuestCredentialResponse(
        id=cred.id,
        username_masked=mask_value(decrypted_username),
        label=cred.label,
        is_active=cred.is_active,
        created_at=cred.created_at.isoformat() if cred.created_at else None,
        updated_at=cred.updated_at.isoformat() if cred.updated_at else None,
        updated_by=cred.updated_by,
    )


def _is_superadmin(user: User) -> bool:
    """Check if user is developer or superadmin."""
    from app.utils.admin_access import is_developer_identity

    if is_developer_identity(user.username, getattr(user, "worker_id", None)):
        return True
    return getattr(user, "role", None) in ("developer", "superadmin")


# ============================================================================
# General App Config Endpoints
# ============================================================================


@router.get(
    "",
    response_model=AppConfigResponse,
    summary="Get application configuration",
    description="Return the current app name, version, description, tab title, and favicon.",
)
def get_app_config(
    db: Session = db_dependency,
    settings=settings_dependency,
):
    config = _get_or_create_config(db, settings)
    return _format_config_response(config)


@router.put(
    "",
    response_model=AppConfigResponse,
    summary="Update application configuration (admin)",
    description="Update app name, version, description, and tab title. Admin only.",
)
def update_app_config(
    payload: AppConfigUpdateRequest,
    db: Session = db_dependency,
    current_user: User = admin_dependency,
    settings=settings_dependency,
):
    config = _get_or_create_config(db, settings)
    config.name = payload.name
    config.version = payload.version
    config.description = payload.description
    config.tab_title = payload.tab_title
    config.updated_by = current_user.username
    config.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(config)
    return _format_config_response(config)


@router.post(
    "/favicon",
    response_model=AppConfigResponse,
    summary="Upload favicon",
    description="Upload a favicon file (.ico, .png, .svg). Superadmin only.",
)
async def upload_favicon(
    file: UploadFile = File(..., description="Favicon file (.ico, .png, .svg)"),
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
    settings=settings_dependency,
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can upload favicon")

    # Validate file type
    allowed_extensions = {".ico", ".png", ".svg"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")

    # Validate file size (max 512KB)
    content = await file.read()
    if len(content) > 512 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 512KB.")

    # Ensure directory exists
    _FAVICON_DIR.mkdir(parents=True, exist_ok=True)

    # Save file
    filename = f"favicon{ext}"
    filepath = _FAVICON_DIR / filename

    with open(filepath, "wb") as f:
        f.write(content)

    # Update config
    config = _get_or_create_config(db, settings)
    config.favicon_path = str(filepath)
    config.updated_by = current_user.username
    config.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(config)

    logger.info(f"Favicon uploaded by {current_user.username}: {filename}")
    return _format_config_response(config)


@router.delete(
    "/favicon",
    summary="Delete favicon",
    description="Remove the custom favicon. Superadmin only.",
)
def delete_favicon(
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
    settings=settings_dependency,
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can delete favicon")

    config = _get_or_create_config(db, settings)
    if config.favicon_path:
        try:
            Path(config.favicon_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete favicon file: {e}")

        config.favicon_path = None
        config.updated_by = current_user.username
        config.updated_at = datetime.now(UTC)
        db.commit()

    return {"message": "Favicon deleted successfully"}


# ============================================================================
# iPLAS Token Endpoints
# ============================================================================


@router.get(
    "/iplas-tokens",
    response_model=IplasTokenListResponse,
    summary="List iPLAS API tokens",
    description="List all iPLAS API tokens (values masked). Superadmin only.",
)
def list_iplas_tokens(
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage iPLAS tokens")

    tokens = db.query(IplasToken).order_by(IplasToken.site, IplasToken.id).all()
    return IplasTokenListResponse(
        tokens=[_format_iplas_token(t) for t in tokens],
        total=len(tokens),
    )


@router.post(
    "/iplas-tokens",
    response_model=IplasTokenResponse,
    status_code=201,
    summary="Create iPLAS API token",
    description="Add a new iPLAS API token. Token value is encrypted at rest.",
)
def create_iplas_token(
    payload: IplasTokenCreateRequest,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage iPLAS tokens")

    site = payload.site.upper()

    # If marking as active, deactivate other tokens for the same site
    if payload.is_active:
        db.query(IplasToken).filter(IplasToken.site == site, IplasToken.is_active.is_(True)).update({"is_active": False})

    token = IplasToken(
        site=site,
        base_url=payload.base_url,
        token_value=encrypt_value(payload.token_value),
        label=payload.label,
        is_active=payload.is_active,
        updated_by=current_user.username,
    )
    db.add(token)
    db.commit()
    db.refresh(token)

    logger.info(f"iPLAS token created for site {site} by {current_user.username}")
    return _format_iplas_token(token)


@router.put(
    "/iplas-tokens/{token_id}",
    response_model=IplasTokenResponse,
    summary="Update iPLAS API token",
)
def update_iplas_token(
    token_id: int,
    payload: IplasTokenUpdateRequest,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage iPLAS tokens")

    token = db.query(IplasToken).filter(IplasToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="iPLAS token not found")

    if payload.site is not None:
        token.site = payload.site.upper()
    if payload.base_url is not None:
        token.base_url = payload.base_url
    if payload.token_value is not None:
        token.token_value = encrypt_value(payload.token_value)
    if payload.label is not None:
        token.label = payload.label
    if payload.is_active is not None:
        if payload.is_active:
            db.query(IplasToken).filter(
                IplasToken.site == token.site,
                IplasToken.id != token.id,
                IplasToken.is_active.is_(True),
            ).update({"is_active": False})
        token.is_active = payload.is_active

    token.updated_by = current_user.username
    token.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(token)

    return _format_iplas_token(token)


@router.delete(
    "/iplas-tokens/{token_id}",
    summary="Delete iPLAS API token",
)
def delete_iplas_token(
    token_id: int,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage iPLAS tokens")

    token = db.query(IplasToken).filter(IplasToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="iPLAS token not found")

    db.delete(token)
    db.commit()

    logger.info(f"iPLAS token {token_id} ({token.site}) deleted by {current_user.username}")
    return {"message": "iPLAS token deleted successfully", "id": token_id}


@router.post(
    "/iplas-tokens/{token_id}/activate",
    response_model=IplasTokenResponse,
    summary="Activate an iPLAS token",
    description="Set this token as the active one for its site. Deactivates other tokens for the same site.",
)
def activate_iplas_token(
    token_id: int,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage iPLAS tokens")

    token = db.query(IplasToken).filter(IplasToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="iPLAS token not found")

    db.query(IplasToken).filter(
        IplasToken.site == token.site,
        IplasToken.id != token.id,
    ).update({"is_active": False})

    token.is_active = True
    token.updated_by = current_user.username
    token.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(token)

    logger.info(f"iPLAS token {token_id} activated for site {token.site} by {current_user.username}")
    return _format_iplas_token(token)


# ============================================================================
# SFISTSP Config Endpoints
# ============================================================================


@router.get(
    "/sfistsp",
    response_model=SfistspConfigListResponse,
    summary="List SFISTSP configurations",
    description="List all SFISTSP configurations (passwords masked). Superadmin only.",
)
def list_sfistsp_configs(
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage SFISTSP configuration")

    configs = db.query(SfistspConfig).order_by(SfistspConfig.id).all()
    return SfistspConfigListResponse(
        configs=[_format_sfistsp_config(c) for c in configs],
        total=len(configs),
    )


@router.post(
    "/sfistsp",
    response_model=SfistspConfigResponse,
    status_code=201,
    summary="Create SFISTSP configuration",
)
def create_sfistsp_config(
    payload: SfistspConfigCreateRequest,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage SFISTSP configuration")

    if payload.is_active:
        db.query(SfistspConfig).filter(SfistspConfig.is_active.is_(True)).update({"is_active": False})

    config = SfistspConfig(
        base_url=payload.base_url,
        program_id=payload.program_id,
        program_password=encrypt_value(payload.program_password),
        timeout=payload.timeout,
        label=payload.label,
        is_active=payload.is_active,
        updated_by=current_user.username,
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    logger.info(f"SFISTSP config created by {current_user.username}")
    return _format_sfistsp_config(config)


@router.put(
    "/sfistsp/{config_id}",
    response_model=SfistspConfigResponse,
    summary="Update SFISTSP configuration",
)
def update_sfistsp_config(
    config_id: int,
    payload: SfistspConfigUpdateRequest,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage SFISTSP configuration")

    config = db.query(SfistspConfig).filter(SfistspConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="SFISTSP configuration not found")

    if payload.base_url is not None:
        config.base_url = payload.base_url
    if payload.program_id is not None:
        config.program_id = payload.program_id
    if payload.program_password is not None:
        config.program_password = encrypt_value(payload.program_password)
    if payload.timeout is not None:
        config.timeout = payload.timeout
    if payload.label is not None:
        config.label = payload.label
    if payload.is_active is not None:
        if payload.is_active:
            db.query(SfistspConfig).filter(
                SfistspConfig.id != config.id,
                SfistspConfig.is_active.is_(True),
            ).update({"is_active": False})
        config.is_active = payload.is_active

    config.updated_by = current_user.username
    config.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(config)

    return _format_sfistsp_config(config)


@router.delete(
    "/sfistsp/{config_id}",
    summary="Delete SFISTSP configuration",
)
def delete_sfistsp_config(
    config_id: int,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage SFISTSP configuration")

    config = db.query(SfistspConfig).filter(SfistspConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="SFISTSP configuration not found")

    db.delete(config)
    db.commit()

    logger.info(f"SFISTSP config {config_id} deleted by {current_user.username}")
    return {"message": "SFISTSP configuration deleted successfully", "id": config_id}


@router.post(
    "/sfistsp/{config_id}/activate",
    response_model=SfistspConfigResponse,
    summary="Activate a SFISTSP configuration",
)
def activate_sfistsp_config(
    config_id: int,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage SFISTSP configuration")

    config = db.query(SfistspConfig).filter(SfistspConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="SFISTSP configuration not found")

    db.query(SfistspConfig).filter(SfistspConfig.id != config.id).update({"is_active": False})

    config.is_active = True
    config.updated_by = current_user.username
    config.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(config)

    logger.info(f"SFISTSP config {config_id} activated by {current_user.username}")
    return _format_sfistsp_config(config)


# ============================================================================
# Guest Credential Endpoints
# ============================================================================


@router.get(
    "/guest-credentials",
    response_model=GuestCredentialListResponse,
    summary="List guest credentials",
    description="List all guest credentials (values masked). Superadmin only.",
)
def list_guest_credentials(
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage guest credentials")

    credentials = db.query(GuestCredential).order_by(GuestCredential.id).all()
    return GuestCredentialListResponse(
        credentials=[_format_guest_credential(c) for c in credentials],
        total=len(credentials),
    )


@router.post(
    "/guest-credentials",
    response_model=GuestCredentialResponse,
    status_code=201,
    summary="Create guest credential",
)
def create_guest_credential(
    payload: GuestCredentialCreateRequest,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage guest credentials")

    if payload.is_active:
        db.query(GuestCredential).filter(GuestCredential.is_active.is_(True)).update({"is_active": False})

    credential = GuestCredential(
        username=encrypt_value(payload.username),
        password=encrypt_value(payload.password),
        label=payload.label,
        is_active=payload.is_active,
        updated_by=current_user.username,
    )
    db.add(credential)
    db.commit()
    db.refresh(credential)

    logger.info(f"Guest credential created by {current_user.username}")
    return _format_guest_credential(credential)


@router.put(
    "/guest-credentials/{credential_id}",
    response_model=GuestCredentialResponse,
    summary="Update guest credential",
)
def update_guest_credential(
    credential_id: int,
    payload: GuestCredentialUpdateRequest,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage guest credentials")

    credential = db.query(GuestCredential).filter(GuestCredential.id == credential_id).first()
    if not credential:
        raise HTTPException(status_code=404, detail="Guest credential not found")

    if payload.username is not None:
        credential.username = encrypt_value(payload.username)
    if payload.password is not None:
        credential.password = encrypt_value(payload.password)
    if payload.label is not None:
        credential.label = payload.label
    if payload.is_active is not None:
        if payload.is_active:
            db.query(GuestCredential).filter(
                GuestCredential.id != credential.id,
                GuestCredential.is_active.is_(True),
            ).update({"is_active": False})
        credential.is_active = payload.is_active

    credential.updated_by = current_user.username
    credential.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(credential)

    return _format_guest_credential(credential)


@router.delete(
    "/guest-credentials/{credential_id}",
    summary="Delete guest credential",
)
def delete_guest_credential(
    credential_id: int,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage guest credentials")

    credential = db.query(GuestCredential).filter(GuestCredential.id == credential_id).first()
    if not credential:
        raise HTTPException(status_code=404, detail="Guest credential not found")

    db.delete(credential)
    db.commit()

    logger.info(f"Guest credential {credential_id} deleted by {current_user.username}")
    return {"message": "Guest credential deleted successfully", "id": credential_id}


@router.post(
    "/guest-credentials/{credential_id}/activate",
    response_model=GuestCredentialResponse,
    summary="Activate a guest credential",
)
def activate_guest_credential(
    credential_id: int,
    db: Session = db_dependency,
    current_user: User = Depends(get_current_user),
):
    if not _is_superadmin(current_user):
        raise HTTPException(status_code=403, detail="Only superadmin can manage guest credentials")

    credential = db.query(GuestCredential).filter(GuestCredential.id == credential_id).first()
    if not credential:
        raise HTTPException(status_code=404, detail="Guest credential not found")

    db.query(GuestCredential).filter(GuestCredential.id != credential.id).update({"is_active": False})

    credential.is_active = True
    credential.updated_by = current_user.username
    credential.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(credential)

    logger.info(f"Guest credential {credential_id} activated by {current_user.username}")
    return _format_guest_credential(credential)
