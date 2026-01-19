import os
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import require_admin
from app.dependencies.external_api_client import get_settings
from app.models.app_config import AppConfig
from app.models.user import User
from app.schemas.app_config_schemas import AppConfigResponse, AppConfigUpdateRequest

router = APIRouter(prefix="/api/app-config", tags=["App_Config"])

db_dependency = Depends(get_db)
settings_dependency = Depends(get_settings)
admin_dependency = Depends(require_admin)


def _default_config(settings) -> dict:
    return {
        "name": settings.app_name,
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "description": "Wireless Test Data Analysis Platform",
    }


def _format_response(config: AppConfig) -> AppConfigResponse:
    return AppConfigResponse(
        name=config.name,
        version=config.version,
        description=config.description,
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


@router.get(
    "",
    response_model=AppConfigResponse,
    summary="Get application configuration",
    description="Return the current app name, version, and description.",
)
def get_app_config(
    db: Session = db_dependency,
    settings = settings_dependency,
):
    config = _get_or_create_config(db, settings)
    return _format_response(config)


@router.put(
    "",
    response_model=AppConfigResponse,
    summary="Update application configuration (admin)",
    description="Update app name, version, and description. Admin only.",
)
def update_app_config(
    payload: AppConfigUpdateRequest,
    db: Session = db_dependency,
    current_user: User = admin_dependency,
    settings = settings_dependency,
):
    config = _get_or_create_config(db, settings)
    config.name = payload.name
    config.version = payload.version
    config.description = payload.description
    config.updated_by = current_user.username
    config.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(config)
    return _format_response(config)
