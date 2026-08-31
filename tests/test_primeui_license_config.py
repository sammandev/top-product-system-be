import base64
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import Base
from app.models.app_config import AppConfig
from app.routers.app_config import (
    get_primeui_license_runtime,
    get_primeui_license_status,
    update_primeui_license,
)
from app.schemas.app_config_schemas import PrimeUILicenseUpdateRequest


def make_license(*, product: str = "primeui", expires_in_days: int = 365) -> str:
    now = datetime.now(UTC)
    claims = {
        "id": "test-license",
        "product": product,
        "tier": "community",
        "type": "dev",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=expires_in_days)).timestamp()),
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{payload}.test-signature"


def encode_payload(payload: object) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{encoded}.test-signature"


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_primeui_license_is_encrypted_masked_and_available_at_runtime(db_session: Session) -> None:
    license_key = make_license()
    settings = SimpleNamespace(app_name="Test App")
    developer = SimpleNamespace(username="developer", worker_id=None, role="developer")

    status = update_primeui_license(
        PrimeUILicenseUpdateRequest(license_key=license_key),
        db_session,
        developer,
        settings,
    )

    stored = db_session.query(AppConfig).one()
    assert stored.primeui_license_key != license_key
    assert status.configured is True
    assert status.valid is True
    assert status.license_masked != license_key
    assert status.product == "primeui"
    assert status.tier == "community"
    assert get_primeui_license_status(db_session, developer, settings) == status
    assert get_primeui_license_runtime(db_session, settings).license == license_key


@pytest.mark.parametrize(
    ("license_key", "expected_detail"),
    [
        (make_license(product="other"), "License key is not valid for PrimeUI"),
        (make_license(expires_in_days=-1), "PrimeUI license key has expired"),
        ("not-a-license-key-at-all-but-long-enough", "Invalid PrimeUI license key format"),
        (encode_payload(["not", "claims"]), "Invalid PrimeUI license key format"),
    ],
)
def test_primeui_license_rejects_invalid_claims(
    db_session: Session,
    license_key: str,
    expected_detail: str,
) -> None:
    settings = SimpleNamespace(app_name="Test App")
    developer = SimpleNamespace(username="developer", worker_id=None, role="developer")

    with pytest.raises(HTTPException) as exc_info:
        update_primeui_license(
            PrimeUILicenseUpdateRequest(license_key=license_key),
            db_session,
            developer,
            settings,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == expected_detail


def test_primeui_license_update_requires_superadmin(db_session: Session) -> None:
    settings = SimpleNamespace(app_name="Test App")
    admin = SimpleNamespace(username="admin", worker_id=None, role="admin")

    with pytest.raises(HTTPException) as exc_info:
        update_primeui_license(
            PrimeUILicenseUpdateRequest(license_key=make_license()),
            db_session,
            admin,
            settings,
        )

    assert exc_info.value.status_code == 403


def test_expired_stored_license_remains_visible_but_not_runtime_active(db_session: Session) -> None:
    from app.utils.encryption import encrypt_value

    license_key = make_license(expires_in_days=-1)
    config = AppConfig(
        name="Test App",
        version="1.0.0",
        primeui_license_key=encrypt_value(license_key),
    )
    db_session.add(config)
    db_session.commit()
    settings = SimpleNamespace(app_name="Test App")
    developer = SimpleNamespace(username="developer", worker_id=None, role="developer")

    status = get_primeui_license_status(db_session, developer, settings)

    assert status.configured is True
    assert status.valid is False
    assert status.validation_error == "PrimeUI license key has expired"
    assert status.license_masked
    assert get_primeui_license_runtime(db_session, settings).license is None