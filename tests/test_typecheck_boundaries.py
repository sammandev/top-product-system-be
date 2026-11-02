import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import main
from app.dependencies import authz
from app.routers import auth


@pytest.mark.parametrize("subject", [None, "", 123, ["user"]])
def test_access_token_requires_string_subject(monkeypatch, subject):
    monkeypatch.setattr(authz, "decode_jwt", lambda _token: {"type": "access", "sub": subject})
    engine = create_engine("sqlite://")
    try:
        with Session(engine) as session, pytest.raises(HTTPException) as exc_info:
            authz.get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials="test"), session)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "invalid or expired token"
    finally:
        engine.dispose()


@pytest.mark.parametrize("subject", [None, "", 123, ["user"]])
def test_refresh_token_requires_string_subject(monkeypatch, subject):
    monkeypatch.setattr(auth.auth_utils, "decode_jwt", lambda _token: {"type": "refresh", "sub": subject})
    engine = create_engine("sqlite://")
    try:
        with Session(engine) as session, pytest.raises(HTTPException) as exc_info:
            auth.token_refresh("test", session)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "invalid refresh token"
    finally:
        engine.dispose()


def test_custom_openapi_retains_canonical_bearer_scheme_and_cache(monkeypatch):
    monkeypatch.setattr(main.app, "openapi_schema", None)
    schema = main.app.openapi()
    assert schema is main.app.openapi()
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    assert schema["paths"]["/"]["get"]["tags"] == ["Root"]


@pytest.mark.parametrize("endpoint", [main.custom_swagger_ui_html, main.custom_redoc_html])
async def test_docs_report_disabled_schema(monkeypatch, endpoint):
    monkeypatch.setattr(main.app, "openapi_url", None)
    with pytest.raises(HTTPException) as exc_info:
        await endpoint()
    assert exc_info.value.status_code == 404
