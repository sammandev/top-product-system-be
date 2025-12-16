import os

import pytest
import requests
from sqlalchemy.exc import IntegrityError

from app.db import Base, engine
from app.models.user import User
from app.utils.auth import hash_password

TEST_BASE = os.environ.get("ASTPARSER_TEST_BASE", "http://127.0.0.1:8002")


def setup_test_db():
    # Ensure SQLite file is used (configured in tests/conftest via env)
    Base.metadata.create_all(bind=engine)
    # Insert a test user
    conn = engine.connect()
    trans = conn.begin()
    try:
        # Use SQL to insert to avoid ORM session complexities in tests
        conn.execute(
            User.__table__.insert(),
            [
                {
                    "username": "testuser",
                    "password_hash": hash_password("testpassword"),
                    "is_admin": False,
                    "is_active": True,
                    "token_version": 1,
                }
            ],
        )
        trans.commit()
    except Exception as exc:
        # If the user already exists (unique constraint), ignore the error so tests are idempotent
        if isinstance(exc, IntegrityError):
            trans.rollback()
        else:
            trans.rollback()
            raise
    finally:
        conn.close()


@pytest.mark.integration
def test_login_and_me():
    """Integration test - requires running server on port 8002."""
    # Check if server is running
    try:
        resp = requests.get(f"{TEST_BASE}/health", timeout=1)
        if resp.status_code != 200:
            pytest.skip("Test server not running - skipping integration test")
    except requests.exceptions.RequestException:
        pytest.skip("Test server not running - skipping integration test")
    
    # Seed DB with test user
    setup_test_db()

    # Login
    resp = requests.post(f"{TEST_BASE}/api/auth/login", data={"username": "testuser", "password": "testpassword"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "access_token" in data and "refresh_token" in data

    access_token = data["access_token"]

    # Use /me with the token
    headers = {"Authorization": f"Bearer {access_token}"}
    resp2 = requests.get(f"{TEST_BASE}/api/auth/me", headers=headers)
    assert resp2.status_code == 200, resp2.text
    me = resp2.json()
    assert me["username"] == "testuser"
