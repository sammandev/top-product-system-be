import os
import threading
import time

import pytest
import uvicorn
from httpx import ASGITransport, AsyncClient

from app.main import app

_uvicorn_server: uvicorn.Server | None = None
_uvicorn_thread: threading.Thread | None = None

# Test server configuration
TEST_PORT = 8002  # Use different port to avoid conflict with dev server
TEST_BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

# Relax auth requirements for tests that call cleanup endpoints.
# Ensure tests force the DATABASE_URL so the app (imported by uvicorn)
# uses the same SQLite file that tests seed. Use assignment, not setdefault,
# to override any previously set value.
os.environ["ASTPARSER_ADMIN_KEY"] = ""  # Changed from setdefault to force empty value
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
# Set the test base URL for tests that need it
os.environ["ASTPARSER_TEST_BASE"] = TEST_BASE_URL


@pytest.fixture
async def client_fixture():
    """Async HTTP client for testing FastAPI endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
async def auth_headers():
    """Mock authentication headers for testing protected endpoints."""
    # For now, return empty dict - tests should override get_current_user dependency
    return {}


def pytest_sessionstart(session):  # noqa: D401  (pytest hook)
    config = uvicorn.Config(
        "app.main:app",
        host="127.0.0.1",
        port=TEST_PORT,  # Use different port to avoid conflict with dev server
        # Use debug logging during tests to capture JWT debug messages
        log_level="debug",
        reload=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="pytest-uvicorn", daemon=True)

    thread.start()

    # Wait until the server signals it has started or the thread exits.
    timeout = time.time() + 15
    while not server.started and thread.is_alive():
        if time.time() > timeout:
            raise RuntimeError("Timed out starting uvicorn server for tests")
        time.sleep(0.1)

    if not thread.is_alive() or not server.started:
        raise RuntimeError("Failed to start uvicorn server for tests")

    global _uvicorn_server, _uvicorn_thread
    _uvicorn_server = server
    _uvicorn_thread = thread


def pytest_sessionfinish(session, exitstatus):  # noqa: D401  (pytest hook)
    global _uvicorn_server, _uvicorn_thread
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
    if _uvicorn_thread is not None:
        _uvicorn_thread.join(timeout=5)
    _uvicorn_server = None
    _uvicorn_thread = None
