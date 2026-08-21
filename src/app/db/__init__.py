import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

# Load the project .env before resolving database settings. This module is
# imported very early by auth/model dependencies, before app.main gets a chance
# to call load_dotenv().
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DB_PASSWORD = os.environ.get("DB_PASSWORD", os.environ.get("DB_PASS", "test123"))

# Build DATABASE_URL from env or use DATABASE_URL if provided
DB_NAME = os.environ.get("DB_NAME", os.environ.get("DB", "postgres"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+psycopg://{os.environ.get('DB_USER', 'postgres')}:{DB_PASSWORD}@{os.environ.get('DB_HOST', 'localhost')}:{os.environ.get('DB_PORT', '5432')}/{DB_NAME}",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _engine_options(database_url: str) -> dict:
    url = make_url(database_url)
    if not url.get_backend_name().startswith("postgresql"):
        return {}

    options: dict = {
        "pool_pre_ping": _env_bool("DB_POOL_PRE_PING", True),
        "pool_recycle": _env_int("DB_POOL_RECYCLE_SECONDS", 1800),
        # Sessions are synchronous, so each in-flight request holds a connection
        # for its whole lifetime. SQLAlchemy's default of 5 + 10 overflow starves
        # quickly once several slow endpoints run concurrently. Keep
        # workers * (pool_size + max_overflow) under Postgres max_connections.
        "pool_size": _env_int("DB_POOL_SIZE", 10),
        "max_overflow": _env_int("DB_MAX_OVERFLOW", 10),
        "pool_timeout": _env_int("DB_POOL_TIMEOUT", 30),
        "connect_args": {"connect_timeout": _env_int("DB_CONNECT_TIMEOUT", 10)},
    }
    return options


# Lazy engine creation to avoid environment races (tests may set env vars before starting uvicorn)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, **_engine_options(DATABASE_URL))
    return _engine


def recreate_engine(new_database_url: str | None = None):
    """
    Re-create the SQLAlchemy engine with an optional new DATABASE_URL.

    This is safe to call from test setup or admin scripts when the
    environment changes. It updates the module-level `engine` and
    re-binds `SessionLocal`.
    """
    global _engine, engine, SessionLocal, DATABASE_URL
    if new_database_url:
        DATABASE_URL = new_database_url
    # dispose old engine if present
    try:
        if _engine is not None:
            _engine.dispose()
    except Exception:
        pass
    _engine = create_engine(DATABASE_URL, **_engine_options(DATABASE_URL))
    engine = _engine
    SessionLocal.configure(bind=engine)
    return engine


# Create the engine now (honors any DATABASE_URL set before import) and
# provide it as the module-level `engine` symbol which tests/tools expect.
engine = get_engine()

# Sessions bind directly to the engine
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    # SQLAlchemy sessionmaker will call get_engine() via the callable bind if needed
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Import models package to register models with Base (no-op if missing)
try:
    import app.models  # noqa: F401
except Exception:
    pass
