import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Build DATABASE_URL from env or use DATABASE_URL if provided
DB_NAME = os.environ.get("DB_NAME", os.environ.get("DB", "postgres"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+psycopg://{os.environ.get('DB_USER', 'postgres')}:{os.environ.get('DB_PASS', 'test123')}@{os.environ.get('DB_HOST', 'localhost')}:{os.environ.get('DB_PORT', '5432')}/{DB_NAME}",
)

# Lazy engine creation to avoid environment races (tests may set env vars before starting uvicorn)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
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
    _engine = create_engine(DATABASE_URL)
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
