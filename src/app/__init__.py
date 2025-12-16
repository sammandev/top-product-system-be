try:
    # Import the FastAPI application if possible. If import fails (for example
    # when running lightweight scripts that only need DB utils), fall back to
    # None so callers can still import subpackages safely.
    from .main import app  # type: ignore
except Exception:
    app = None

__all__ = ["app"]
