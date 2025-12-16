import asyncio
import logging
import os
import warnings
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from starlette.staticfiles import StaticFiles

try:
    from swagger_ui_bundle import swagger_ui_5_path
except ImportError:  # pragma: no cover - optional dependency for offline Swagger
    swagger_ui_5_path = None

from app.dependencies.external_api_client import get_settings
from app.utils.helpers import _start_cleanup_worker

# Suppress Pydantic v1 compatibility warning for Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater", category=UserWarning, module="fastapi._compat.v1")

logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Initialize FastAPI Cache with Redis backend
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis_client), prefix="api-cache")
    logger.info(f"FastAPI Cache initialized with Redis at {redis_url}")

    try:
        yield
    except asyncio.CancelledError:
        logger.info("Application shutdown requested (CancelledError). Exiting gracefully.")
    except Exception:
        logger.exception("Unhandled exception during application lifespan shutdown.")
        raise
    finally:
        await redis_client.close()
        logger.info("Redis connection closed")


settings = get_settings()

tags_metadata = [
    {"name": "Root", "description": "Basic status endpoint."},
    {"name": "Auth", "description": "Authentication and token management endpoints."},
    {"name": "User_Management", "description": "🔒 **Admin Only** - User account management APIs. Requires admin authentication."},
    {"name": "RBAC_Management", "description": "🔒 **Admin Only** - Role and permission administration APIs. Requires admin authentication."},
    {"name": "Dashboard", "description": "Dashboard statistics, metrics, activity logs and history."},
    {"name": "Top_Products_Database", "description": "Top product analyses database management."},
    {"name": "Parsing", "description": "Upload and parse CSV/XLSX files."},
    {"name": "Comparison", "description": "Format comparison and conversion endpoints."},
    {"name": "MultiDUT", "description": "Analyze multiple DUT data."},
    {"name": "DVT_MC2", "description": "DVT to MC2 conversion."},
    {"name": "DUT_Management", "description": "External DUT information API endpoints."},
    {"name": "Test_Log_Processing", "description": "Parse and compare test log files."},
]

use_local_swagger = swagger_ui_5_path is not None

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    openapi_tags=tags_metadata,
    docs_url=None if use_local_swagger else "/swagger",
    swagger_ui_parameters={"displayRequestDuration": True, "persistAuthorization": True},
    redoc_url=None if use_local_swagger else "/redoc",
    lifespan=_lifespan,
)

# Move any "default" content into "Root" and remove "default"
new_tags = []
for t in tags_metadata:
    name = t.get("name")
    if name == "Root":
        new_tags.append({"name": "Root", "description": "DUT_Management API"})
    elif name == "default":
        continue
    else:
        new_tags.append(t)
_original_openapi = app.openapi


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = _original_openapi()
    schema["tags"] = new_tags

    # Ensure we have a bearer security scheme so Swagger UI shows the Authorize button
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    # create canonical scheme
    canonical = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Provide a Bearer token obtained from /api/auth/login or /api/auth/external-login",
    }
    # Insert/overwrite the canonical name we want
    security_schemes["BearerAuth"] = canonical

    # Detect any other auto-generated http/bearer schemes and remap them to the canonical name
    other_bearers = [k for k, v in list(security_schemes.items()) if k != "BearerAuth" and isinstance(v, dict) and v.get("type") == "http" and str(v.get("scheme", "")).lower() == "bearer"]

    if other_bearers:
        # remove duplicate entries
        for k in other_bearers:
            security_schemes.pop(k, None)

        # Helper to remap security requirement entries (list[dict]) to use BearerAuth
        def _remap_security_list(sec_list):
            if not isinstance(sec_list, list):
                return sec_list
            new = []
            for item in sec_list:
                if not isinstance(item, dict):
                    new.append(item)
                    continue
                # If any key in the dict was one of the old bearers, replace with canonical
                remapped = {}
                replaced = False
                for sk, scopes in item.items():
                    if sk in other_bearers:
                        # merge scopes (usually empty list for http bearer)
                        remapped.setdefault("BearerAuth", [])
                        remapped["BearerAuth"] = list(set(remapped["BearerAuth"]) | set(scopes or []))
                        replaced = True
                    else:
                        remapped[sk] = scopes
                if replaced and "BearerAuth" not in remapped:
                    remapped["BearerAuth"] = []
                new.append(remapped)
            return new

        # Remap top-level security
        if "security" in schema:
            schema["security"] = _remap_security_list(schema.get("security"))

        # Remap per-operation security entries
        for path_item in schema.get("paths", {}).values():
            if not isinstance(path_item, dict):
                continue
            for op in path_item.values():
                if not isinstance(op, dict):
                    continue
                if "security" in op:
                    op["security"] = _remap_security_list(op.get("security"))

    paths = schema.get("paths", {})
    root_path = paths.get("/", {})
    get_op = root_path.get("get")
    if get_op is not None:
        # Make the root operation appear under Root tag
        get_op["tags"] = ["Root"]
        # Provide a short description on the operation itself
        get_op.setdefault("description", "DUT Management API")

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


if use_local_swagger:
    app.mount("/swagger-ui-assets", StaticFiles(directory=swagger_ui_5_path), name="swagger-ui-assets")

    @app.get("/swagger", include_in_schema=False)
    async def swagger_ui():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{settings.app_name} - Swagger UI",
            swagger_js_url="/swagger-ui-assets/swagger-ui-bundle.js",
            swagger_css_url="/swagger-ui-assets/swagger-ui.css",
        )

    @app.get("/swagger/oauth2-redirect", include_in_schema=False)
    async def swagger_oauth_redirect():
        return get_swagger_ui_oauth2_redirect_html()


# Basic root endpoint
@app.get("/")
async def root():
    return {"message": "DUT Management API"}


# Set up GZip compression middleware (BEFORE CORS)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses > 1KB
    compresslevel=6,  # Default compression level (1-9)
)

# Set up CORS middleware
cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://172.18.220.56,http://172.18.220.56:3000"
)
cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]

logger.info(f"CORS enabled for origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
from .routers import (  # noqa: E402
    activity,
    admin_rbac,
    admin_users,
    auth,
    cache_admin,
    compare,
    dashboard,
    dvt_mc2_converter,
    external_api_client,
    format_compare,
    health,
    multi_dut_analysis,
    parsing,
    rbac,
    test_log,
    top_products,
)

app.include_router(activity.router)
app.include_router(admin_rbac.router)
app.include_router(admin_users.router)
app.include_router(auth.router)
app.include_router(cache_admin.router)
app.include_router(compare.router)
app.include_router(dashboard.router)
app.include_router(external_api_client.router)
app.include_router(dvt_mc2_converter.router)
app.include_router(format_compare.router)
app.include_router(health.router)
app.include_router(multi_dut_analysis.router)
app.include_router(parsing.router)
app.include_router(rbac.router)
app.include_router(test_log.router)
app.include_router(top_products.router)

# debug router removed after diagnostics

# start background cleanup of uploads
_start_cleanup_worker()


__all__ = ["app"]
