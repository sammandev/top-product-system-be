import asyncio
import logging
from datetime import UTC, datetime
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.external_services.dut_api_client import DUTAPIClient
from app.services.redis_cache_service import RedisCacheService

logger = logging.getLogger(__name__)

_client_lock = asyncio.Lock()


@lru_cache
def _get_cached_client(base_url: str) -> DUTAPIClient:
    return DUTAPIClient(base_url=base_url)


class Settings(BaseSettings):
    dut_api_base_url: str = "http://172.18.220.56:9001"
    dut_api_username: str | None = None
    dut_api_password: str | None = None
    app_name: str = "DUT Management API"
    debug: bool = False
    redis_url: str | None = None
    redis_dut_cache_ttl: int = 300  # 5 minutes default
    enable_dut_cache: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug(cls, value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            # Treat common logging level strings as non-debug defaults instead of erroring.
            if normalized in {"warn", "warning", "info", "error", "critical"}:
                return False
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


async def get_dut_client() -> DUTAPIClient:
    """Return a cached DUT API client, authenticating with service credentials when configured."""

    settings = get_settings()
    client = _get_cached_client(settings.dut_api_base_url)

    if settings.dut_api_username and settings.dut_api_password:
        token_expired = _token_expired(client.token_expiry)
        if not client.access_token or token_expired:
            async with _client_lock:
                token_expired = _token_expired(client.token_expiry)
                if not client.access_token or token_expired:
                    try:
                        await client.authenticate(settings.dut_api_username, settings.dut_api_password)
                    except Exception as exc:
                        logger.warning("Unable to authenticate DUT service account: %s", exc)

    return client


def _token_expired(expiry: datetime | None) -> bool:
    if not expiry:
        return True
    if expiry.tzinfo is None:
        return datetime.now() >= expiry
    return datetime.now(UTC) >= expiry


@lru_cache
def get_redis_client() -> Redis | None:
    """Get cached Redis client instance."""
    settings = get_settings()

    if not settings.redis_url:
        logger.warning("REDIS_URL not configured, caching disabled")
        return None

    try:
        client = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,  # We handle encoding with orjson
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        client.ping()
        logger.info("Redis client connected successfully")
        return client
    except (RedisConnectionError, Exception) as exc:
        logger.warning("Failed to connect to Redis: %s", exc)
        return None


@lru_cache
def get_cache_service() -> RedisCacheService:
    """Get cached RedisCacheService instance."""
    settings = get_settings()
    redis_client = get_redis_client()

    return RedisCacheService(
        redis_client=redis_client,
        default_ttl=settings.redis_dut_cache_ttl,
        enabled=settings.enable_dut_cache,
    )
