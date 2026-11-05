import threading
from datetime import UTC, datetime

import pytest

from app.routers import iplas_proxy
from app.services.redis_cache_service import RedisCacheService


class TrackingRedis:
    def __init__(self) -> None:
        self.threads: list[int] = []

    def ping(self) -> bool:
        return True

    def get(self, _key: str) -> None:
        self.threads.append(threading.get_ident())
        return None

    def setex(self, _key: str, _ttl: int, _value: bytes | str) -> bool:
        self.threads.append(threading.get_ident())
        return True

    def exists(self, _key: str) -> int:
        self.threads.append(threading.get_ident())
        return 0


@pytest.mark.asyncio
async def test_cache_service_redis_io_runs_off_event_loop() -> None:
    redis = TrackingRedis()
    cache = RedisCacheService(redis)
    event_loop_thread = threading.get_ident()

    async def fetch() -> dict[str, str]:
        return {"value": "fresh"}

    assert await cache.get_or_fetch("test", fetch) == {"value": "fresh"}
    assert len(redis.threads) == 2
    assert all(thread != event_loop_thread for thread in redis.threads)


@pytest.mark.asyncio
async def test_iplas_cache_write_runs_off_event_loop() -> None:
    redis = TrackingRedis()
    event_loop_thread = threading.get_ident()

    await iplas_proxy._set_redis_cached_value(redis, "test", 30, b"value")

    assert redis.threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_station_bucket_lock_check_runs_off_event_loop() -> None:
    redis = TrackingRedis()
    event_loop_thread = threading.get_ident()

    await iplas_proxy._wait_for_station_bucket_unlock(
        redis,
        "scope",
        datetime.now(UTC),
        timeout_seconds=0.1,
    )

    assert redis.threads[0] != event_loop_thread
