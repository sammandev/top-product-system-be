import threading

import pytest

from app.routers import iplas_proxy


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_value", [b"cached", "cached", None])
async def test_cached_value_get_runs_off_event_loop(cached_value, monkeypatch):
    calling_thread = threading.get_ident()
    get_threads: list[int] = []

    redis = iplas_proxy.Redis.from_url("redis://localhost")

    def get(key):
        assert key == "cache-key"
        get_threads.append(threading.get_ident())
        return cached_value

    monkeypatch.setattr(redis, "get", get)

    assert await iplas_proxy._get_redis_cached_value(redis, "cache-key") == cached_value
    assert get_threads[0] != calling_thread


@pytest.mark.asyncio
async def test_cached_value_get_rejects_non_string_values(monkeypatch):
    redis = iplas_proxy.Redis.from_url("redis://localhost")
    monkeypatch.setattr(redis, "get", lambda _key: object())

    with pytest.raises(TypeError, match="not bytes or text"):
        await iplas_proxy._get_redis_cached_value(redis, "cache-key")
