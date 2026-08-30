import asyncio
import time

import pytest

from app.routers import iplas_proxy


@pytest.mark.asyncio
async def test_redis_client_is_reused_between_health_checks(monkeypatch):
    created = 0

    class StubRedis:
        ping_count = 0
        closed = False

        def ping(self):
            self.ping_count += 1

        def close(self):
            self.closed = True

    client = StubRedis()

    class StubRedisFactory:
        @staticmethod
        def from_url(*_args, **_kwargs):
            nonlocal created
            created += 1
            return client

    monkeypatch.setattr(iplas_proxy, "Redis", StubRedisFactory)
    monkeypatch.setattr(iplas_proxy.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(iplas_proxy, "_redis_client", None)
    monkeypatch.setattr(iplas_proxy, "_redis_last_health_check", 0.0)
    monkeypatch.setattr(iplas_proxy, "_redis_unavailable_since", None)

    first = iplas_proxy.get_redis_client()
    second = iplas_proxy.get_redis_client()

    assert first is second is client
    assert created == 1
    assert client.ping_count == 1

    await iplas_proxy.close_redis_client()
    assert client.closed is True


@pytest.mark.asyncio
async def test_async_redis_acquisition_does_not_block_event_loop(monkeypatch):
    events: list[str] = []

    def slow_connect():
        time.sleep(0.05)
        events.append("redis")
        return None

    async def heartbeat():
        await asyncio.sleep(0.005)
        events.append("heartbeat")

    monkeypatch.setattr(iplas_proxy, "get_redis_client", slow_connect)

    await asyncio.gather(iplas_proxy._get_redis_client_async(), heartbeat())

    assert events == ["heartbeat", "redis"]
