import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from app.routers import iplas_proxy


@pytest.mark.asyncio
async def test_server_time_requests_are_coalesced_and_cached(monkeypatch):
    call_count = 0
    observed_timeouts: list[float] = []

    class StubClient:
        async def get(self, _url, *, timeout):
            nonlocal call_count
            call_count += 1
            observed_timeouts.append(timeout)
            await asyncio.sleep(0.01)
            return httpx.Response(200, text="1788082400000")

    async def get_client():
        return StubClient()

    monkeypatch.setattr(iplas_proxy, "_server_time_cache", None)
    monkeypatch.setattr(iplas_proxy, "_get_http_client", get_client)
    monkeypatch.setattr(iplas_proxy, "IPLAS_SERVER_TIME_TIMEOUT", 3.0)

    results = await asyncio.gather(*(iplas_proxy._get_iplas_server_time() for _ in range(5)))
    cached = await iplas_proxy._get_iplas_server_time()

    assert call_count == 1
    assert observed_timeouts == [3.0]
    assert all(result.tzinfo is UTC for result in results)
    assert cached >= datetime.fromtimestamp(1788082400, tz=UTC)
