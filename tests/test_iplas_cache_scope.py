from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from app.routers import iplas_proxy
from app.routers.iplas_proxy import (
    _build_station_bucket_scope_hash,
    _credential_cache_scope,
    _generate_cache_key,
)
from app.schemas.iplas_schemas import IplasDeviceListRequest


def test_custom_tokens_have_distinct_non_reversible_cache_scopes():
    first = _credential_cache_scope("PTB", "token-one")
    second = _credential_cache_scope("PTB", "token-two")

    assert first != second
    assert "token-one" not in first
    assert "token-two" not in second
    assert _credential_cache_scope("PTB") == "configured"


def test_raw_data_and_bucket_keys_include_credential_scope():
    begin = datetime(2026, 8, 21, 8, tzinfo=UTC)
    end = datetime(2026, 8, 21, 9, tzinfo=UTC)
    first_scope = _credential_cache_scope("PTB", "token-one")
    second_scope = _credential_cache_scope("PTB", "token-two")

    first_key = _generate_cache_key("PTB", "HH5K", "Wireless", "ALL", begin, end, "ALL", "token-one")
    second_key = _generate_cache_key("PTB", "HH5K", "Wireless", "ALL", begin, end, "ALL", "token-two")
    first_bucket = _build_station_bucket_scope_hash("PTB", "HH5K", "Wireless", "ALL", "ALL", first_scope)
    second_bucket = _build_station_bucket_scope_hash("PTB", "HH5K", "Wireless", "ALL", "ALL", second_scope)

    assert first_key != second_key
    assert first_bucket != second_bucket
    assert "token-one" not in first_key
    assert "token-two" not in second_key


@pytest.mark.asyncio
async def test_empty_device_response_is_cached_briefly(monkeypatch):
    store: dict[str, bytes] = {}
    upstream_calls = 0
    observed_ttls: list[int] = []

    class StubRedis:
        def get(self, key):
            return store.get(key)

        def setex(self, key, ttl, value):
            observed_ttls.append(ttl)
            store[key] = value

    class StubResponse:
        status_code = 200

        @staticmethod
        def json():
            return []

    class StubClient:
        async def get(self, *_args, **_kwargs):
            nonlocal upstream_calls
            upstream_calls += 1
            return StubResponse()

    @asynccontextmanager
    async def pooled_client():
        yield StubClient()

    async def get_redis():
        return StubRedis()

    monkeypatch.setattr(iplas_proxy, "_get_redis_client_async", get_redis)
    monkeypatch.setattr(iplas_proxy, "_pooled_client", pooled_client)
    monkeypatch.setattr(
        iplas_proxy,
        "_get_site_config",
        lambda _site, _token: {"v2_url": "http://iplas.test", "token": "configured"},
    )
    request = IplasDeviceListRequest(
        site="PTB",
        project="HH5K",
        station="Wireless_Test_2_5G",
        start_time=datetime(2026, 8, 21, 8),
        end_time=datetime(2026, 8, 21, 9),
    )

    first = await iplas_proxy.get_devices(request)
    second = await iplas_proxy.get_devices(request)

    assert first.cached is False
    assert second.cached is True
    assert upstream_calls == 1
    assert observed_ttls == [iplas_proxy.IPLAS_CACHE_TTL_EMPTY]
