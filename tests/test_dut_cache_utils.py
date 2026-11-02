from typing import Any

import pytest

from app.external_services.dut_api_client import DUTAPIClient
from app.utils import dut_cache


class _PassthroughCache:
    async def get_or_fetch(self, _key, fetch, *args, ttl=None, **kwargs):
        return await fetch(*args, **kwargs)


class _CatalogClient(DUTAPIClient):
    def __init__(self):
        self.sites = [{"id": 1, "name": "North"}, {"id": 2, "name": "South"}]
        self.models = {1: [{"id": 10, "name": "Model A"}], 2: [{"id": 20, "name": "Model B"}]}
        self.stations = {10: [{"id": 100, "name": "Station A"}], 20: [{"id": 200, "name": "Station B"}]}
        self.record_requests = []

    async def get_sites(self) -> list[dict[str, Any]]:
        return self.sites

    async def get_models_by_site(self, site_id: int) -> list[dict[str, Any]]:
        return self.models[site_id]

    async def get_stations_by_model(self, model_id: int) -> list[dict[str, Any]]:
        return self.stations[model_id]

    async def get_dut_records(self, dut_id: str) -> dict[str, Any]:
        self.record_requests.append(dut_id)
        return {"dut_id": dut_id}

    async def get_complete_dut_info(self, dut_id: str, site_name: str | None = None) -> dict[str, Any]:
        return {"dut_id": dut_id, "stations": []}


@pytest.mark.asyncio
async def test_cached_catalog_helpers_use_supported_client_methods(monkeypatch):
    monkeypatch.setattr(dut_cache, "_cache", _PassthroughCache())
    client = _CatalogClient()

    assert await dut_cache.cached_get_site_by_id(client, 2) == {"id": 2, "name": "South"}
    assert await dut_cache.cached_get_models(client) == [
        {"id": 10, "name": "Model A"},
        {"id": 20, "name": "Model B"},
    ]
    assert await dut_cache.cached_get_model_by_id(client, 20) == {"id": 20, "name": "Model B"}
    assert await dut_cache.cached_get_stations(client) == [
        {"id": 100, "name": "Station A"},
        {"id": 200, "name": "Station B"},
    ]
    assert await dut_cache.cached_get_station_by_id(client, 200) == {"id": 200, "name": "Station B"}


@pytest.mark.asyncio
async def test_cached_dut_wrappers_use_supported_record_client_methods(monkeypatch):
    monkeypatch.setattr(dut_cache, "_cache", _PassthroughCache())
    client = _CatalogClient()

    records = await dut_cache.cached_get_dut_records(client, "DUT-1", site_id=1, model_id=10)
    summary = await dut_cache.cached_get_dut_summary(client, "DUT-2", site_id=1, model_id=10)

    assert records == {"dut_id": "DUT-1"}
    assert summary == {"dut_id": "DUT-2", "stations": []}
    assert client.record_requests == ["DUT-1"]
