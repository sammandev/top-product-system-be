import asyncio

import pytest

from app.external_services import dut_api_client


@pytest.mark.asyncio
async def test_complete_dut_info_uses_bounded_concurrency(monkeypatch):
    active_calls = 0
    max_active_calls = 0

    async def tracked(value):
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0.01)
        active_calls -= 1
        return value

    client = object.__new__(dut_api_client.DUTAPIClient)
    client.base_url = "http://dut.test"

    async def get_records(_dut_id):
        return await tracked({"record_data": []})

    async def get_sites():
        return await tracked([{"id": 1, "name": "PTB"}, {"id": 2, "name": "TY"}])

    async def get_models(site_id):
        return await tracked([{"id": site_id * 10 + index, "name": f"model-{site_id}-{index}"} for index in range(2)])

    async def get_stations(model_id):
        return await tracked([{"id": model_id * 10, "name": f"station-{model_id}"}])

    monkeypatch.setattr(dut_api_client, "DUT_MAX_PARALLEL_REQUESTS", 2)
    monkeypatch.setattr(client, "get_dut_records", get_records)
    monkeypatch.setattr(client, "get_sites", get_sites)
    monkeypatch.setattr(client, "get_models_by_site", get_models)
    monkeypatch.setattr(client, "get_stations_by_model", get_stations)

    result = await client.get_complete_dut_info("ISN123")

    assert max_active_calls == 2
    assert len(result["models"]) == 4
    assert len(result["stations"]) == 4
    assert result["stations"][0]["site_name"] == "PTB"


def test_shared_client_configures_connection_retries_and_limits(monkeypatch):
    transport_options = {}

    class StubTransport:
        def __init__(self, **options):
            transport_options.update(options)

    class StubClient:
        def __init__(self, **_options):
            pass

    dut_api_client._shared_clients.clear()
    monkeypatch.setattr(dut_api_client.httpx, "AsyncHTTPTransport", StubTransport)
    monkeypatch.setattr(dut_api_client.httpx, "AsyncClient", StubClient)
    monkeypatch.setattr(dut_api_client, "DUT_CONNECT_RETRIES", 2)

    dut_api_client._get_shared_client("http://dut-retry.test")

    assert transport_options["retries"] == 2
    limits = transport_options["limits"]
    assert limits.max_connections == dut_api_client.DUT_MAX_CONNECTIONS
    assert limits.max_keepalive_connections == dut_api_client.DUT_MAX_KEEPALIVE_CONNECTIONS
    dut_api_client._shared_clients.clear()
