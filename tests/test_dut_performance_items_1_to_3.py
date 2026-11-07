import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache

from app.dependencies.authz import get_current_user
from app.external_services.dut_api_client import DUTAPIClient
from app.main import app
from app.routers import external_api_client as dut_router
from app.routers.external_api_client import get_user_dut_client
from app.services import dut_metadata_cache


@pytest.fixture(autouse=True)
def _clean_caches():
    dut_metadata_cache._local_cache.clear()
    dut_metadata_cache._locks.clear()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    dut_metadata_cache._local_cache.clear()
    dut_metadata_cache._locks.clear()


def test_fastapi_cache_hits_across_distinct_user_dut_client_instances(monkeypatch):
    """Item 1: @cache on GET /api/dut/sites must hit even when get_user_dut_client returns a new instance per request."""
    endpoint_calls = 0

    async def fake_get_cached_sites(_client):
        nonlocal endpoint_calls
        endpoint_calls += 1
        return [{"id": 1, "name": "PTB", "iplas_url": "http://iplas", "ftp_url": "ftp://ftp"}]

    monkeypatch.setattr(dut_router, "_get_cached_sites", fake_get_cached_sites)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1, username="tester", is_admin=False
    )
    # Return a brand-new DUTAPIClient instance on every request (mimicking get_user_dut_client)
    shared_base_url = f"http://dut-cache-test-{id(fake_get_cached_sites)}"
    app.dependency_overrides[get_user_dut_client] = lambda: DUTAPIClient(base_url=shared_base_url)

    with TestClient(app) as client:
        r1 = client.get("/api/dut/sites")
        r2 = client.get("/api/dut/sites")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
    assert endpoint_calls == 1, f"Expected @cache hit on second request, but endpoint ran {endpoint_calls} times"

    # Key builder must also preserve path for pattern invalidation
    key_builder = FastAPICache.get_key_builder()
    key = key_builder(
        dut_router.get_sites,
        "api-cache",
        args=(),
        kwargs={"client": DUTAPIClient(base_url=shared_base_url)},
    )
    assert isinstance(key, str)
    assert "get_sites" in key or "/sites" in key


@pytest.mark.asyncio
async def test_compute_top_product_fetches_station_records_in_parallel_with_bounded_concurrency(monkeypatch):
    """Item 2: _compute_top_product_response must fetch station records concurrently under DUT_CATALOG_MAX_PARALLEL_REQUESTS."""
    active_calls = 0
    max_active_calls = 0

    record_data = [
        {
            "id": 100 + i,
            "name": f"Station_{i}",
            "order": i,
            "model_id": 44,
            "model_name": "HH5K",
            "site_id": 2,
            "site_name": "PTB",
            "dut_isn": "ISN_PARALLEL",
            "dut_id": 500 + i,
            "data": [
                {
                    "id": 9000 + i,
                    "test_date": "2026-01-15T10:00:00Z",
                    "test_duration": 10.0,
                    "test_result": 1,
                    "error_item": "",
                    "device_id": 10 + i,
                    "device_id__name": f"DEV_{i}",
                    "dut_id": 500 + i,
                    "dut_id__isn": "ISN_PARALLEL",
                }
            ],
        }
        for i in range(6)
    ]

    class _ParallelStubClient(DUTAPIClient):
        def __init__(self) -> None:
            self.base_url = f"http://parallel-stub-{id(record_data)}"

        async def get_dut_records(self, dut_id: str) -> dict[str, Any]:
            return {"site_name": "PTB", "model_name": "HH5K", "record_data": record_data}

        async def get_station_records(self, station_id: int, dut_id: int) -> dict[str, Any]:
            nonlocal active_calls, max_active_calls
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            await asyncio.sleep(0.03)
            active_calls -= 1
            return {
                "record": [
                    {
                        "test_date": "2026-01-15T10:00:00Z",
                        "device": f"DEV_{station_id - 100}",
                        "station": f"Station_{station_id - 100}",
                        "isn": "ISN_PARALLEL",
                        "error_item": "",
                        "site_name": "PTB",
                    }
                ],
                "data": [["WiFi_TX1_POW_2412_11B", 23.0, 15.0, "19.0"]],
            }

    monkeypatch.setattr(dut_router, "DUT_CATALOG_MAX_PARALLEL_REQUESTS", 3)
    stub_client = _ParallelStubClient()

    result = await dut_router._compute_top_product_response(
        stub_client,
        None,
        "ISN_PARALLEL",
        stations=None,
        site_identifier=None,
        model_identifier=None,
        device_identifiers=None,
        test_item_filters=None,
        exclude_test_item_filters=None,
        station_filters_map=None,
        criteria_rules=None,
        criteria_label="default",
        include_pa_trends=False,
    )

    assert len(result.test_result) == 6
    assert max_active_calls > 1, "Expected parallel station record fetches, but calls ran sequentially"
    assert max_active_calls <= 3, f"Expected concurrency bounded by 3, got {max_active_calls}"


@pytest.mark.asyncio
async def test_station_records_and_pa_trends_cached_across_requests(monkeypatch):
    """Item 3: Station records, nonvalue records, and PA trends must be cached across requests."""
    station_record_calls = 0
    nonvalue_calls = 0
    pa_trend_calls = 0

    record_data = [
        {
            "id": 145,
            "name": "Wireless_Test_6G",
            "order": 1,
            "model_id": 44,
            "model_name": "HH5K",
            "site_id": 2,
            "site_name": "PTB",
            "dut_isn": "ISN_CACHE",
            "dut_id": 777,
            "data": [
                {
                    "id": 9100,
                    "test_date": "2026-01-15T10:00:00Z",
                    "test_duration": 12.0,
                    "test_result": 1,
                    "error_item": "",
                    "device_id": 11,
                    "device_id__name": "DEV_1",
                    "dut_id": 777,
                    "dut_id__isn": "ISN_CACHE",
                }
            ],
        }
    ]

    class _CountingStubClient(DUTAPIClient):
        def __init__(self) -> None:
            self.base_url = f"http://cache-stub-{id(record_data)}"

        async def get_dut_records(self, dut_id: str) -> dict[str, Any]:
            return {"site_name": "PTB", "model_name": "HH5K", "record_data": record_data}

        async def get_station_records(self, station_id: int, dut_id: int) -> dict[str, Any]:
            nonlocal station_record_calls
            station_record_calls += 1
            return {
                "record": [
                    {
                        "test_date": "2026-01-15T10:00:00Z",
                        "device": "DEV_1",
                        "station": "Wireless_Test_6G",
                        "isn": "ISN_CACHE",
                        "error_item": "",
                        "site_name": "PTB",
                    }
                ],
                "data": [["WiFi_TX1_POW_5985_11AX_MCS9_B80", 23.0, 15.0, "19.0"]],
            }

        async def get_latest_nonvalue_record(self, station_id: int, dut_id: int) -> dict[str, Any]:
            nonlocal nonvalue_calls
            nonvalue_calls += 1
            return {
                "record": [{"test_date": "2026-01-15T10:00:00Z", "device": "DEV_1", "error_item": ""}],
                "data": [
                    ["WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "11200"],
                    ["WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80", "11328"],
                ],
            }

        async def get_pa_test_items_trend(self, payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal pa_trend_calls
            pa_trend_calls += 1
            return {
                "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80": {"mid": 11200.0, "mean": 11200.0},
                "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80": {"mid": 11328.0, "mean": 11328.0},
            }

    stub_client = _CountingStubClient()

    for _ in range(2):
        await dut_router._compute_top_product_response(
            stub_client,
            None,
            "ISN_CACHE",
            stations=None,
            site_identifier=None,
            model_identifier=None,
            device_identifiers=None,
            test_item_filters=None,
            exclude_test_item_filters=None,
            station_filters_map=None,
            criteria_rules=None,
            criteria_label="default",
            include_pa_trends=True,
        )

    assert station_record_calls == 1, f"Expected station records cached across calls, got {station_record_calls}"
    assert nonvalue_calls == 1, f"Expected nonvalue records cached across calls, got {nonvalue_calls}"
    assert pa_trend_calls == 1, f"Expected PA trend data cached across calls, got {pa_trend_calls}"
