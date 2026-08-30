"""
Tests for station and DUT ID resolution logic.

Verifies that endpoints can handle ISNs from different stations correctly
by using loose matching when exact ISN matching fails.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.routers import external_api_client as external_api_router
from app.routers.external_api_client import (
    _match_station_entry,
    _match_station_entry_loose,
)


class TestStationDUTResolution:
    """Test station and DUT ID resolution with cross-station ISNs."""

    def test_match_station_entry_exact_match_by_isn(self):
        """Should match when ISN belongs to the specified station."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": 12345,
                "dut_isn": "DM2527470036123",
                "data": [],
            }
        ]

        match = _match_station_entry(record_data, "Wireless_Test_6G", "DM2527470036123")

        assert match is not None
        assert match.station_id == 145
        assert match.station_dut_id == 12345
        assert match.station_dut_isn == "DM2527470036123"

    def test_match_station_entry_no_match_wrong_isn(self):
        """Should NOT match when ISN doesn't belong to the specified station."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": 12345,
                "dut_isn": "DM2527470036123",
                "data": [],
            },
            {
                "id": 146,
                "name": "Wireless_Test_5G",
                "dut_id": 67890,
                "dut_isn": "261534750003154",
                "data": [],
            },
        ]

        # Try to match station 145 with ISN from station 146
        match = _match_station_entry(record_data, "Wireless_Test_6G", "261534750003154")

        assert match is None  # Should not match

    def test_match_station_entry_loose_finds_any_dut(self):
        """Loose matching should find any DUT that tested on the station."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": 12345,
                "dut_isn": "DM2527470036123",
                "data": [],
            },
            {
                "id": 146,
                "name": "Wireless_Test_5G",
                "dut_id": 67890,
                "dut_isn": "261534750003154",
                "data": [],
            },
        ]

        # Loose match should find the station's DUT even with wrong ISN
        match = _match_station_entry_loose(record_data, "Wireless_Test_6G")

        assert match is not None
        assert match.station_id == 145
        assert match.station_dut_id == 12345
        assert match.station_dut_isn == "DM2527470036123"

    def test_match_station_entry_loose_by_station_id(self):
        """Loose matching should work with numeric station ID."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": 12345,
                "dut_isn": "DM2527470036123",
                "data": [],
            }
        ]

        match = _match_station_entry_loose(record_data, "145")

        assert match is not None
        assert match.station_id == 145
        assert match.station_dut_id == 12345

    def test_match_station_entry_case_insensitive(self):
        """Station name matching should be case-insensitive."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": 12345,
                "dut_isn": "DM2527470036123",
                "data": [],
            }
        ]

        match = _match_station_entry_loose(record_data, "wireless_test_6g")

        assert match is not None
        assert match.station_id == 145

    def test_match_station_entry_extracts_dut_from_data(self):
        """Should extract DUT ID from data array if not at station level."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": None,
                "dut_isn": None,
                "data": [
                    {
                        "dut_id": 12345,
                        "dut_id__isn": "DM2527470036123",
                    }
                ],
            }
        ]

        match = _match_station_entry_loose(record_data, "Wireless_Test_6G")

        assert match is not None
        assert match.station_id == 145
        assert match.station_dut_id == 12345
        assert match.station_dut_isn == "DM2527470036123"

    def test_match_station_entry_no_match_missing_station(self):
        """Should return None when station doesn't exist."""
        record_data = [
            {
                "id": 145,
                "name": "Wireless_Test_6G",
                "dut_id": 12345,
                "dut_isn": "DM2527470036123",
                "data": [],
            }
        ]

        match = _match_station_entry_loose(record_data, "NonExistentStation")

        assert match is None


@pytest.mark.asyncio
async def test_device_catalog_fetches_metadata_with_bounded_concurrency(monkeypatch):
    active_calls = 0
    max_active_calls = 0

    async def tracked(value):
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0.01)
        active_calls -= 1
        return value

    async def get_or_set(_key, loader, ttl):
        return await loader()

    async def get_sites(_client):
        return [{"id": 1}, {"id": 2}]

    async def get_models(_client, site_id):
        return await tracked([{"id": site_id * 10 + index} for index in range(3)])

    async def get_stations(_client, model_id):
        return await tracked([{"id": model_id * 10}])

    async def get_devices(_client, station_id):
        return await tracked([{"id": station_id, "name": f"device-{station_id}"}])

    monkeypatch.setattr(external_api_router, "DUT_CATALOG_MAX_PARALLEL_REQUESTS", 3)
    monkeypatch.setattr(external_api_router.dut_metadata_cache, "get_or_set", get_or_set)
    monkeypatch.setattr(external_api_router, "_get_cached_sites", get_sites)
    monkeypatch.setattr(external_api_router, "_get_cached_models", get_models)
    monkeypatch.setattr(external_api_router, "_get_cached_stations", get_stations)
    monkeypatch.setattr(external_api_router, "_get_cached_station_devices", get_devices)

    catalog = await external_api_router._get_device_catalog(SimpleNamespace(base_url="http://dut"))

    assert max_active_calls == 3
    assert len(catalog["by_id"]) == 6
    assert len(catalog["by_name"]) == 6


def test_match_station_entry_no_match_no_dut_id():
    """Should return None when DUT ID cannot be found."""
    record_data = [
        {
            "id": 145,
            "name": "Wireless_Test_6G",
            "dut_id": None,
            "dut_isn": None,
            "data": [],  # No DUT ID in data either
        }
    ]

    match = _match_station_entry_loose(record_data, "Wireless_Test_6G")

    assert match is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
