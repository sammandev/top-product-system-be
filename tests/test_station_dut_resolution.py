"""
Tests for station and DUT ID resolution logic.

Verifies that endpoints can handle ISNs from different stations correctly
by using loose matching when exact ISN matching fails.
"""

import pytest

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

    def test_match_station_entry_no_match_no_dut_id(self):
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
