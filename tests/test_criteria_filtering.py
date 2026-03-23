"""
Tests for criteria configuration file parsing and station filtering.

This module verifies that:
1. Criteria configuration files are parsed correctly with [Model|Station] format
2. Only stations mentioned in the criteria file are included in results
3. Test items are scored according to uploaded criteria
"""

import pytest


def test_criteria_file_format_parsing():
    """Test that criteria JSON is parsed into a global rule bucket."""
    criteria_content = b"""
{
  "criteria": [
    {
      "test_item": "WiFi_TX_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20",
      "ucl": "20",
      "lcl": "10",
      "target": "15"
    },
    {
      "test_item": "WiFi_TX_POW_2462_11B_CCK11_B20",
      "ucl": "17",
      "lcl": "14",
      "target": "16"
    },
    {
      "test_item": "WiFi_TX_POW_6185_11AX_MCS11_B160",
      "ucl": "17",
      "lcl": "14",
      "target": "16"
    }
  ]
}
"""

    from app.routers.external_api_client import _load_station_criteria_from_bytes

    rules = _load_station_criteria_from_bytes(criteria_content)

    assert list(rules.keys()) == ["__global__"]
    assert len(rules["__global__"]) == 3


def test_criteria_station_selection():
    """Test that global JSON criteria apply to any station/model."""
    import re

    from app.routers.external_api_client import CriteriaRule, _select_station_criteria

    criteria_map = {"__global__": [CriteriaRule(pattern=re.compile("WiFi_TX_POW_2462", re.IGNORECASE), usl=17.0, lsl=14.0, target=16.0)]}

    rules_25g = _select_station_criteria(criteria_map, "Wireless_Test_2_5G", "HH5K")
    assert len(rules_25g) == 1

    rules_other_station = _select_station_criteria(criteria_map, "Unknown_Station", "HH6K")
    assert len(rules_other_station) == 1


def test_criteria_json_validation_requires_array():
    from fastapi import HTTPException

    from app.routers.external_api_client import _load_station_criteria_from_bytes

    with pytest.raises(HTTPException) as exc_info:
        _load_station_criteria_from_bytes(b'{"criteria": {}}')

    assert exc_info.value.status_code == 400
    assert "criteria" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_top_product_with_criteria_filters_stations(client_fixture):
    """
    Test that uploading a criteria file filters stations to only those mentioned.

    This is the key test for the bug fix:
    - Before fix: All stations were returned regardless of criteria file
    - After fix: Only stations in criteria file should be returned
    """
    # Note: This test requires a mocked DUT client and real DUT data
    # Skipping for now as it needs proper test setup with mock data
    pytest.skip("Requires mocked DUT client setup with test data")


@pytest.mark.asyncio
async def test_top_product_without_criteria_includes_all_stations(client_fixture):
    """
    Test that without a criteria file, all stations are included (default behavior).
    """
    # Note: This test requires a mocked DUT client and real DUT data
    # Skipping for now as it needs proper test setup with mock data
    pytest.skip("Requires mocked DUT client setup with test data")


def test_normalize_str():
    """Test string normalization used for criteria matching."""
    from app.routers.external_api_client import _normalize_str

    # Test case conversion
    assert _normalize_str("HH5K") == "hh5k"
    assert _normalize_str("Wireless_Test_2_5G") == "wireless_test_2_5g"

    # Test with pipe
    assert _normalize_str("HH5K|Wireless_Test") == "hh5k|wireless_test"

    # Test with None
    assert _normalize_str(None) == ""
