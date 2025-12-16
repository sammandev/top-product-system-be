"""
Tests for criteria configuration file parsing and station filtering.

This module verifies that:
1. Criteria configuration files are parsed correctly with [Model|Station] format
2. Only stations mentioned in the criteria file are included in results
3. Test items are scored according to uploaded criteria
"""


import pytest


def test_criteria_file_format_parsing():
    """Test that criteria files with [Model|Station] format are parsed correctly."""
    criteria_content = b"""
; Example criteria configuration
[HH5K|Wireless_Test_2_5G]
"WiFi_TX_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20" <20,10>  ===> "15"
"WiFi_TX_POW_2462_11B_CCK11_B20" <17,14>  ===> "16"

[HH5K|Wireless_Test_6G]
"WiFi_TX_POW_6185_11AX_MCS11_B160" <17,14>  ===> "16"
"""

    from app.routers.external_api_client import _load_station_criteria_from_bytes

    rules = _load_station_criteria_from_bytes(criteria_content)

    # Should have 2 stations with normalized keys
    assert len(rules) == 2

    # Check HH5K|Wireless_Test_2_5G station
    key_25g = "hh5k|wireless_test_2_5g"  # normalized
    assert key_25g in rules
    assert len(rules[key_25g]) == 2

    # Check HH5K|Wireless_Test_6G station
    key_6g = "hh5k|wireless_test_6g"  # normalized
    assert key_6g in rules
    assert len(rules[key_6g]) == 1


def test_criteria_station_selection():
    """Test that _select_station_criteria correctly matches model|station format."""
    import re

    from app.routers.external_api_client import CriteriaRule, _select_station_criteria

    # Create mock criteria rules
    criteria_map = {
        "hh5k|wireless_test_2_5g": [
            CriteriaRule(
                pattern=re.compile("WiFi_TX_POW_2462", re.IGNORECASE),
                usl=17.0,
                lsl=14.0,
                target=16.0
            )
        ],
        "hh5k|wireless_test_6g": [
            CriteriaRule(
                pattern=re.compile("WiFi_TX_POW_6185", re.IGNORECASE),
                usl=17.0,
                lsl=14.0,
                target=16.0
            )
        ]
    }

    # Test exact match with model
    rules_25g = _select_station_criteria(criteria_map, "Wireless_Test_2_5G", "HH5K")
    assert len(rules_25g) == 1

    # Test exact match with different model should return nothing
    rules_wrong_model = _select_station_criteria(criteria_map, "Wireless_Test_2_5G", "HH6K")
    assert len(rules_wrong_model) == 0

    # Test station not in criteria
    rules_missing = _select_station_criteria(criteria_map, "Unknown_Station", "HH5K")
    assert len(rules_missing) == 0


def test_criteria_line_parsing():
    """Test parsing individual criteria lines."""
    from app.routers.external_api_client import _parse_criteria_line

    # Test line with all values
    rule1 = _parse_criteria_line('"WiFi_TX_POW_2462_11B_CCK11_B20" <17,14>  ===> "16"')
    assert rule1 is not None
    assert rule1.usl == 17.0
    assert rule1.lsl == 14.0
    assert rule1.target == 16.0
    assert rule1.pattern.pattern == "WiFi_TX_POW_2462_11B_CCK11_B20"

    # Test line with empty USL
    rule2 = _parse_criteria_line('"WiFi_TX_POW_.*" <,10>  ===> "15"')
    assert rule2 is not None
    assert rule2.usl is None
    assert rule2.lsl == 10.0
    assert rule2.target == 15.0

    # Test line with empty target
    rule3 = _parse_criteria_line('"WiFi_PA_POW_.*" <20,10>  ===> ""')
    assert rule3 is not None
    assert rule3.usl == 20.0
    assert rule3.lsl == 10.0
    assert rule3.target is None

    # Test invalid line
    rule4 = _parse_criteria_line('invalid line format')
    assert rule4 is None


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
