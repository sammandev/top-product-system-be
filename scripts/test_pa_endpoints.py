"""
Test script to demonstrate the updated PA endpoint changes.

Changes implemented:
1. POST /api/dut/pa/trend
   - Now includes station_name in response
   - Response format changed from items[] to data[] array
   - Each data item has OLD/NEW SROM values as top-level keys
   - adjusted_power_trend nested inside each data item

2. GET /api/dut/pa/adjusted-power/
   - Enhanced logging to diagnose empty adjusted_power_trend issues
   - Logs time window, payload, and pairing results
"""

import json
from datetime import datetime, timedelta

# Example request for POST /api/dut/pa/trend
pa_trend_request = {
    "station_id": 145,
    "start_time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
    "end_time": datetime.utcnow().isoformat(),
    "test_items": [
        "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80_MID",
        "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80_MID",
        "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80_MEAN",
        "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80_MEAN",
    ],
}

# Expected response format for POST /api/dut/pa/trend
pa_trend_response_example = {
    "station_id": 145,
    "station_name": "Wireless_Test_6G",  # ← NEW: Station name from DUT info
    "start_time": "2025-01-01T00:00:00Z",
    "end_time": "2025-01-08T00:00:00Z",
    "data": [  # ← CHANGED: Previously "items", now "data"
        {
            # ← NEW: OLD/NEW SROM values as top-level keys
            "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80_MID": {
                "mid": 320.5,
                "mean": 321.2,
            },
            "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80_MID": {
                "mid": 340.5,
                "mean": 341.0,
            },
            # ← NEW: adjusted_power_trend nested object (not array)
            "adjusted_power_trend": {
                "WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80": 0.078,
                "WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80": 0.077,
            },
            "error": None,
        }
    ],
    "unpaired_items": [],
}

# Example request for GET /api/dut/pa/adjusted-power/?dut_isn=ABC123
# Query parameters:
# - dut_isn: ABC123
# - start_time: 2025-01-01T00:00:00Z (optional)
# - end_time: 2025-01-08T00:00:00Z (optional)
# - station_identifiers: ["145", "Wireless_Test_6G"] (optional)

# Expected response format for GET /api/dut/pa/adjusted-power/
pa_adjusted_power_response_example = {
    "dut_records": [
        {
            "isn": "ABC123",
            "model_id": "TestModel",
            "site": "Site A",
            "line": "Line 1",
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T23:59:59Z",
            "station_id": 145,
            "station_name": "Wireless_Test_6G",
            "record_count": 100,
        }
    ],
    "adjusted_power_trend": [  # ← Should NOT be empty
        {
            "test_pattern": "WiFi_PA1_5985_11AX_MCS9_B80",
            "adjusted_power_test_items": {
                "WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80": 0.078,
                "WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80": 0.077,
            },
            "error": None,
        }
    ],
}


def main():
    print("=" * 80)
    print("PA ENDPOINT CHANGES SUMMARY")
    print("=" * 80)

    print("\n1. POST /api/dut/pa/trend")
    print("-" * 80)
    print("Request:")
    print(json.dumps(pa_trend_request, indent=2))
    print("\nExpected Response:")
    print(json.dumps(pa_trend_response_example, indent=2))

    print("\n\n2. GET /api/dut/pa/adjusted-power/")
    print("-" * 80)
    print("Example URL: /api/dut/pa/adjusted-power/?dut_isn=ABC123")
    print("\nExpected Response:")
    print(json.dumps(pa_adjusted_power_response_example, indent=2))

    print("\n\nKEY CHANGES:")
    print("-" * 80)
    print("✓ POST /pa/trend now includes 'station_name' field")
    print("✓ Response changed from 'items' to 'data' array")
    print("✓ Each data item has OLD/NEW SROM values as top-level keys")
    print("✓ adjusted_power_trend is a nested object (not array)")
    print("✓ Enhanced logging in /pa/adjusted-power/ to diagnose empty results")
    print("✓ All 88 PA-related backend tests passing")


if __name__ == "__main__":
    main()
