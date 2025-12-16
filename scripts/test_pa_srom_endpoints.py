"""
Test script for PA SROM hex-to-decimal conversion endpoints.

This script tests the helper functions for the new endpoints:
- /api/dut/pa/latest/decimal/{station_id}/{dut_id}
- /api/dut/pa/latest/decimal-old/{station_id}/{dut_id}
- /api/dut/pa/latest/decimal-new/{station_id}/{dut_id}
"""

import re
import sys


def _convert_hex_to_decimal(hex_value: str) -> int | None:
    """
    Convert hexadecimal string to decimal integer.

    Args:
        hex_value: Hexadecimal string (e.g., "0x235e" or "235e")

    Returns:
        Decimal integer or None if conversion fails
    """
    if not hex_value:
        return None

    try:
        # Remove whitespace and convert to string
        hex_str = str(hex_value).strip()

        # Handle empty string
        if not hex_str:
            return None

        # Convert hex to decimal (handles both "0x235e" and "235e" formats)
        if hex_str.lower().startswith("0x"):
            return int(hex_str, 16)
        else:
            # Assume it's hex without 0x prefix
            return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def _is_pa_srom_test_item(test_item_name: str, pattern_type: str = "all") -> bool:
    """
    Check if a test item name matches PA SROM patterns.

    Args:
        test_item_name: Test item name to check
        pattern_type: "old", "new", or "all" (default)

    Returns:
        True if test item matches the specified pattern type
    """
    if not test_item_name:
        return False

    name_upper = test_item_name.upper()

    # Check for PA{1-4}_SROM_OLD or PA{1-4}_SROM_NEW patterns
    has_old = bool(re.search(r"PA[1-4]_SROM_OLD", name_upper))
    has_new = bool(re.search(r"PA[1-4]_SROM_NEW", name_upper))

    if pattern_type == "old":
        return has_old
    elif pattern_type == "new":
        return has_new
    else:  # "all"
        return has_old or has_new


def test_hex_to_decimal_conversion():
    """Test hexadecimal to decimal conversion."""
    print("\n=== Testing Hex to Decimal Conversion ===")

    test_cases = [
        ("0x235e", 9054),
        ("0x244c", 9292),
        ("0x247e", 9342),
        ("235e", 9054),  # Without 0x prefix
        ("244c", 9292),  # Without 0x prefix
        ("0x237d", 9085),
        ("", None),
        (None, None),
        ("invalid", None),
    ]

    all_passed = True
    for hex_val, expected in test_cases:
        result = _convert_hex_to_decimal(hex_val)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"{status} {hex_val!r:15s} -> {result} (expected: {expected})")
        if not passed:
            all_passed = False

    return all_passed


def test_pa_srom_pattern_matching():
    """Test PA SROM test item pattern matching."""
    print("\n=== Testing PA SROM Pattern Matching ===")

    test_cases = [
        # (test_item, pattern_type, expected)
        ("WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "old", True),
        ("WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "new", False),
        ("WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "all", True),
        ("WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "old", False),
        ("WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "new", True),
        ("WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "all", True),
        ("WiFi_PA2_SROM_OLD_5300_11AX_MCS9_B80", "old", True),
        ("WiFi_PA3_SROM_NEW_6185_11AX_MCS11_B160", "new", True),
        ("WiFi_PA4_SROM_OLD_2437_11G_54M_B20", "all", True),
        ("WiFi_TX1_POW_2412_11B_CCK11_B20", "old", False),  # Not PA SROM
        ("WiFi_TX1_POW_2412_11B_CCK11_B20", "new", False),  # Not PA SROM
        ("WiFi_TX1_POW_2412_11B_CCK11_B20", "all", False),  # Not PA SROM
        ("", "all", False),
        (None, "all", False),
    ]

    all_passed = True
    for test_item, pattern_type, expected in test_cases:
        result = _is_pa_srom_test_item(test_item, pattern_type)
        passed = result == expected
        status = "✓" if passed else "✗"
        item_display = test_item[:40] + "..." if test_item and len(test_item) > 40 else test_item
        print(f"{status} {item_display!r:45s} | type={pattern_type:4s} -> {result} (expected: {expected})")
        if not passed:
            all_passed = False

    return all_passed


def test_combined_workflow():
    """Test combined workflow of filtering and conversion."""
    print("\n=== Testing Combined Workflow ===")

    # Simulate data from external API
    sample_data = [
        ["WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "0x237d", "0x244c", "0x235e"],
        ["WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "0x244c", "0x244c", "0x247e"],
        ["WiFi_PA2_SROM_OLD_5300_11AX_MCS9_B80", "0x1234", "0x5678", "0x9abc"],
        ["WiFi_TX1_POW_2412_11B_CCK11_B20", "15.5", "18.2", "17.8"],  # Not PA SROM
    ]

    print("\nOriginal data:")
    for row in sample_data:
        print(f"  {row}")

    # Filter and convert PA SROM items (all)
    print("\nFiltered PA SROM (all) with decimal conversion:")
    all_passed = True
    for row in sample_data:
        test_item = row[0]
        latest_hex = row[-1]

        if _is_pa_srom_test_item(test_item, "all"):
            decimal_val = _convert_hex_to_decimal(latest_hex)
            print(f"  ✓ {test_item[:40]:45s} -> {decimal_val}")
        else:
            print(f"  ✗ {test_item[:40]:45s} -> (skipped, not PA SROM)")

    # Filter PA SROM OLD only
    print("\nFiltered PA SROM_OLD only:")
    for row in sample_data:
        test_item = row[0]
        latest_hex = row[-1]

        if _is_pa_srom_test_item(test_item, "old"):
            decimal_val = _convert_hex_to_decimal(latest_hex)
            print(f"  ✓ {test_item[:40]:45s} -> {decimal_val}")

    # Filter PA SROM NEW only
    print("\nFiltered PA SROM_NEW only:")
    for row in sample_data:
        test_item = row[0]
        latest_hex = row[-1]

        if _is_pa_srom_test_item(test_item, "new"):
            decimal_val = _convert_hex_to_decimal(latest_hex)
            print(f"  ✓ {test_item[:40]:45s} -> {decimal_val}")

    return all_passed


def main():
    """Run all tests."""
    print("=" * 70)
    print("PA SROM Hex-to-Decimal Conversion Test Suite")
    print("=" * 70)

    results = []

    # Test 1: Hex to decimal conversion
    results.append(("Hex to Decimal Conversion", test_hex_to_decimal_conversion()))

    # Test 2: PA SROM pattern matching
    results.append(("PA SROM Pattern Matching", test_pa_srom_pattern_matching()))

    # Test 3: Combined workflow
    results.append(("Combined Workflow", test_combined_workflow()))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12s} - {test_name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 70))
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
