"""
Quick test to verify PA SROM sorting logic.
"""
import re


def _get_pa_srom_sort_key(test_item_name: str) -> tuple:
    """
    Extract sort key from PA SROM test item name for ordering.

    Sorts by: 1) Frequency (ascending), 2) Antenna number (PA1-4), 3) SROM type (OLD before NEW), 4) Name

    Args:
        test_item_name: Test item name (e.g., "WiFi_PA2_SROM_NEW_5985_11AX_MCS9_B80")

    Returns:
        Tuple of (frequency, antenna_num, srom_priority, test_item_name)
    """
    name_upper = test_item_name.upper()

    # Extract frequency (e.g., "2412", "5985") - default to 0 if not found
    frequency = 0
    freq_match = re.search(r"_(\d{4})_", test_item_name)
    if freq_match:
        frequency = int(freq_match.group(1))

    # Extract antenna number (PA1, PA2, PA3, PA4) - default to 99 if not found
    antenna_num = 99
    pa_match = re.search(r"PA([1-4])_SROM", name_upper)
    if pa_match:
        antenna_num = int(pa_match.group(1))

    # SROM type priority: OLD (0) before NEW (1)
    srom_priority = 0 if "_SROM_OLD" in name_upper else 1

    return (frequency, antenna_num, srom_priority, test_item_name)


def test_sorting():
    """Test the sorting logic with sample data."""
    
    # Sample unsorted data (as from your example)
    unsorted_items = [
        {"test_item_name": "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "mid": 10960, "mean": 10959},
        {"test_item_name": "WiFi_PA2_SROM_OLD_5985_11AX_MCS9_B80", "mid": 11209, "mean": 11212},
        {"test_item_name": "WiFi_PA2_SROM_NEW_5985_11AX_MCS9_B80", "mid": 11319, "mean": 11318},
        {"test_item_name": "WiFi_PA2_SROM_NEW_5985_11AX_MCS9_B40", "mid": 11319, "mean": 11318},
        {"test_item_name": "WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "mid": 9054, "mean": 9053},
        {"test_item_name": "WiFi_PA3_SROM_OLD_2412_11B_CCK11_B20", "mid": 9100, "mean": 9099},
        {"test_item_name": "WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "mid": 9000, "mean": 8999},
        {"test_item_name": "WiFi_PA4_SROM_NEW_5985_11AX_MCS9_B80", "mid": 11400, "mean": 11399},
    ]
    
    # Sort using the same logic as the endpoint
    sorted_items = sorted(unsorted_items, key=lambda item: _get_pa_srom_sort_key(item["test_item_name"]))
    
    print("=" * 80)
    print("SORTED PA SROM ITEMS (Frequency → Antenna → SROM Type → Name)")
    print("=" * 80)
    
    for idx, item in enumerate(sorted_items, 1):
        name = item["test_item_name"]
        key = _get_pa_srom_sort_key(name)
        print(f"{idx:2d}. {name}")
        print(f"    Sort Key: freq={key[0]}, antenna=PA{key[1]}, srom_type={'OLD' if key[2]==0 else 'NEW'}")
        print(f"    Values: mid={item['mid']}, mean={item['mean']}")
        print()
    
    # Verify sorting order
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    expected_order = [
        "WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20",  # freq=2412, PA1, OLD
        "WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20",  # freq=2412, PA1, NEW
        "WiFi_PA3_SROM_OLD_2412_11B_CCK11_B20",  # freq=2412, PA3, OLD
        "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80",  # freq=5985, PA1, OLD
        "WiFi_PA2_SROM_OLD_5985_11AX_MCS9_B80",  # freq=5985, PA2, OLD
        "WiFi_PA2_SROM_NEW_5985_11AX_MCS9_B40",  # freq=5985, PA2, NEW
        "WiFi_PA2_SROM_NEW_5985_11AX_MCS9_B80",  # freq=5985, PA2, NEW
        "WiFi_PA4_SROM_NEW_5985_11AX_MCS9_B80",  # freq=5985, PA4, NEW
    ]
    
    actual_order = [item["test_item_name"] for item in sorted_items]
    
    if actual_order == expected_order:
        print("✅ Sorting is CORRECT!")
        print("\nOrder: Frequency (low→high) → Antenna (PA1→PA4) → SROM (OLD→NEW) → Name (A→Z)")
    else:
        print("❌ Sorting mismatch!")
        print("\nExpected:")
        for name in expected_order:
            print(f"  - {name}")
        print("\nActual:")
        for name in actual_order:
            print(f"  - {name}")
    
    print("=" * 80)


if __name__ == "__main__":
    test_sorting()
