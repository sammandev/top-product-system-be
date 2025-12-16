"""
Test PA sorting performance with hundreds of items.
Tests both single-key (frequency only) and dual-key (frequency + PA number) sorting.
"""

import re
import time


def extract_frequency(item: dict) -> int:
    """Extract 4-digit frequency from test_pattern."""
    pattern = item.get("test_pattern", "")
    match = re.search(r'_(\d{4})_', pattern)
    return int(match.group(1)) if match else 0


def extract_sort_key(item: dict) -> tuple[int, int]:
    """Extract frequency and PA number for dual-key sorting."""
    pattern = item.get("test_pattern", "")

    # Extract frequency (4 digits)
    freq_match = re.search(r'_(\d{4})_', pattern)
    frequency = int(freq_match.group(1)) if freq_match else 0

    # Extract PA number (1-4)
    pa_match = re.search(r'PA(\d)', pattern, re.IGNORECASE)
    pa_number = int(pa_match.group(1)) if pa_match else 0

    return (frequency, pa_number)


def main():
    # Generate realistic test data: 4 PAs x 200 frequencies x 2 test types = 1600 items
    items = []
    for pa in range(1, 5):  # PA1-PA4
        for freq in range(5000, 7000, 10):  # 200 frequencies
            for test in ['11AX_MCS9_B20', '11AX_MCS9_B40']:
                items.append({
                    'test_pattern': f'WiFi_PA{pa}_{freq}_{test}',
                    'adjusted_power_test_items': {
                        f'WiFi_PA{pa}_ADJUSTED_POW_MID_{freq}_{test}': 0.5,
                        f'WiFi_PA{pa}_ADJUSTED_POW_MEAN_{freq}_{test}': 0.48,
                    }
                })

    print(f"Dataset size: {len(items)} items")
    print(f"Memory size: ~{len(str(items)) / 1024:.1f} KB\n")

    # Test single-key sorting (frequency only)
    items_copy = items.copy()
    start = time.perf_counter()
    sorted_items_single = sorted(items_copy, key=extract_frequency)
    elapsed_single = (time.perf_counter() - start) * 1000

    # Test dual-key sorting (frequency + PA number)
    items_copy = items.copy()
    start = time.perf_counter()
    sorted_items_dual = sorted(items_copy, key=extract_sort_key)
    elapsed_dual = (time.perf_counter() - start) * 1000

    print("=" * 60)
    print("SINGLE-KEY SORTING (frequency only)")
    print("=" * 60)
    print(f"Sorting time: {elapsed_single:.2f}ms\n")
    print("First 5 items (same frequency, random PA order):")
    for i in range(5):
        freq = extract_frequency(sorted_items_single[i])
        pa = re.search(r'PA(\d)', sorted_items_single[i]['test_pattern']).group(1)
        print(f"  {sorted_items_single[i]['test_pattern'][:30]:<30} -> {freq} MHz, PA{pa}")

    print("\n" + "=" * 60)
    print("DUAL-KEY SORTING (frequency + PA number)")
    print("=" * 60)
    print(f"Sorting time: {elapsed_dual:.2f}ms")
    print(f"Performance difference: +{elapsed_dual - elapsed_single:.2f}ms ({((elapsed_dual - elapsed_single) / elapsed_single * 100):.1f}%)\n")
    print("First 5 items (sorted by freq, then PA1→PA4):")
    for i in range(5):
        key = extract_sort_key(sorted_items_dual[i])
        pa = re.search(r'PA(\d)', sorted_items_dual[i]['test_pattern']).group(1)
        print(f"  {sorted_items_dual[i]['test_pattern'][:30]:<30} -> {key[0]} MHz, PA{pa}")

    # Verify dual-key sorting correctness
    sort_keys = [extract_sort_key(item) for item in sorted_items_dual]
    is_sorted = all(sort_keys[i] <= sort_keys[i + 1] for i in range(len(sort_keys) - 1))
    print(f"\n✓ Correctly sorted: {is_sorted}")
    print(f"\nConclusion: Dual-key sorting adds only ~{elapsed_dual - elapsed_single:.2f}ms")
    print(f"            ({((elapsed_dual - elapsed_single) / elapsed_single * 100):.1f}% overhead) - NEGLIGIBLE!")


if __name__ == "__main__":
    main()
