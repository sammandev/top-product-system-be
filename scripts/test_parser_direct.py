"""
Direct test of the TestLogParser service (no API calls needed).

This script tests the parser directly to verify functionality.
"""

from pathlib import Path
import json

# Import the parser service directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.services.test_log_parser import TestLogParser


def test_parser_directly():
    """Test the parser service directly without HTTP requests."""
    print("\n" + "=" * 80)
    print(" Test Log Parser - Direct Service Test")
    print("=" * 80)

    # Test file path
    sample_file = Path("data/sample_data/WIFI_6G_000000000000018_2025_02_18_103203004.txt")

    if not sample_file.exists():
        print(f"\n❌ Sample file not found: {sample_file}")
        return False

    print(f"\n📄 Testing with file: {sample_file.name}")

    # Test 1: Parse single file
    print("\n" + "-" * 80)
    print(" Test 1: Parse Single File")
    print("-" * 80)

    try:
        result = TestLogParser.parse_file(str(sample_file))

        print(f"\n✅ Parse Results:")
        print(f"  Filename: {result['filename']}")
        print(f"  Total Lines: {result['total_lines']}")
        print(f"  Test Section Lines: {result['test_section_lines']}")
        print(f"  Parsed Items: {result['parsed_count']}")
        print(f"  Skipped Items: {result['skipped_items']}")
        print(f"  Errors: {len(result['errors'])}")

        # Show sample items
        print(f"\n📊 Sample Parsed Items (first 10):")
        for idx, item in enumerate(result['parsed_items'][:10], 1):
            print(f"  {idx}. {item['test_item']}")
            print(f"     UCL={item['ucl']}, LCL={item['lcl']}, Value={item['value']}")

        # Show items with numeric values
        print(f"\n🔢 Sample Items with Numeric Values:")
        numeric_items = []
        for item in result['parsed_items']:
            try:
                float(item['value'])
                numeric_items.append(item)
                if len(numeric_items) >= 5:
                    break
            except (ValueError, TypeError):
                continue

        for idx, item in enumerate(numeric_items, 1):
            print(f"  {idx}. {item['test_item']}: {item['value']}")

        # Show items that were skipped (would have been PASS/FAIL/VALUE)
        print(f"\n⏭️  Items Skipped: {result['skipped_items']} (PASS/FAIL/VALUE values)")

        # Save to file
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "direct_parse_test.json"

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n💾 Full results saved to: {output_file}")

        test1_passed = result['parsed_count'] > 0

    except Exception as e:
        print(f"\n❌ Test 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        test1_passed = False

    # Test 2: Compare two files (using same file twice for demo)
    print("\n" + "-" * 80)
    print(" Test 2: Compare Files")
    print("-" * 80)

    try:
        # For demo, compare file with itself
        file_paths = [str(sample_file), str(sample_file)]

        result = TestLogParser.compare_files(file_paths)

        print(f"\n✅ Comparison Results:")
        print(f"  Files Compared: {result['total_files']}")
        print(f"  Total Unique Items: {result['total_unique_items']}")
        print(f"  Common Items: {result['common_items_count']}")

        # Show file summaries
        print(f"\n📁 File Summaries:")
        for summary in result['file_summaries']:
            print(f"  - {summary['filename']}")
            print(f"    Parsed: {summary['parsed_count']}, Skipped: {summary['skipped_items']}")

        # Show sample common items with numeric analysis
        print(f"\n🔍 Sample Common Items with Numeric Values:")
        common_numeric = [
            item for item in result['comparison_data']
            if item['is_common'] and item.get('min_value') is not None
        ]

        for idx, item in enumerate(common_numeric[:5], 1):
            print(f"  {idx}. {item['test_item']}")
            print(f"     UCL={item['ucl']}, LCL={item['lcl']}")
            print(f"     Min={item['min_value']}, Max={item['max_value']}, Avg={item['avg_value']:.4f}")
            print(f"     Range={item['range']}")

        # Save to file
        output_file = output_dir / "direct_compare_test.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n💾 Full comparison saved to: {output_file}")

        test2_passed = result['common_items_count'] > 0

    except Exception as e:
        print(f"\n❌ Test 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        test2_passed = False

    # Summary
    print("\n" + "=" * 80)
    print(" Test Summary")
    print("=" * 80)
    print(f"  Test 1 (Parse): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Compare): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    all_passed = test1_passed and test2_passed
    print(f"\n{'🎉 All tests passed!' if all_passed else '⚠️ Some tests failed.'}\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = test_parser_directly()
    sys.exit(0 if success else 1)
