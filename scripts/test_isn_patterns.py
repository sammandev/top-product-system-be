"""
Test ISN extraction from various filename formats.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.services.test_log_parser import TestLogParser


def test_all_isn_patterns():
    """Test ISN extraction from all supported filename patterns."""
    print("\n" + "=" * 70)
    print("TESTING ISN EXTRACTION - ALL FILENAME PATTERNS")
    print("=" * 70)

    test_cases = [
        # Pattern 1: [TestStation]_[ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
        {
            "filename": "Wireless_Test_6G_DM2524470075517_2025_11_20_105056100.txt",
            "expected_isn": "DM2524470075517",
            "pattern": "Pattern 1: [TestStation]_[ISN]_[Date]",
            "description": "Station with underscores, then ISN"
        },
        {
            "filename": "WIFI_6G_ABC123XYZ_2025_02_18_103203004.txt",
            "expected_isn": "ABC123XYZ",
            "pattern": "Pattern 1: [TestStation]_[ISN]_[Date]",
            "description": "Simple station name, alphanumeric ISN"
        },
        {
            "filename": "Station1_ISN001_2025_01_01_120000000.txt",
            "expected_isn": "ISN001",
            "pattern": "Pattern 1: [TestStation]_[ISN]_[Date]",
            "description": "Simple station, simple ISN"
        },
        
        # Pattern 2: [ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
        {
            "filename": "DM2527470012971_2025_11_20_172659650.txt",
            "expected_isn": "DM2527470012971",
            "pattern": "Pattern 2: [ISN]_[Date] (no station)",
            "description": "ISN directly followed by date"
        },
        {
            "filename": "ABC123_2025_01_15_093045123.txt",
            "expected_isn": "ABC123",
            "pattern": "Pattern 2: [ISN]_[Date] (no station)",
            "description": "Short ISN with date"
        },
        {
            "filename": "SN999888777_2025_12_31_235959999.txt",
            "expected_isn": "SN999888777",
            "pattern": "Pattern 2: [ISN]_[Date] (no station)",
            "description": "Numeric ISN with date"
        },
        
        # Pattern 3: [ISN]_[TestStation]_[YYYY]_[MM]_[DD]_[HHmmssmsec]
        {
            "filename": "DM2524470075517_Wireless_Test_6G_2025_11_20_105056100.txt",
            "expected_isn": "DM2524470075517",
            "pattern": "Pattern 3: [ISN]_[TestStation]_[Date]",
            "description": "ISN first, then station with underscores"
        },
        {
            "filename": "ABC123XYZ_WIFI_6G_2025_02_18_103203004.txt",
            "expected_isn": "ABC123XYZ",
            "pattern": "Pattern 3: [ISN]_[TestStation]_[Date]",
            "description": "ISN first, then station"
        },
        {
            "filename": "ISN001_Station1_2025_01_01_120000000.txt",
            "expected_isn": "ISN001",
            "pattern": "Pattern 3: [ISN]_[TestStation]_[Date]",
            "description": "Simple ISN, then station"
        },
        
        # Edge cases
        {
            "filename": "no_isn_pattern_here.txt",
            "expected_isn": None,
            "pattern": "Invalid",
            "description": "No valid pattern"
        },
        {
            "filename": "just_some_file.txt",
            "expected_isn": None,
            "pattern": "Invalid",
            "description": "Random filename"
        },
    ]

    passed = 0
    failed = 0

    for idx, test_case in enumerate(test_cases, 1):
        filename = test_case["filename"]
        expected_isn = test_case["expected_isn"]
        pattern = test_case["pattern"]
        description = test_case["description"]

        extracted_isn = TestLogParser.extract_isn_from_filename(filename)

        if extracted_isn == expected_isn:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1

        print(f"\n{status} Test {idx}: {pattern}")
        print(f"   Filename: {filename}")
        print(f"   Description: {description}")
        print(f"   Expected ISN: {expected_isn}")
        print(f"   Extracted ISN: {extracted_isn}")

        if extracted_isn != expected_isn:
            print(f"   ⚠️  MISMATCH!")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 70)

    if failed > 0:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All ISN extraction tests passed!")


def test_with_actual_content():
    """Test parsing with actual file content to verify ISN is included in response."""
    print("\n" + "=" * 70)
    print("TESTING ISN IN PARSE RESPONSE")
    print("=" * 70)

    test_content = """=========[Start SFIS Test Result]======
"TEST_ITEM_1" <100,50>  ===> "75.5"
"TEST_ITEM_2" <200,100>  ===> "150.0"
=========[End SFIS Test Result]======
"""

    test_filenames = [
        "Wireless_Test_6G_DM2524470075517_2025_11_20_105056100.txt",
        "DM2527470012971_2025_11_20_172659650.txt",
        "DM2524470075517_Wireless_Test_6G_2025_11_20_105056100.txt",
    ]

    expected_isns = [
        "DM2524470075517",
        "DM2527470012971",
        "DM2524470075517",
    ]

    for filename, expected_isn in zip(test_filenames, expected_isns):
        result = TestLogParser.parse_content(test_content, filename)

        print(f"\n✅ Filename: {filename}")
        print(f"   ISN in response: {result['isn']}")
        print(f"   Expected ISN: {expected_isn}")
        print(f"   Parsed items: {result['parsed_count']}")

        assert result['isn'] == expected_isn, f"ISN mismatch! Expected {expected_isn}, got {result['isn']}"

    print("\n✅ All parse response tests passed!")


def main():
    """Run all ISN extraction tests."""
    try:
        test_all_isn_patterns()
        test_with_actual_content()

        print("\n" + "=" * 70)
        print("✅ ALL ISN EXTRACTION TESTS PASSED!")
        print("=" * 70)
        print("\nSupported filename patterns:")
        print("  1. [TestStation]_[ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]")
        print("  2. [ISN]_[YYYY]_[MM]_[DD]_[HHmmssmsec]")
        print("  3. [ISN]_[TestStation]_[YYYY]_[MM]_[DD]_[HHmmssmsec]")
        print("\nNote: ISN never contains underscore character")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
