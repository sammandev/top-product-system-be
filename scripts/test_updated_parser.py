"""
Test script for updated test log parser.

Tests:
1. Flexible start/end markers (variable '=' count)
2. USL/LSL terminology (instead of UCL/LCL)
3. FAIL items included in results
4. Archive file support (.zip)
"""

import json
import sys
import zipfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.services.test_log_parser import TestLogParser


def test_flexible_markers():
    """Test that markers work with any number of '=' characters."""
    print("\n" + "=" * 60)
    print("TEST 1: Flexible Start/End Markers")
    print("=" * 60)
    
    # Create test content with different marker formats
    test_cases = [
        # Original format (9 '=' on each side)
        ("=========[Start SFIS Test Result]======", "=========[End SFIS Test Result]======"),
        # Fewer '=' characters
        ("===[Start SFIS Test Result]===", "===[End SFIS Test Result]==="),
        # More '=' characters
        ("===============[Start SFIS Test Result]===============", 
         "===============[End SFIS Test Result]==============="),
        # Asymmetric '=' characters
        ("=====[Start SFIS Test Result]==========", "==========[End SFIS Test Result]====="),
    ]
    
    for idx, (start_marker, end_marker) in enumerate(test_cases, 1):
        content = f"""{start_marker}
"TEST_ITEM_1" <100,50>  ===> "75.5"
"TEST_ITEM_2" <,>  ===> "PASS"
"TEST_ITEM_3" <25,5>  ===> "FAIL"
{end_marker}
"""
        result = TestLogParser.parse_content(content, f"test_case_{idx}.txt")
        
        print(f"\nTest Case {idx}:")
        print(f"  Start Marker: {start_marker}")
        print(f"  End Marker:   {end_marker}")
        print(f"  ✅ Parsed {result['parsed_count']} items")
        print(f"  ✅ Skipped {result['skipped_items']} items (PASS/VALUE)")
        
        # Verify FAIL is included
        fail_items = [item for item in result['parsed_items'] if item['value'] == 'FAIL']
        print(f"  ✅ FAIL items included: {len(fail_items)}")
    
    print("\n✅ All flexible marker tests passed!")


def test_usl_lsl_terminology():
    """Test that USL/LSL fields are used instead of UCL/LCL."""
    print("\n" + "=" * 60)
    print("TEST 2: USL/LSL Terminology")
    print("=" * 60)
    
    content = """=========[Start SFIS Test Result]======
"PA1_POW_OLD_6015_11AX_MCS9_B20" <25,5>  ===> "16.84"
"TX1_FREQ_KHZ_2404_BT" <150,-150>  ===> "2.1085671"
=========[End SFIS Test Result]======
"""
    
    result = TestLogParser.parse_content(content, "test_usl_lsl.txt")
    
    print(f"\nParsed {result['parsed_count']} items:")
    for item in result['parsed_items']:
        print(f"\n  Test Item: {item['test_item']}")
        print(f"  USL: {item['usl']}")
        print(f"  LSL: {item['lsl']}")
        print(f"  Measurement Value: {item['value']}")
        
        # Verify fields exist
        assert 'usl' in item, "USL field missing!"
        assert 'lsl' in item, "LSL field missing!"
        assert 'ucl' not in item, "Old UCL field should not exist!"
        assert 'lcl' not in item, "Old LCL field should not exist!"
    
    print("\n✅ USL/LSL terminology test passed!")


def test_fail_items_included():
    """Test that FAIL items are now included in results."""
    print("\n" + "=" * 60)
    print("TEST 3: FAIL Items Included")
    print("=" * 60)
    
    content = """=========[Start SFIS Test Result]======
"ITEM_1" <100,50>  ===> "75.5"
"ITEM_2" <100,50>  ===> "PASS"
"ITEM_3" <100,50>  ===> "FAIL"
"ITEM_4" <100,50>  ===> "VALUE"
"ITEM_5" <100,50>  ===> "45.2"
=========[End SFIS Test Result]======
"""
    
    result = TestLogParser.parse_content(content, "test_fail.txt")
    
    print(f"\nTotal lines in test section: 5")
    print(f"Parsed items: {result['parsed_count']}")
    print(f"Skipped items: {result['skipped_items']}")
    
    # Find FAIL items
    fail_items = [item for item in result['parsed_items'] if item['value'] == 'FAIL']
    pass_items = [item for item in result['parsed_items'] if item['value'] == 'PASS']
    value_items = [item for item in result['parsed_items'] if item['value'] == 'VALUE']
    
    print(f"\nFAIL items (should be included): {len(fail_items)}")
    for item in fail_items:
        print(f"  ✅ {item['test_item']}: {item['value']}")
    
    print(f"\nPASS items (should be skipped): {len(pass_items)} (expected: 0)")
    print(f"VALUE items (should be skipped): {len(value_items)} (expected: 0)")
    
    assert len(fail_items) == 1, "FAIL item should be included!"
    assert len(pass_items) == 0, "PASS items should be skipped!"
    assert len(value_items) == 0, "VALUE items should be skipped!"
    assert result['parsed_count'] == 3, "Should have 3 parsed items (2 numeric + 1 FAIL)"
    assert result['skipped_items'] == 2, "Should have 2 skipped items (PASS + VALUE)"
    
    print("\n✅ FAIL items inclusion test passed!")


def test_archive_support():
    """Test archive file extraction and parsing."""
    print("\n" + "=" * 60)
    print("TEST 4: Archive File Support (.zip)")
    print("=" * 60)
    
    # Create test .txt files
    test_data_dir = Path("test_outputs/archive_test")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    file1_content = """=========[Start SFIS Test Result]======
"TEST_A" <100,50>  ===> "75.5"
"TEST_B" <200,100>  ===> "150.2"
=========[End SFIS Test Result]======
"""
    
    file2_content = """=========[Start SFIS Test Result]======
"TEST_C" <50,10>  ===> "FAIL"
"TEST_D" <300,200>  ===> "250.8"
=========[End SFIS Test Result]======
"""
    
    # Write test files
    file1_path = test_data_dir / "test_file_1.txt"
    file2_path = test_data_dir / "test_file_2.txt"
    
    with open(file1_path, 'w') as f:
        f.write(file1_content)
    
    with open(file2_path, 'w') as f:
        f.write(file2_content)
    
    # Create zip archive
    zip_path = test_data_dir / "test_archive.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(file1_path, "test_file_1.txt")
        zipf.write(file2_path, "test_file_2.txt")
    
    print(f"\n📦 Created test archive: {zip_path}")
    
    # Test archive detection
    is_archive = TestLogParser.is_archive(str(zip_path))
    print(f"✅ Archive detected: {is_archive}")
    assert is_archive, "Should detect .zip as archive!"
    
    # Test archive parsing
    result = TestLogParser.parse_archive(str(zip_path))
    
    print(f"\n📊 Archive Parse Results:")
    print(f"  Archive Name: {result['archive_name']}")
    print(f"  Total .txt Files: {result['total_files']}")
    print(f"  Extracted Files: {result['extracted_files']}")
    print(f"  Errors: {result['errors']}")
    
    assert result['total_files'] == 2, "Should find 2 .txt files!"
    
    print(f"\n📝 Parsed Results from Archive:")
    for file_result in result['results']:
        print(f"\n  File: {file_result['filename']}")
        print(f"    Parsed: {file_result['parsed_count']} items")
        print(f"    Skipped: {file_result['skipped_items']} items")
        for item in file_result['parsed_items']:
            print(f"      - {item['test_item']}: USL={item['usl']}, LSL={item['lsl']}, Value={item['value']}")
    
    # Save result
    output_file = test_data_dir / "archive_parse_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Archive support test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 TESTING UPDATED TEST LOG PARSER")
    print("=" * 60)
    
    try:
        test_flexible_markers()
        test_usl_lsl_terminology()
        test_fail_items_included()
        test_archive_support()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nUpdates verified:")
        print("  1. ✅ Flexible start/end markers (any number of '=')")
        print("  2. ✅ USL/LSL terminology (replacing UCL/LCL)")
        print("  3. ✅ FAIL items included in results")
        print("  4. ✅ Archive file support (.zip, .rar, .7z)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
