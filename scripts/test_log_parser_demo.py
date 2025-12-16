"""
Demonstration script for Test Log Parser endpoints.

This script shows how to use the new /api/test-log endpoints to:
1. Parse a single test log file
2. Compare multiple test log files

Prerequisites:
- Backend server must be running (make dev or uvicorn)
- Sample data file must exist: backend_fastapi/data/sample_data/WIFI_6G_000000000000018_2025_02_18_103203004.txt
"""

import json
from pathlib import Path

import requests

# Configuration
BASE_URL = "http://localhost:8000"
PARSE_ENDPOINT = f"{BASE_URL}/api/test-log/parse"
COMPARE_ENDPOINT = f"{BASE_URL}/api/test-log/compare"
HEALTH_ENDPOINT = f"{BASE_URL}/api/test-log/health"

# Sample file path
SAMPLE_FILE = Path("data/sample_data/WIFI_6G_000000000000018_2025_02_18_103203004.txt")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}\n")


def test_health_check():
    """Test the health check endpoint."""
    print_section("Testing Health Check Endpoint")
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print("✅ Health Check Response:")
        print(json.dumps(data, indent=2))
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_parse_single_file():
    """Test parsing a single test log file."""
    print_section("Testing Single File Parsing")
    
    if not SAMPLE_FILE.exists():
        print(f"❌ Sample file not found: {SAMPLE_FILE}")
        return False
    
    try:
        with open(SAMPLE_FILE, 'rb') as f:
            files = {'file': (SAMPLE_FILE.name, f, 'text/plain')}
            response = requests.post(PARSE_ENDPOINT, files=files, timeout=10)
        
        response.raise_for_status()
        data = response.json()
        
        print("✅ Parse Response Summary:")
        print(f"  Filename: {data['filename']}")
        print(f"  Total Lines: {data['total_lines']}")
        print(f"  Test Section Lines: {data['test_section_lines']}")
        print(f"  Parsed Items: {data['parsed_count']}")
        print(f"  Skipped Items: {data['skipped_items']}")
        print(f"  Errors: {len(data['errors'])}")
        
        if data['errors']:
            print("\n⚠️ Parsing Errors:")
            for error in data['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")
        
        # Show first 5 parsed items
        print("\n📊 Sample Parsed Items (first 5):")
        for item in data['parsed_items'][:5]:
            print(f"  - {item['test_item']}")
            print(f"    UCL: {item['ucl']}, LCL: {item['lcl']}")
            print(f"    Value: {item['value']}")
            print()
        
        # Save full response to file for inspection
        output_file = Path("test_outputs/parse_response.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Full response saved to: {output_file}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Parse test failed: {e}")
        return False


def test_compare_files():
    """Test comparing multiple test log files."""
    print_section("Testing File Comparison")
    
    if not SAMPLE_FILE.exists():
        print(f"❌ Sample file not found: {SAMPLE_FILE}")
        return False
    
    # For demonstration, we'll compare the same file with itself
    # In practice, you would use different files
    try:
        with open(SAMPLE_FILE, 'rb') as f1, open(SAMPLE_FILE, 'rb') as f2:
            files = [
                ('files', (SAMPLE_FILE.name, f1, 'text/plain')),
                ('files', (f"copy_{SAMPLE_FILE.name}", f2, 'text/plain'))
            ]
            response = requests.post(COMPARE_ENDPOINT, files=files, timeout=15)
        
        response.raise_for_status()
        data = response.json()
        
        print("✅ Compare Response Summary:")
        print(f"  Files Compared: {data['total_files']}")
        print(f"  Total Unique Items: {data['total_unique_items']}")
        print(f"  Common Items: {data['common_items_count']}")
        
        # Show file summaries
        print("\n📁 File Summaries:")
        for summary in data['file_summaries']:
            print(f"  - {summary['filename']}")
            print(f"    Parsed: {summary['parsed_count']}, Skipped: {summary['skipped_items']}")
        
        # Show sample comparison data
        print("\n🔍 Sample Comparison Items (first 5 common items):")
        common_items = [item for item in data['comparison_data'] if item['is_common']]
        for item in common_items[:5]:
            print(f"  - {item['test_item']}")
            print(f"    UCL: {item['ucl']}, LCL: {item['lcl']}")
            print(f"    Values:")
            for val in item['values']:
                print(f"      {val['filename']}: {val['value']}")
            
            # Show numeric analysis if available
            if item.get('min_value') is not None:
                print(f"    Numeric Analysis:")
                print(f"      Min: {item['min_value']}, Max: {item['max_value']}")
                print(f"      Range: {item['range']}, Avg: {item['avg_value']:.2f}")
            print()
        
        # Save full response to file for inspection
        output_file = Path("test_outputs/compare_response.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Full response saved to: {output_file}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Compare test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" Test Log Parser API - Demonstration Script")
    print("=" * 80)
    print(f"\nBase URL: {BASE_URL}")
    print(f"Sample File: {SAMPLE_FILE}")
    
    # Run tests
    results = {
        "Health Check": test_health_check(),
        "Single File Parsing": test_parse_single_file(),
        "File Comparison": test_compare_files()
    }
    
    # Summary
    print_section("Test Results Summary")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'🎉 All tests passed!' if all_passed else '⚠️ Some tests failed.'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
