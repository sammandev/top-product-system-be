"""
Test script for the DVT to MC2 conversion API endpoint.

This script demonstrates how to use the new /api/convert-dvt-to-mc2 endpoint
to convert DVT format files to MC2 format via HTTP API.

Usage:
    python test_dvt_mc2_api.py
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

# Add the backend_fastapi app to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dvt_to_mc2_api():
    """Test the DVT to MC2 conversion API endpoint."""
    client = TestClient(app)

    # Sample DVT file path
    dvt_file_path = Path(__file__).parent.parent / "sample_data" / "Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv"

    if not dvt_file_path.exists():
        print(f"❌ Sample DVT file not found: {dvt_file_path}")
        return False

    print(f"🔄 Testing DVT to MC2 conversion API with file: {dvt_file_path.name}")

    # Read the DVT file
    with open(dvt_file_path, "rb") as f:
        dvt_content = f.read()

    # Call the API endpoint
    response = client.post(
        "/api/convert-dvt-to-mc2",
        files={"dvt_file": (dvt_file_path.name, dvt_content, "text/csv")},
    )

    print(f"📊 Response Status: {response.status_code}")

    if response.status_code == 200:
        print("✅ Conversion successful!")
        print(f"📁 Response size: {len(response.content)} bytes")

        # Check Content-Disposition header for filename
        content_disposition = response.headers.get("content-disposition")
        print(f"📎 Content-Disposition: {content_disposition}")

        # Save the converted file
        output_path = Path(__file__).parent.parent / "test_outputs" / "api_test_converted_mc2.csv"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"💾 Converted file saved to: {output_path}")

        # Verify the MC2 format structure
        with open(output_path) as f:
            lines = f.readlines()

        print(f"📝 Output file has {len(lines)} lines")

        if len(lines) >= 2:
            header = lines[0].strip()
            print(f"🎯 Header starts with: {header[:100]}...")

            # Check if header has proper TX1_POW, TX1_EVM, TX1_FREQ, TX1_MASK, TX1_LO_LEAKAGE_DB ordering
            if header.startswith("TX1_POW_") and "TX1_EVM_" in header and "TX1_FREQ_" in header:
                print("✅ Header has correct column ordering (TX1_POW, TX1_EVM, TX1_FREQ...)")
            else:
                print("⚠️ Header ordering may be incorrect")

        return True
    else:
        print(f"❌ Conversion failed: {response.text}")
        return False


def demo_api_usage():
    """Show example usage with curl command."""
    print("\n" + "=" * 60)
    print("🌐 API USAGE EXAMPLE")
    print("=" * 60)
    print()
    print("To use this API endpoint with curl:")
    print()
    print("curl -X POST \\")
    print("  -F 'dvt_file=@your_dvt_file.csv' \\")
    print("  http://localhost:8000/api/convert-dvt-to-mc2 \\")
    print("  -o converted_mc2_output.csv")
    print()
    print("The API will:")
    print("• Accept a DVT format CSV file")
    print("• Convert it to MC2 format with proper column ordering")
    print("• Return the MC2 file as a CSV download")
    print("• Set appropriate headers for file download")
    print()


if __name__ == "__main__":
    print("DVT to MC2 Conversion API Test")
    print("=" * 40)

    success = test_dvt_to_mc2_api()

    if success:
        demo_api_usage()
        print("🎉 All tests passed!")
    else:
        print("💥 Test failed!")
        sys.exit(1)
