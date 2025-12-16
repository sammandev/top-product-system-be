"""
DVT to MC2 Format Converter Script

Convert WiFi DVT test result files to MasterControlV2 (MC2) format.

Usage:
    python convert_dvt_to_mc2.py --input <dvt_file.csv> --output <mc2_file.csv>
    python convert_dvt_to_mc2.py --input sample_data/Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv --output test_outputs/converted_mc2.csv
"""

import argparse
import sys
from pathlib import Path

from app.utils.dvt_to_mc2_converter import convert_dvt_file_to_mc2

# Add backend_fastapi to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Convert DVT format to MC2 format")
    parser.add_argument("--input", "-i", required=True, help="Input DVT CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output MC2 CSV file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        return 1

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.verbose:
            print(f"Converting {input_path} to {output_path}...")

        convert_dvt_file_to_mc2(str(input_path), str(output_path))

        if args.verbose:
            print("Conversion completed successfully!")
            print(f"Output written to: {output_path}")

            # Show file size info
            size_bytes = output_path.stat().st_size
            print(f"Output file size: {size_bytes} bytes")

        return 0

    except Exception as e:
        print(f"Conversion failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
