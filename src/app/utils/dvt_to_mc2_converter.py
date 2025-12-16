"""
DVT to MC2 format converter.

Transforms data from DVT format to MC2 format.

DVT format has rows with columns: Standard, BandWidth, Freq, DataRate, Antenna, T. Power, M. Power, etc.
MC2 format has a header row with columns like TX{1-4}_POW_2462_11B_CCK11_B20, TX{1-4}_EVM_2462_11B_CCK11_B20, etc.
and a single data row with values.

Usage:
    from app.utils.dvt_to_mc2_converter import convert_dvt_to_mc2, convert_dvt_file_to_mc2

    # Convert content string
    mc2_content = convert_dvt_to_mc2(dvt_content)

    # Convert file
    convert_dvt_file_to_mc2("input.csv", "output.csv")
"""

import csv
import re
from io import StringIO
from pathlib import Path
from typing import Any


def parse_dvt_file(content: str) -> list[dict[str, Any]]:
    """Parse DVT format CSV content and extract test data rows."""
    lines = content.strip().split("\n")

    # Find the header row (starts with "Standard,BandWidth,Freq...")
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        # Robustly detect header row if it contains Standard and Freq and at least one of Antenna/BandWidth/DataRate
        if "standard" in low and "freq" in low and ("antenna" in low or "bandwidth" in low or "datarate" in low):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find DVT data header row")

    # Parse CSV from header onwards
    csv_content = "\n".join(lines[header_idx:])
    reader = csv.DictReader(StringIO(csv_content))

    data_rows = []
    for row in reader:
        # Skip empty rows or rows with missing key data
        if not row.get("Standard") or not row.get("Freq") or not row.get("DataRate") or row.get("Antenna") is None:
            continue
        data_rows.append(row)

    return data_rows


def standardize_names(standard: str, datarate: str) -> tuple[str, str]:
    """Standardize standard and datarate names to match MC2 format."""
    # Map DVT standard names to MC2 format
    standard_map = {
        "b": "11B",
        "a": "11AG",
        "g": "11AG",
        "n": "11N",
        "ac": "11AC",
        "ax": "11AX",
        "be": "11BE",
    }

    # Map datarate formats
    datarate_clean = datarate.strip()
    if "CCK" in datarate_clean:
        # Extract CCK rate (e.g., " CCK_11" -> "CCK11")
        match = re.search(r"CCK[_\s]*(\d+)", datarate_clean)
        if match:
            datarate_clean = f"CCK{match.group(1)}"
    elif "MCS" in datarate_clean:
        # Extract MCS number (e.g., " MCS0" -> "MCS0")
        match = re.search(r"MCS(\d+)", datarate_clean)
        if match:
            datarate_clean = f"MCS{match.group(1)}"

    return standard_map.get(standard.lower(), standard.upper()), datarate_clean


def extract_serial_number(content: str) -> str | None:
    """Extract a serial number from the DVT content.

    Looks for lines like "Serial Number: <value>" or CSV-like "Serial Number,<value>" in the first 30 lines.
    Returns the serial string or None if not found.
    """
    lines = content.split("\n")
    # Try strict CSV-style: Row 3 (index 2), Column 2 (index 1) if present
    try:
        if len(lines) >= 3:
            third = lines[2]
            parts = [p.strip() for p in third.split(",")]
            if len(parts) >= 2 and re.search(r"serial\s*number", parts[0], re.I):
                val = parts[1]
                if val:
                    return val.strip()
    except Exception:
        pass

    # Fallback: scan first 30 lines for any Serial Number pattern
    for line in lines[:30]:
        m = re.search(r"Serial\s*Number[:\s,]*([^,\r\n]+)", line, re.I)
        if m:
            return m.group(1).strip()

    # Last resort: search whole content for common tokens like 'Serial' followed by a value
    m2 = re.search(r"Serial[:\s,]*([A-Za-z0-9_.\-]+)", content, re.I)
    if m2:
        return m2.group(1).strip()

    return None


def _map_freq_to_band(freq_val: float) -> str | None:
    """Map a frequency in MHz to a band token like '2G','5G','6G','25G','26G','56G'.

    NOTE: These ranges are heuristic and chosen to cover common test frequencies. If a frequency
    falls outside these ranges it will be ignored.
    """
    # Heuristic band boundaries (MHz)
    if freq_val is None:
        return None
    try:
        f = float(freq_val)
    except Exception:
        return None

    if 2000 <= f < 3000:
        return "2"
    # 5 GHz band roughly 5000-5924
    if 5000 <= f < 5925:
        return "5"
    # 6 GHz band roughly 5925-7125
    if 5925 <= f < 10000:
        return "6"
    # mmWave/other high bands (20-27 GHz)
    if 20000 <= f < 25500:
        return "25"
    if 25500 <= f < 27000:
        return "26"
    if f >= 50000:
        return "56"
    return None


def determine_bands_from_rows(dvt_rows: list[dict[str, Any]]) -> list[str]:
    """Determine a sorted list of band tokens present in the DVT rows (e.g. ['2G','5G'])."""
    bands = set()
    for r in dvt_rows:
        try:
            freq = r.get("Freq") or r.get("frequency")
            if freq is None:
                continue
            b = _map_freq_to_band(freq)
            if b:
                bands.add(b)
        except Exception:
            continue

    # Return sorted numeric tokens as strings
    return sorted(bands, key=lambda x: float(x))


def make_output_filename(serial: str | None, bands: list[str]) -> str:
    """Create output filename according to the requested pattern.

    Pattern: {serial_number}_MeasuredPower_Lab_{bands}.csv
    - serial: the serial string or 'UNKNOWN'
    - bands: list like ['2G','5G'] -> joined by '-' (or 'UNKNOWN' if empty)
    """
    s = (serial or "UNKNOWN").replace(" ", "_")
    if bands:
        # Concatenate numeric tokens and append 'G', e.g. ['2','5'] -> '25G', ['25'] -> '25G'
        band_part = "".join(bands) + "G"
    else:
        band_part = "UNKNOWN"

    return f"{s}_MeasuredPower_Lab_{band_part}.csv"


def dvt_to_mc2_header_name(antenna: int, metric: str, freq: str, standard: str, datarate: str, bandwidth: str) -> str:
    """Generate MC2 format column header name from DVT data."""
    std_name, dr_name = standardize_names(standard, datarate)

    # Antenna is 0-indexed in DVT, 1-indexed in MC2
    tx_num = int(antenna) + 1

    # Clean bandwidth format (e.g., "20" -> "B20", "40" -> "B40")
    bw_clean = bandwidth.strip()
    if not bw_clean.startswith("B"):
        bw_clean = f"B{bw_clean}"

    # Map metric names
    metric_map = {
        "POW": "POW",
        "EVM": "EVM",
        "FREQ": "FREQ",
        "MASK": "MASK",
        "LO_LEAKAGE_DB": "LO_LEAKAGE_DB",
    }

    metric_clean = metric_map.get(metric, metric.upper())

    return f"TX{tx_num}_{metric_clean}_{freq}_{std_name}_{dr_name}_{bw_clean}"


def generate_mc2_column_order(dvt_rows: list[dict[str, Any]]) -> list[str]:
    """Generate properly ordered MC2 column headers based on DVT data."""
    # Extract unique combinations of freq/standard/datarate/bandwidth
    test_configs = set()
    for row in dvt_rows:
        try:
            freq = row["Freq"].strip()
            standard = row["Standard"].strip()
            datarate = row["DataRate"].strip()
            bandwidth = row["BandWidth"].strip()
            test_configs.add((freq, standard, datarate, bandwidth))
        except (KeyError, AttributeError):
            continue

    # Sort test configs for consistent ordering
    sorted_configs = sorted(test_configs)

    # Define metric order
    metrics = ["POW", "EVM", "FREQ", "MASK", "LO_LEAKAGE_DB"]
    antennas = ["TX1", "TX2", "TX3", "TX4"]

    # Generate ordered headers
    ordered_headers = []
    for freq, standard, datarate, bandwidth in sorted_configs:
        std_name, dr_name = standardize_names(standard, datarate)
        bw_clean = bandwidth.strip()
        if not bw_clean.startswith("B"):
            bw_clean = f"B{bw_clean}"

        # For each config, add all antenna/metric combinations in order
        for antenna in antennas:
            for metric in metrics:
                header = f"{antenna}_{metric}_{freq}_{std_name}_{dr_name}_{bw_clean}"
                ordered_headers.append(header)

    # Add PEGARetryCount at the end
    ordered_headers.append("PEGARetryCount")

    return ordered_headers


def convert_dvt_to_mc2(dvt_content: str, include_usl_lsl: bool = False) -> str:
    """Convert DVT format content to MC2 format.

    Args:
        dvt_content: DVT format CSV content
        include_usl_lsl: If True, include USL/LSL rows (rows 2-3). If False, skip them.

    Returns:
        MC2 format CSV content with proper row structure:
        - Row 1: Headers
        - Row 2: USL values (if include_usl_lsl=True)
        - Row 3: LSL values (if include_usl_lsl=True)
        - Row 4 (or 2 if no USL/LSL): Measurement data
    """
    dvt_rows = parse_dvt_file(dvt_content)

    # Generate properly ordered column headers
    ordered_headers = generate_mc2_column_order(dvt_rows)

    # Create MC2 column mapping
    mc2_columns = {}  # header_name -> value

    for row in dvt_rows:
        try:
            antenna = int(row["Antenna"])
            freq = row["Freq"].strip()
            standard = row["Standard"].strip()
            datarate = row["DataRate"].strip()
            bandwidth = row["BandWidth"].strip()

            # Map key measurement values
            measurements = [
                ("POW", row.get("M. Power", "")),
                ("EVM", row.get("EVM", "")),
                ("FREQ", row.get("FreqError", "")),
                ("MASK", "0.00"),  # DVT doesn't have direct equivalent, use default
                ("LO_LEAKAGE_DB", row.get("LO Leakage", "")),
            ]

            for metric, value in measurements:
                if value and value.strip():
                    header_name = dvt_to_mc2_header_name(antenna, metric, freq, standard, datarate, bandwidth)
                    # Clean numeric values
                    try:
                        if metric == "LO_LEAKAGE_DB":
                            # LO Leakage values are negative in DVT format
                            mc2_columns[header_name] = str(float(value))
                        else:
                            mc2_columns[header_name] = str(float(value))
                    except ValueError:
                        mc2_columns[header_name] = value.strip()
                else:
                    # Set default values for missing data
                    header_name = dvt_to_mc2_header_name(antenna, metric, freq, standard, datarate, bandwidth)
                    if metric == "MASK":
                        mc2_columns[header_name] = "0.00"
                    else:
                        mc2_columns[header_name] = ""

        except (ValueError, KeyError):
            # Skip rows with invalid data
            continue

    # Generate MC2 CSV content with proper structure
    if not mc2_columns:
        raise ValueError("No valid measurement data found in DVT file")

    # Ensure PEGARetryCount is included
    mc2_columns["PEGARetryCount"] = "0.00"

    # Build CSV content with proper row structure
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)

    # Row 1: Write header
    writer.writerow(ordered_headers)

    # Rows 2-3: Always write USL and LSL rows (empty if we don't have spec data)
    usl_row = ["" for _ in ordered_headers]
    writer.writerow(usl_row)

    lsl_row = ["" for _ in ordered_headers]
    writer.writerow(lsl_row)

    # Row 4: Write data row (measurement values)
    data_row = [mc2_columns.get(header, "") for header in ordered_headers]
    writer.writerow(data_row)

    return csv_buffer.getvalue()


def convert_dvt_file_to_mc2(dvt_path: str, mc2_output_path: str, include_usl_lsl: bool = False) -> None:
    """Convert a DVT format file to MC2 format."""
    # Support reading CSV or XLSX input files
    p = Path(dvt_path)
    if not p.exists():
        raise FileNotFoundError(dvt_path)

    if p.suffix.lower() in (".xls", ".xlsx"):
        # Use pandas to read first sheet and convert to CSV-like content with expected headers
        import pandas as pd

        df = pd.read_excel(p, sheet_name=0, dtype=str)
        # Convert dataframe to CSV string (no index)
        dvt_content = df.to_csv(index=False)
    else:
        dvt_content = p.read_text(encoding="utf-8", errors="ignore")

    mc2_content = convert_dvt_to_mc2(dvt_content, include_usl_lsl=include_usl_lsl)
    Path(mc2_output_path).write_text(mc2_content, encoding="utf-8")
    print(f"Converted {dvt_path} to {mc2_output_path}")


if __name__ == "__main__":
    # Example usage
    dvt_file = r"d:\Projects\AST_Parser\backend_fastapi\sample_data\Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv"
    mc2_file = r"d:\Projects\AST_Parser\backend_fastapi\test_outputs\converted_dvt_to_mc2.csv"

    try:
        convert_dvt_file_to_mc2(dvt_file, mc2_file)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Conversion failed: {e}")
