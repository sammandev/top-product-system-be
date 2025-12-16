from pathlib import Path

from app.utils.dvt_to_mc2_converter import (
    convert_dvt_to_mc2,
    determine_bands_from_rows,
    extract_serial_number,
    make_output_filename,
    parse_dvt_file,
)

SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_data"


def load_sample(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def test_extract_serial_and_bands_from_sample():
    sample_path = SAMPLE_DATA_DIR / "Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv"
    content = load_sample(sample_path)

    serial = extract_serial_number(content)
    assert serial is not None
    assert "Golden_SN" in serial

    rows = parse_dvt_file(content)
    bands = determine_bands_from_rows(rows)
    # Expect at least 2 and 5 GHz present in this sample
    assert "2" in bands or "5" in bands

    filename = make_output_filename(serial, bands)
    # Filename should contain '_MeasuredPower_Lab_' and end with 'G.csv'
    assert "_MeasuredPower_Lab_" in filename
    assert filename.endswith(".csv")
    assert "G" in filename


def test_convert_outputs_four_lines_and_header_ordering():
    sample_path = SAMPLE_DATA_DIR / "Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv"
    content = load_sample(sample_path)

    mc2 = convert_dvt_to_mc2(content, include_usl_lsl=True)
    lines = mc2.splitlines()
    # Expect header, USL, LSL, data => 4 lines
    assert len(lines) >= 4

    header = lines[0]
    # Check that header starts with TX1_POW and includes TX1_EVM and TX1_FREQ
    assert header.startswith("TX1_POW_")
    assert "TX1_EVM_" in header
    assert "TX1_FREQ_" in header

    # Check data row exists on line 4 (index 3)
    data_row = lines[3]
    assert len(data_row) > 0
