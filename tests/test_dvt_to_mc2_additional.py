import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.utils.dvt_to_mc2_converter import (
    extract_serial_number,
)


def test_extract_serial_various_formats():
    # CSV style row 3 column 2
    csv_content = "\n".join(
        [
            "Model Name:,X",
            "Chipset:,Y",
            "Serial Number:,SERIAL_ABC123",
            "Standard,BandWidth,Freq,DataRate,Antenna",
            "b,20,2412, CCK_11,0",
        ]
    )
    s = extract_serial_number(csv_content)
    assert s == "SERIAL_ABC123"

    # Inline style
    inline = "Some header\nSerial Number: INLINE_987\nOther"
    s2 = extract_serial_number(inline)
    assert s2 == "INLINE_987"


def test_xlsx_integration_post(tmp_path):
    # Create a small XLSX file mimicking DVT layout
    # Build rows that match DVT layout: include header row and one data row
    rows = [
        ["Model Name:", "HH5K"],
        ["Chipset:", "BCM"],
        ["Serial Number:", "Golden_SN..XLSX_1"],
        ["", ""],
        ["Standard", "BandWidth", "Freq", "DataRate", "Antenna"],
        ["b", "20", "2412", "CCK_11", "0"],
    ]
    df = pd.DataFrame(rows)
    xlsx_path = tmp_path / "sample_dvt.xlsx"
    df.to_excel(xlsx_path, index=False, header=False)

    client = TestClient(app)
    with open(xlsx_path, "rb") as f:
        resp = client.post(
            "/api/convert-dvt-to-mc2",
            files={
                "dvt_file": (
                    "sample_dvt.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert resp.status_code == 200
    # Check the Content-Disposition filename includes the serial and MeasuredPower_Lab
    cd = resp.headers.get("content-disposition", "")
    assert "Golden_SN..XLSX_1" in cd
    assert "MeasuredPower_Lab" in cd
    # Response should be CSV text
    text = resp.content.decode("utf-8")
    lines = text.splitlines()
    # Should have header + USL + LSL + data = at least 4 lines
    assert len(lines) >= 4
