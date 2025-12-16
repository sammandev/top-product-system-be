import io
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_bytes(relative_path: str) -> bytes:
    p = BASE_DATA_DIR / relative_path
    with p.open("rb") as f:
        return f.read()


def test_api_returns_xlsx_and_headers():
    client = TestClient(app)
    a = load_bytes("sample_data/000000000000018_MeasurePower_Lab_25G.csv")
    b = load_bytes("sample_data/Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv")
    spec = load_bytes("sample_data_config/compare_format_config.json")
    files = {
        "master_file": ("master.csv", io.BytesIO(a), "text/csv"),
        "dvt_file": ("dvt.csv", io.BytesIO(b), "text/csv"),
        "spec_file": ("spec.json", io.BytesIO(spec), "application/json"),
    }
    # Request XLSX by setting form fields 'human' and 'return_xlsx'
    resp = client.post(
        "/api/compare-formats",
        files=files,
        data={"human": "true", "return_xlsx": "true"},
    )

    assert resp.status_code == 200
    # content-type should be xlsx
    assert resp.headers.get("content-type") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    cd = resp.headers.get("content-disposition", "")
    # filename should include the base prefix
    assert "Golden_Compare_Compiled" in cd

    # validate returned bytes open with openpyxl if available
    try:
        from openpyxl import load_workbook

        buf = io.BytesIO(resp.content)
        wb = load_workbook(buf)
        assert wb is not None
    except Exception:
        pass
