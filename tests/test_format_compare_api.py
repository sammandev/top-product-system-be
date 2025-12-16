import io
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_bytes(relative_path: str) -> bytes:
    p = BASE_DATA_DIR / relative_path
    with p.open("rb") as f:
        return f.read()


def test_compare_formats_endpoint():
    a = load_bytes("sample_data/000000000000018_MeasurePower_Lab_25G.csv")
    b = load_bytes("sample_data/Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv")
    spec_bytes = load_bytes("sample_data_config/compare_format_config.json")
    files = {
        "master_file": ("master.csv", io.BytesIO(a), "text/csv"),
        "dvt_file": ("dvt.csv", io.BytesIO(b), "text/csv"),
        "spec_file": ("spec.json", io.BytesIO(spec_bytes), "application/json"),
    }
    resp = client.post("/api/compare-formats", files=files, data={"threshold": "1.0"})
    assert resp.status_code == 200
    j = resp.json()
    assert "rows" in j
    assert isinstance(j["rows"], list)
    if j["rows"]:
        r = j["rows"][0]
        assert "usl" in r and "lsl" in r
        assert "mc2_spec_diff" in r and "dvt_spec_diff" in r
        assert "result_label" in r
        assert "mc2_result" in r and "dvt_result" in r
