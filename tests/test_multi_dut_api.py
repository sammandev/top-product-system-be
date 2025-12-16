from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

BASE_DIR = Path(__file__).resolve().parents[1]


def test_analyze_multi_dut():
    client = TestClient(app)
    mc2_path = BASE_DIR / "data" / "sample_data" / "2025_09_18_Wireless_Test_2_5G_Sampling_HH5K.csv"
    spec_path = BASE_DIR / "data" / "sample_data_config" / "multi-dut_all_specs.json"

    with mc2_path.open("rb") as f_mc2, spec_path.open("rb") as f_spec:
        files = {
            "mc2_file": ("sample.csv", f_mc2, "text/csv"),
            "spec_file": ("spec.json", f_spec, "application/json"),
        }
        resp = client.post("/api/analyze-multi-dut", files=files)
        assert resp.status_code == 200
        # Response should be xlsx bytes
        assert resp.headers.get("content-type", "").startswith("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        assert resp.content[:4] == b"PK\x03\x04"
