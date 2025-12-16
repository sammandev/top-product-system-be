from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "testdata"


def test_compare_endpoint_with_sample_files():
    client = TestClient(app)
    master_path = DATA_DIR / "sample_master.txt"
    dvt_path = DATA_DIR / "sample_dvt.csv"

    spec_path = DATA_DIR.parents[1] / "data" / "sample_data_config" / "compare_format_config.json"

    with master_path.open("rb") as m, dvt_path.open("rb") as d, spec_path.open("rb") as s:
        files = {
            "master_file": ("sample_master.txt", m, "text/plain"),
            "dvt_file": ("sample_dvt.csv", d, "text/csv"),
            "spec_file": ("spec.json", s, "application/json"),
        }
        resp = client.post("/api/compare-formats", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "rows" in data and "summary" in data
        # basic sanity checks
        assert isinstance(data["rows"], list)
        assert isinstance(data["summary"], dict)
