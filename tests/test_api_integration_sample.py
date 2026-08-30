from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "testdata"
SPEC_PATH = DATA_DIR.parents[1] / "data" / "sample_data_config" / "compare_format_config.json"
REQUIRED_FIXTURES = [DATA_DIR / "sample_master.txt", DATA_DIR / "sample_dvt.csv", SPEC_PATH]


@pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_FIXTURES),
    reason="Sample compare fixtures are not available",
)
def test_compare_endpoint_with_sample_files():
    client = TestClient(app)
    master_path = DATA_DIR / "sample_master.txt"
    dvt_path = DATA_DIR / "sample_dvt.csv"

    with master_path.open("rb") as m, dvt_path.open("rb") as d, SPEC_PATH.open("rb") as s:
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
