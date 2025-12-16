import sys
from pathlib import Path

from fastapi.testclient import TestClient

# ensure repository root is on sys.path
sys.path.insert(0, r"d:/Projects/AST_Parser")

from app.main import app

client = TestClient(app)
master = Path(__file__).resolve().parent.parent / "testdata" / "sample_master.txt"
dvt = Path(__file__).resolve().parent.parent / "testdata" / "sample_dvt.csv"

with master.open("rb") as m, dvt.open("rb") as d:
    files = {
        "master_file": ("sample_master.txt", m, "text/plain"),
        "dvt_file": ("sample_dvt.csv", d, "text/csv"),
    }
    r = client.post("/api/compare-formats", files=files)
    print("status", r.status_code)
    try:
        j = r.json()
        print("json:", j)
    except Exception:
        print("text:", r.text)
