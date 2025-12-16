from pathlib import Path

import requests

BASE = "http://127.0.0.1:8001"
master = Path(__file__).resolve().parent.parent / "testdata" / "sample_master.txt"
dvt = Path(__file__).resolve().parent.parent / "testdata" / "sample_dvt.csv"

with master.open("rb") as m, dvt.open("rb") as d:
    files = {
        "master_file": ("sample_master.txt", m, "text/plain"),
        "dvt_file": ("sample_dvt.csv", d, "text/csv"),
    }
    r = requests.post(f"{BASE}/api/compare-formats", files=files)
    print("status", r.status_code)
    try:
        print("json:", r.json())
    except Exception:
        print("text:", r.text)
