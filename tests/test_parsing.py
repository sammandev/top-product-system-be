import json
import os
from pathlib import Path

import requests

BASE = os.environ.get("ASTPARSER_TEST_BASE", "http://127.0.0.1:8001")
TESTDATA_DIR = Path(__file__).resolve().parents[1] / "data" / "testdata"


def post_upload(path: Path):
    with path.open("rb") as f:
        files = {"file": (path.name, f)}
        r = requests.post(f"{BASE}/api/upload-preview", files=files)
    r.raise_for_status()
    return r.json()


def test_preview_detects_header_and_columns():
    path = TESTDATA_DIR / "sample.csv"
    resp = post_upload(path)
    assert "columns" in resp and len(resp["columns"]) > 2


def test_parse_without_header():
    # Create a small CSV without header
    data = b"1,2,3\n4,5,6\n7,8,9\n"
    files = {"file": ("noheader.csv", data)}
    r = requests.post(f"{BASE}/api/upload-preview", files=files, data={"has_header": "false"})
    r.raise_for_status()
    resp = r.json()
    # columns should be auto-generated (0,1,2) or similar
    assert len(resp["columns"]) == 3


def test_duplicate_column_handling():
    # CSV with duplicate column names
    data = b"A,B,A\n1,2,3\n4,5,6\n"
    files = {"file": ("dupcols.csv", data)}
    r = requests.post(f"{BASE}/api/upload-preview", files=files)
    r.raise_for_status()
    resp = r.json()
    # ensure columns list length matches header count
    assert len(resp["columns"]) == 3


def test_encoding_latin1():
    # CSV with latin-1 encoded characters (e.g., accented)
    s = "Name,Value\nJos\xe9,100\nAna,200\n"
    b = s.encode("latin-1")
    files = {"file": ("latin1.csv", b)}
    r = requests.post(f"{BASE}/api/upload-preview", files=files)
    r.raise_for_status()
    resp = r.json()
    assert "José" in ",".join(resp.get("columns", [])) or any("José" in str(v) for row in resp.get("preview", []) for v in row.values())


def test_xlsx_upload(tmp_path):
    # create a small xlsx file using pandas
    import pandas as pd

    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    p = tmp_path / "small.xlsx"
    df.to_excel(p, index=False)
    with p.open("rb") as fh:
        files = {"file": fh}
        r = requests.post(f"{BASE}/api/upload-preview", files=files)
    r.raise_for_status()
    resp = r.json()
    assert "A" in resp["columns"] and "B" in resp["columns"]


def test_cleanup_endpoint():
    # Call cleanup without admin key
    r = requests.post(f"{BASE}/api/cleanup-uploads")
    r.raise_for_status()
    data = r.json()
    assert "removed" in data


def test_large_file_streaming(tmp_path):
    # create a large CSV (~10000 rows) to ensure server handles it
    p = tmp_path / "large.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("c1,c2,c3\n")
        for i in range(10000):
            f.write(f"{i},val{i},x{i}\n")
    with p.open("rb") as fh:
        files = {"file": fh}
        r = requests.post(f"{BASE}/api/upload-preview", files=files)
    r.raise_for_status()
    resp = r.json()
    assert len(resp["columns"]) == 3


def test_duplicate_column_selection():
    # CSV where columns repeat; ensure selection by index works
    data = b"A,B,A\n1,2,3\n4,5,6\n"
    files = {"file": ("dupcols2.csv", data)}
    r = requests.post(f"{BASE}/api/upload-preview", files=files)
    r.raise_for_status()
    resp = r.json()
    fid = resp["file_id"]
    # select column index 0 and 2 (both 'A') using form post
    payload = {
        "file_id": fid,
        "mode": "columns",
        "selected_columns": json.dumps([0, 2]),
        "exclude_columns": json.dumps([]),
    }
    r2 = requests.post(f"{BASE}/api/parse", data=payload)
    r2.raise_for_status()
    out = r2.json()
    assert len(out["columns"]) == 2
