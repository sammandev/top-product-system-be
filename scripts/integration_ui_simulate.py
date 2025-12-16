"""
Simulate UI flows via HTTP to validate frontend/back-end integration.
Flows:
 1) mode=rows with no rows selected -> should return full dataset
 2) mode=both selecting a couple of rows and columns -> should return subset
"""

import json
import os

import requests

BASE = os.environ.get("ASTPARSER_BASE", "http://127.0.0.1:8001")

sample = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "frontend-sakai",
        "public",
        "sample_data",
        "6gg_2024_10_22_09-01-38.csv",
    )
)


def upload(p):
    with open(p, "rb") as f:
        r = requests.post(f"{BASE}/api/upload-preview", files={"file": f})
        r.raise_for_status()
        return r.json()


def simulate():
    info = upload(sample)
    fid = info["file_id"]
    print("uploaded", fid)

    # Flow 1: mode=rows with no selected rows -> should return all rows
    data1 = {"file_id": fid, "mode": "rows"}
    r1 = requests.post(f"{BASE}/api/parse", data=data1)
    print("flow1 status", r1.status_code)
    print("flow1 rows", len(r1.json().get("rows", [])))

    # Flow 2: mode=both pick first two rows and first three columns
    payload = {
        "file_id": fid,
        "mode": "both",
        "selected_columns": json.dumps([0, 1, 2]),
        "selected_rows": json.dumps([0, 1]),
    }
    r2 = requests.post(f"{BASE}/api/parse", data=payload)
    print("flow2 status", r2.status_code)
    print("flow2 columns", r2.json().get("columns"))
    print("flow2 rows", len(r2.json().get("rows", [])))


if __name__ == "__main__":
    simulate()
