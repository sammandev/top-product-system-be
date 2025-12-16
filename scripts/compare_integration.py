import json
import os
import sys

import requests

BASE = os.environ.get("ASTPARSER_BASE", "http://127.0.0.1:8001")


def upload(path):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/upload-preview", files={"file": (os.path.basename(path), f)})
        r.raise_for_status()
        return r.json()["file_id"]


def compare_and_save(a_path, b_path, out_csv="compare_out.csv"):
    fa = upload(a_path)
    fb = upload(b_path)
    payload = {
        "file_a": fa,
        "file_b": fb,
        "mode": "both",
        "a_selected_columns": json.dumps([0, 1]),
        "b_selected_columns": json.dumps([0, 1]),
        "a_selected_rows": json.dumps([0, 1]),
        "b_selected_rows": json.dumps([0, 1]),
    }
    r = requests.post(f"{BASE}/api/compare-download", data=payload)
    r.raise_for_status()
    with open(out_csv, "wb") as f:
        f.write(r.content)
    print("Saved", out_csv)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: compare_integration.py fileA fileB")
        sys.exit(1)
    compare_and_save(sys.argv[1], sys.argv[2])
