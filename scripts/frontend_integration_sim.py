"""Simulate frontend FileCompare.vue interactions using TestClient.

Uploads two sample CSVs, calls /api/compare, and then /api/compare-download, printing status and a snippet of the CSV.
"""

import json
from io import BytesIO

from backend_fastapi import app
from fastapi.testclient import TestClient

client = TestClient(app)

CSV_A = "id,a\n1,10\n2,20\n3,30\n"
CSV_B = "id,a\n1,10\n2,99\n3,30\n"


def upload(content, name):
    files = {"file": (name, BytesIO(content.encode("utf-8")), "text/csv")}
    r = client.post("/api/upload-preview", files=files)
    print("upload", name, r.status_code)
    return r.json()["file_id"]


def main_sim():
    a_id = upload(CSV_A, "a.csv")
    b_id = upload(CSV_B, "b.csv")

    # Simulate compare with selecting all columns (mode both)
    payload = {
        "file_a": a_id,
        "file_b": b_id,
        "mode": "both",
        "a_selected_columns": json.dumps(None),
        "b_selected_columns": json.dumps(None),
    }
    r = client.post("/api/compare", data=payload)
    print("/api/compare", r.status_code)
    print("compare json columns:", r.json().get("columns")[:5])

    # Download CSV
    r2 = client.post("/api/compare-download", data=payload)
    print(
        "/api/compare-download",
        r2.status_code,
        "headers:",
        r2.headers.get("Content-Disposition"),
    )
    text = r2.content.decode("utf-8")
    print("csv preview:\n", "\n".join(text.splitlines()[:6]))


if __name__ == "__main__":
    main_sim()
