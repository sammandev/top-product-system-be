import json
import os

import requests

BASE = os.environ.get("ASTPARSER_TEST_BASE", "http://127.0.0.1:8001")


def upload_bytes(name, b):
    r = requests.post(f"{BASE}/api/upload-preview", files={"file": (name, b)})
    r.raise_for_status()
    return r.json()["file_id"]


def test_compare_simple_match():
    a = b"A,B\n1,2\n3,4\n"
    b = b"A,B\n1,2\n3,9\n"
    fa = upload_bytes("a.csv", a)
    fb = upload_bytes("b.csv", b)
    payload = {
        "file_a": fa,
        "file_b": fb,
        "mode": "both",
        "a_selected_columns": json.dumps([0, 1]),
        "b_selected_columns": json.dumps([0, 1]),
        "a_selected_rows": json.dumps([0, 1]),
        "b_selected_rows": json.dumps([0, 1]),
    }
    r = requests.post(f"{BASE}/api/compare", data=payload)
    r.raise_for_status()
    j = r.json()
    # first row should match both cols
    assert j["rows"][0]["MATCH::0"] is True
    assert j["rows"][0]["MATCH::1"] is True
    # second row: second column mismatches
    assert j["rows"][1]["MATCH::0"] is True
    assert j["rows"][1]["MATCH::1"] is False
