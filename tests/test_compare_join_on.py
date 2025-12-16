import json
from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CSV_A = "id,a\n1,10\n2,20\n3,30\n"
CSV_B = "id,b\n1,100\n2,200\n4,400\n"


def upload_content(content: str, name="file.csv"):
    files = {"file": (name, BytesIO(content.encode("utf-8")), "text/csv")}
    resp = client.post("/api/upload-preview", files=files)
    assert resp.status_code == 200
    data = resp.json()
    return data["file_id"]


def test_missing_join_column():
    # A uses join 'id', B uses 'missing'
    a_id = upload_content(CSV_A, name="a.csv")
    b_id = upload_content(CSV_B, name="b.csv")

    payload = {
        "file_a": a_id,
        "file_b": b_id,
        "mode": "both",
        "a_join_on": json.dumps("id"),
        "b_join_on": json.dumps("missing"),
    }

    resp = client.post("/api/compare", data=payload)
    # Expect a 400 or a helpful error
    assert resp.status_code in (400, 422)


def test_join_type_mismatch():
    # Create CSVs where join columns have mismatched types
    csv1 = "key,val\na,1\nb,2\n"
    csv2 = "key,val\n1,100\n2,200\n"
    a_id = upload_content(csv1, name="a.csv")
    b_id = upload_content(csv2, name="b.csv")

    payload = {
        "file_a": a_id,
        "file_b": b_id,
        "mode": "both",
        "a_join_on": json.dumps("key"),
        "b_join_on": json.dumps("key"),
    }

    resp = client.post("/api/compare", data=payload)
    # If the service attempts to align by converting types, it may succeed.
    # Accept either a 200 with compare result or a 400/422 if it rejects mismatch.
    assert resp.status_code in (200, 400, 422)


def test_duplicate_keys_in_join():
    # Duplicate keys in A
    csv_a = "id,v\n1,10\n1,11\n2,20\n"
    csv_b = "id,v\n1,100\n2,200\n"
    a_id = upload_content(csv_a, name="a.csv")
    b_id = upload_content(csv_b, name="b.csv")

    payload = {
        "file_a": a_id,
        "file_b": b_id,
        "mode": "both",
        "a_join_on": json.dumps("id"),
        "b_join_on": json.dumps("id"),
    }

    resp = client.post("/api/compare", data=payload)
    # Service should handle duplicates gracefully (e.g., multiple matched rows), so expect 200
    assert resp.status_code == 200
