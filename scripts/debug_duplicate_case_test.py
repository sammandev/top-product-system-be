import json
from io import BytesIO

from backend_fastapi import app
from fastapi.testclient import TestClient

client = TestClient(app)

csv_a = "id,v\n1,10\n1,11\n2,20\n"
csv_b = "id,v\n1,100\n2,200\n"

files = {"file": ("a.csv", BytesIO(csv_a.encode("utf-8")), "text/csv")}
resp1 = client.post("/api/upload-preview", files=files)
print("upload a", resp1.status_code, resp1.json())
files = {"file": ("b.csv", BytesIO(csv_b.encode("utf-8")), "text/csv")}
resp2 = client.post("/api/upload-preview", files=files)
print("upload b", resp2.status_code, resp2.json())

payload = {
    "a_file_id": resp1.json()["file_id"],
    "b_file_id": resp2.json()["file_id"],
    # note: intentionally do NOT set mode to mirror the test
    "a_join_on": json.dumps("id"),
    "b_join_on": json.dumps("id"),
}

resp = client.post("/api/compare", data=payload)
print("compare status", resp.status_code)
try:
    print("compare json:", resp.json())
except Exception:
    print("compare text:", resp.text)
