import json

import requests

BASE = "http://127.0.0.1:8001"
# upload a and b
r1 = requests.post(f"{BASE}/api/upload-preview", files={"file": ("a.csv", b"A,B\n1,2\n3,4\n")})
print("upload a status", r1.status_code, r1.text)
fa = r1.json()["file_id"]
r2 = requests.post(f"{BASE}/api/upload-preview", files={"file": ("b.csv", b"A,B\n1,2\n3,9\n")})
print("upload b status", r2.status_code, r2.text)
fb = r2.json()["file_id"]
print("fa,fb", fa, fb)
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
print("compare status", r.status_code)
try:
    print("compare json:", r.json())
except Exception:
    print("compare text:", r.text)
