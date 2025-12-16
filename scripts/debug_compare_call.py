import sys

from fastapi.testclient import TestClient

sys.path.insert(0, r"d:/Projects/AST_Parser/backend_fastapi")
import json
from pathlib import Path

from app.main import app

client = TestClient(app)
tmp = Path("tmp_debug")
tmp.mkdir(exist_ok=True)
a = tmp / "a.csv"
b = tmp / "b.csv"
a.write_text("x,y\n1,2\n3,4\n")
b.write_text("x,y\n1,2\n3,9\n")

with open(str(a), "rb") as fa:
    r1 = client.post("/api/upload-preview", files={"file": ("a.csv", fa, "text/csv")})
with open(str(b), "rb") as fb:
    r2 = client.post("/api/upload-preview", files={"file": ("b.csv", fb, "text/csv")})

print("upload statuses", r1.status_code, r2.status_code)
id1 = r1.json()["file_id"]
id2 = r2.json()["file_id"]
form = {
    "file_a": id1,
    "file_b": id2,
    "mode": "both",
    "a_selected_columns": json.dumps([0, 1]),
    "b_selected_columns": json.dumps([0, 1]),
}
resp = client.post("/api/compare-download", data=form)
print("status", resp.status_code)
try:
    print("json", resp.json())
except Exception:
    print("text", resp.text)
