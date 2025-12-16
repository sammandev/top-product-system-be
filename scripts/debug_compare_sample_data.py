import io
import sys
from pathlib import Path

from backend_fastapi import app
from fastapi.testclient import TestClient

sys.path.insert(0, r"d:/Projects/AST_Parser")

client = TestClient(app)

root = Path(__file__).resolve().parent.parent
sample_a = root / "sample_data" / "000000000000018_MeasurePower_Lab_25G.csv"
sample_b = root / "sample_data" / "Golden_SN..9403_25G_TX_2025_09_10_15-16-19.csv"

print("Using files:", sample_a, sample_b)

with sample_a.open("rb") as af, sample_b.open("rb") as bf:
    a_bytes = af.read()
    b_bytes = bf.read()

print("local sizes:", len(a_bytes), len(b_bytes))

files = [
    ("master_file", ("master.csv", io.BytesIO(a_bytes), "text/csv")),
    ("dvt_file", ("dvt.csv", io.BytesIO(b_bytes), "text/csv")),
]
print("prepared files:")
for k, v in files:
    try:
        fname = v[0]
        fobj = v[1]
        size = len(fobj.getvalue()) if hasattr(fobj, "getvalue") else "<no getvalue>"
    except Exception:
        fname = "<err>"
        size = "<err>"
    print(" ", k, fname, size)

r = client.post("/api/compare-formats", files=files)
print("status", r.status_code)
try:
    print("json:", r.json())
except Exception:
    print("text:", r.text)
