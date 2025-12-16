"""Test API key and admin key enforcement for endpoints.

- Call /api/compare without API key header (should be allowed if no env API key set).
- Set ASTPARSER_API_KEY in environment and verify /api/compare rejects missing header and accepts correct header.
- Call /api/cleanup-uploads without admin key (should reject if env set), then with correct admin key.
"""

import os
from io import BytesIO

from fastapi.testclient import TestClient

# We'll import main dynamically to control env


def run_tests(with_env=False):
    if with_env:
        os.environ["ASTPARSER_API_KEY"] = "secret123"
        os.environ["ASTPARSER_ADMIN_KEY"] = "adminkey"
    else:
        os.environ.pop("ASTPARSER_API_KEY", None)
        os.environ.pop("ASTPARSER_ADMIN_KEY", None)

    # reload the package-level app module (avoid importing app.main)
    import importlib

    import app.main as main_mod

    importlib.reload(main_mod)
    client = TestClient(main_mod.app)

    # upload small files
    csv = "id,v\n1,1\n2,2\n"
    files = {"file": ("a.csv", BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/api/upload-preview", files=files)
    fid = r.json()["file_id"]

    payload = {"file_a": fid, "file_b": fid, "mode": "both"}

    # call compare without header
    r2 = client.post("/api/compare", data=payload)
    print("compare without header status", r2.status_code)

    # call compare with header
    headers = {"X-API-KEY": os.environ.get("ASTPARSER_API_KEY", "")}
    r3 = client.post("/api/compare", data=payload, headers=headers)
    print("compare with header status", r3.status_code)

    # cleanup-uploads
    r4 = client.post("/api/cleanup-uploads", data={"admin_key": ""})
    print("cleanup without key", r4.status_code)
    r5 = client.post(
        "/api/cleanup-uploads",
        data={"admin_key": os.environ.get("ASTPARSER_ADMIN_KEY", "")},
    )
    print("cleanup with key", r5.status_code)


if __name__ == "__main__":
    print("=== without env ===")
    run_tests(with_env=False)
    print("\n=== with env ===")
    run_tests(with_env=True)
