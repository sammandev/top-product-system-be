import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.helpers import _TEMP_UPLOADS, UPLOAD_DIR

client = TestClient(app)


def test_inmemory_expiration(monkeypatch):
    # Disable disk persistence: use in-memory storage
    monkeypatch.setenv("UPLOAD_PERSIST", "0")
    _TEMP_UPLOADS.clear()
    # Upload a small CSV
    data = b"A,B\n1,2\n"
    response = client.post("/api/upload-preview", files={"file": ("temp.csv", data)})
    assert response.status_code == 200
    result = response.json()
    fid = result["file_id"]
    # File should not exist on disk
    disk_path = os.path.join(UPLOAD_DIR, fid)
    assert not os.path.exists(disk_path)
    # Immediate cleanup with zero TTL override
    cleanup_resp = client.post("/api/cleanup-uploads", data={"ttl": 0})
    assert cleanup_resp.status_code == 200
    removed = cleanup_resp.json().get("removed", [])
    assert fid in removed
    # In-memory store should no longer contain the file
    assert fid not in _TEMP_UPLOADS


def test_disk_expiration(monkeypatch):
    # Enable disk persistence
    monkeypatch.setenv("UPLOAD_PERSIST", "1")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    data = b"X,Y\n3,4\n"
    response = client.post("/api/upload-preview", files={"file": ("temp2.csv", data)})
    assert response.status_code == 200
    result = response.json()
    fid = result["file_id"]
    # File should be on disk
    disk_path = os.path.join(UPLOAD_DIR, fid)
    assert os.path.exists(disk_path)
    # Immediate cleanup with zero TTL override
    cleanup_resp = client.post("/api/cleanup-uploads", data={"ttl": 0})
    assert cleanup_resp.status_code == 200
    removed = cleanup_resp.json().get("removed", [])
    assert fid in removed
    # File removed from disk
    assert not os.path.exists(disk_path)


def test_upload_sanitizes_filename(monkeypatch):
    monkeypatch.setenv("UPLOAD_PERSIST", "0")
    _TEMP_UPLOADS.clear()

    response = client.post("/api/upload-preview", files={"file": ("..\\evil.csv", b"A,B\n1,2\n")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "evil.csv"
    assert payload["file_id"].endswith("_evil.csv")
    assert "/" not in payload["file_id"]
    assert "\\" not in payload["file_id"]


def test_upload_rejects_large_files(monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_SIZE_MB", "1")

    oversized = b"a" * ((1024 * 1024) + 1)
    response = client.post("/api/upload-preview", files={"file": ("large.csv", oversized)})

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded file exceeds the 1 MB limit"


def test_upload_preview_hides_parser_details_for_invalid_excel(monkeypatch):
    monkeypatch.setenv("UPLOAD_PERSIST", "0")

    response = client.post(
        "/api/upload-preview",
        files={"file": ("broken.xlsx", b"this-is-not-a-valid-excel-file")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Failed to parse uploaded file preview"


def test_parse_rejects_invalid_file_identifier():
    response = client.post("/api/parse", data={"file_id": "../escape.csv", "mode": "columns"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file identifier"


if __name__ == "__main__":
    pytest.main([__file__])
