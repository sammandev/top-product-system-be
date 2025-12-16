import json
import os
import sys

import requests

BASE = os.environ.get("ASTPARSER_BASE", "http://127.0.0.1:8001")


def upload(path):
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f)}
        r = requests.post(f"{BASE}/api/upload-preview", files=files)
        r.raise_for_status()
        return r.json()


def parse_and_download(file_id, selected_columns):
    payload = {
        "file_id": file_id,
        "mode": "columns",
        "selected_columns": json.dumps(selected_columns),
        "exclude_columns": json.dumps([]),
    }
    r = requests.post(f"{BASE}/api/parse-download", files=payload)
    r.raise_for_status()
    with open("integration_parsed.csv", "wb") as out:
        out.write(r.content)
    print("Saved integration_parsed.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: integration_http.py <file-path>")
        sys.exit(1)
    info = upload(sys.argv[1])
    print("Uploaded:", info.get("file_id"))
    parse_and_download(info.get("file_id"), [0, 1, 2])
