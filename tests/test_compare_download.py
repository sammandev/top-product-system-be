import json

from fastapi.testclient import TestClient

from app.main import app


def test_compare_download_local(tmp_path):
    client = TestClient(app)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x,y\n1,2\n3,4\n")
    b.write_text("x,y\n1,2\n3,9\n")

    with open(str(a), "rb") as fa:
        r1 = client.post("/api/upload-preview", files={"file": ("a.csv", fa, "text/csv")})
    with open(str(b), "rb") as fb:
        r2 = client.post("/api/upload-preview", files={"file": ("b.csv", fb, "text/csv")})

    assert r1.status_code == 200 and r2.status_code == 200
    id1 = r1.json()["file_id"]
    id2 = r2.json()["file_id"]

    form = {
        "file_a": id1,
        "file_b": id2,
        "mode": "both",
        "a_selected_columns": json.dumps([0, 1]),
        "b_selected_columns": json.dumps([0, 1]),
    }
    r = client.post("/api/compare-download", data=form)
    assert r.status_code == 200
    assert "attachment" in r.headers.get("Content-Disposition", "")
    text = r.content.decode("utf-8")
    assert "A::x" in text or "B::x" in text
