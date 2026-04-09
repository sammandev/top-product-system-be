from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.dependencies.authz import get_current_user
from app.main import app
from app.routers import compare as compare_router
from app.routers import dvt_mc2_converter as dvt_mc2_router
from app.routers import external_api_client as external_api_router
from app.routers import format_compare as format_compare_router
from app.routers import iplas_proxy as iplas_router
from app.routers import multi_dut_analysis as multi_dut_router
from app.routers import scoring as scoring_router
from app.routers import test_log as test_log_router
from app.routers.external_api_client import get_user_dut_client


BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
client = TestClient(app)


def load_bytes(relative_path: str) -> bytes:
    with (BASE_DATA_DIR / relative_path).open("rb") as handle:
        return handle.read()


class _StubResponse:
    def __init__(self, status_code: int, *, text: str = "", json_data=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data if json_data is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._json_data


class _StubAsyncClient:
    def __init__(self, *, get_response=None, post_response=None, get_error=None, post_error=None):
        self._get_response = get_response
        self._post_response = post_response
        self._get_error = get_error
        self._post_error = post_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *args, **kwargs):
        if self._get_error is not None:
            raise self._get_error
        return self._get_response

    async def post(self, *args, **kwargs):
        if self._post_error is not None:
            raise self._post_error
        return self._post_response


def test_iplas_csv_test_items_defaults_to_compact_records(monkeypatch):
    async def fake_fetch(_request, _redis):
        return (
            [
                {
                    "Site": "PTB",
                    "Project": "HH5K",
                    "station": "Wireless_Test_2_5G",
                    "TSP": "TSP1",
                    "Model": "MODEL",
                    "MO": "MO1",
                    "Line": "L1",
                    "ISN": "ISN1",
                    "DeviceId": "DEV1",
                    "Test Status": "PASS",
                    "Test Start Time": "2026-01-01T00:00:00Z",
                    "Test end Time": "2026-01-01T00:10:00Z",
                    "ErrorCode": "",
                    "ErrorName": "",
                    "TestItem": [
                        {
                            "NAME": "ITEM1",
                            "STATUS": "PASS",
                            "VALUE": "1",
                            "UCL": "2",
                            "LCL": "0",
                            "CYCLE": "",
                        }
                    ],
                }
            ],
            False,
            1,
            1,
            False,
            True,
            iplas_router.StationRangeResponseMetadata(
                cache_coverage="full",
                validated_until=None,
                bucket_stats=[],
            ),
        )

    monkeypatch.setattr(iplas_router, "get_redis_client", lambda: None)
    monkeypatch.setattr(iplas_router, "_fetch_station_range_for_request", fake_fetch)

    response = client.post(
        "/api/iplas/csv-test-items",
        json={
            "site": "PTB",
            "project": "HH5K",
            "station": "Wireless_Test_2_5G",
            "device_id": "ALL",
            "begin_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T01:00:00Z",
            "test_status": "PASS",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["includes_test_items"] is False
    assert payload["data"][0]["TestItemCount"] == 1
    assert "TestItem" not in payload["data"][0]


def test_iplas_csv_test_items_include_test_items_opt_in(monkeypatch):
    async def fake_fetch(_request, _redis):
        return (
            [
                {
                    "Site": "PTB",
                    "Project": "HH5K",
                    "station": "Wireless_Test_2_5G",
                    "TSP": "TSP1",
                    "Model": "MODEL",
                    "MO": "MO1",
                    "Line": "L1",
                    "ISN": "ISN1",
                    "DeviceId": "DEV1",
                    "Test Status": "PASS",
                    "Test Start Time": "2026-01-01T00:00:00Z",
                    "Test end Time": "2026-01-01T00:10:00Z",
                    "ErrorCode": "",
                    "ErrorName": "",
                    "TestItem": [
                        {
                            "NAME": "ITEM1",
                            "STATUS": "PASS",
                            "VALUE": "1",
                            "UCL": "2",
                            "LCL": "0",
                            "CYCLE": "",
                        }
                    ],
                }
            ],
            False,
            1,
            1,
            False,
            True,
            iplas_router.StationRangeResponseMetadata(
                cache_coverage="full",
                validated_until=None,
                bucket_stats=[],
            ),
        )

    monkeypatch.setattr(iplas_router, "get_redis_client", lambda: None)
    monkeypatch.setattr(iplas_router, "_fetch_station_range_for_request", fake_fetch)

    response = client.post(
        "/api/iplas/csv-test-items",
        json={
            "site": "PTB",
            "project": "HH5K",
            "station": "Wireless_Test_2_5G",
            "device_id": "ALL",
            "begin_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T01:00:00Z",
            "test_status": "PASS",
            "include_test_items": True,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["includes_test_items"] is True
    assert payload["data"][0]["TestItem"][0]["NAME"] == "ITEM1"


def test_compare_endpoint_hides_internal_error(monkeypatch, tmp_path):
    def explode(*args, **kwargs):
        raise RuntimeError("compare secret details")

    monkeypatch.setattr(compare_router, "_compare_dfs", explode)

    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text("x,y\n1,2\n", encoding="utf-8")
    file_b.write_text("x,y\n1,3\n", encoding="utf-8")

    with file_a.open("rb") as handle_a:
        upload_a = client.post("/api/upload-preview", files={"file": ("a.csv", handle_a, "text/csv")})
    with file_b.open("rb") as handle_b:
        upload_b = client.post("/api/upload-preview", files={"file": ("b.csv", handle_b, "text/csv")})

    assert upload_a.status_code == 200
    assert upload_b.status_code == 200

    response = client.post(
        "/api/compare",
        data={
            "file_a": upload_a.json()["file_id"],
            "file_b": upload_b.json()["file_id"],
            "mode": "both",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Comparison request could not be processed"
    assert "secret" not in response.json()["detail"].lower()


def test_compare_formats_hides_internal_parser_details(monkeypatch):
    monkeypatch.setattr(format_compare_router, "parse_mastercontrol_text", lambda *_args, **_kwargs: {})

    def explode(*args, **kwargs):
        raise RuntimeError("dvt parser secret details")

    monkeypatch.setattr(format_compare_router, "parse_wifi_dvt_text", explode)

    response = client.post(
        "/api/compare-formats",
        files={
            "master_file": ("master.csv", b"header\nvalue\n", "text/csv"),
            "dvt_file": ("dvt.csv", b"header\nvalue\n", "text/csv"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Failed to parse uploaded DVT file"
    assert "secret" not in response.json()["detail"].lower()


def test_multi_dut_analysis_hides_internal_error(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("analysis secret details")

    monkeypatch.setattr(multi_dut_router, "analyze_mc2_with_spec", explode)

    response = client.post(
        "/api/analyze-multi-dut",
        files={
            "mc2_file": ("sample.csv", b"col1,col2\n1,2\n", "text/csv"),
            "spec_file": ("spec.json", b"{}", "application/json"),
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Multi-DUT analysis failed"
    assert "secret" not in response.json()["detail"].lower()


def test_dvt_converter_hides_internal_error(monkeypatch):
    monkeypatch.setattr(dvt_mc2_router, "parse_dvt_file", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dvt_mc2_router, "extract_serial_number", lambda *_args, **_kwargs: "SERIAL")
    monkeypatch.setattr(dvt_mc2_router, "determine_bands_from_rows", lambda *_args, **_kwargs: ["25G"])
    monkeypatch.setattr(dvt_mc2_router, "make_output_filename", lambda *_args, **_kwargs: "output.csv")

    def explode(*args, **kwargs):
        raise RuntimeError("conversion secret details")

    monkeypatch.setattr(dvt_mc2_router, "convert_dvt_to_mc2", explode)

    response = client.post(
        "/api/convert-dvt-to-mc2",
        files={"dvt_file": ("sample.csv", b"header\nvalue\n", "text/csv")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "DVT to MC2 conversion failed"
    assert "secret" not in response.json()["detail"].lower()


def test_test_log_parse_hides_internal_error(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("parse secret details")

    monkeypatch.setattr(test_log_router.TestLogParser, "parse_file_enhanced", explode)

    response = client.post(
        "/api/test-log/parse",
        files={"file": ("sample.txt", b"<< START TESTING >>  << START TESTING >>  << START TESTING >>  << START TESTING >>\n", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to parse uploaded test log"
    assert "secret" not in response.json()["detail"].lower()


def test_test_log_compare_hides_internal_error(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("compare secret details")

    monkeypatch.setattr(test_log_router.TestLogParser, "compare_files_enhanced", explode)

    response = client.post(
        "/api/test-log/compare",
        files=[
            ("files", ("sample.txt", b"<< START TESTING >>  << START TESTING >>  << START TESTING >>  << START TESTING >>\n", "text/plain")),
        ],
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to compare uploaded test logs"
    assert "secret" not in response.json()["detail"].lower()


def test_scoring_preview_hides_internal_error(monkeypatch):
    import app.services.scoring_service as scoring_service

    def explode(*args, **kwargs):
        raise RuntimeError("preview secret details")

    monkeypatch.setattr(scoring_service, "score_symmetrical", explode)

    response = client.post(
        "/api/scoring/preview",
        params={"value": 1.5, "scoring_type": "symmetrical", "ucl": 2.0, "lcl": 1.0},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Failed to calculate score preview"
    assert "secret" not in response.json()["detail"].lower()


def test_scoring_calculate_hides_internal_error(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(username="tester")

    def explode(*args, **kwargs):
        raise RuntimeError("calculate secret details")

    monkeypatch.setattr(scoring_router, "calculate_scores", explode)

    try:
        response = client.post(
            "/api/scoring/calculate",
            json={"records": []},
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to calculate scores"
    assert "secret" not in response.json()["detail"].lower()


def test_scoring_detect_types_hides_internal_error(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(username="tester")

    def explode(*args, **kwargs):
        raise RuntimeError("detect secret details")

    monkeypatch.setattr(scoring_router, "detect_scoring_type", explode)

    try:
        response = client.post(
            "/api/scoring/detect-types",
            json={"TestItem": [{"NAME": "ITEM_A", "VALUE": "1.0", "UCL": "2.0", "LCL": "0.0"}]},
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to detect scoring types"
    assert "secret" not in response.json()["detail"].lower()


def test_dut_sites_hides_internal_error(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(username="tester")
    app.dependency_overrides[get_user_dut_client] = lambda: object()

    def explode(*args, **kwargs):
        raise RuntimeError("dut sites secret details")

    monkeypatch.setattr(external_api_router, "_get_cached_sites", explode)

    try:
        response = client.get("/api/dut/sites")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_user_dut_client, None)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve DUT sites"
    assert "secret" not in response.json()["detail"].lower()


def test_iplas_stations_hides_upstream_body(monkeypatch):
    monkeypatch.setattr(iplas_router, "get_redis_client", lambda: None)
    monkeypatch.setattr(
        iplas_router.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(
            get_response=_StubResponse(status_code=502, text="upstream secret details"),
        ),
    )

    response = client.post(
        "/api/iplas/stations",
        json={"site": "PTB", "project": "HH5K"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "iPLAS v2 request failed"
    assert "secret" not in response.json()["detail"].lower()


def test_iplas_download_csv_hides_request_error(monkeypatch):
    request = iplas_router.httpx.Request("POST", "http://iplas.test/raw/get_test_log")
    request_error = iplas_router.httpx.RequestError("socket secret details", request=request)
    monkeypatch.setattr(
        iplas_router.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(post_error=request_error),
    )

    response = client.post(
        "/api/iplas/download-csv-log",
        json={
            "query_list": [
                {
                    "site": "PTB",
                    "project": "HH5K",
                    "station": "Wireless_Test_6G",
                    "line": "LINE1",
                    "model": "ALL",
                    "deviceid": "1234",
                    "isn": "ISN123",
                    "test_end_time": "2026/01/22 18:57:05.000",
                    "data_source": 0,
                }
            ]
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "iPLAS v1 service unavailable"
    assert "secret" not in response.json()["detail"].lower()