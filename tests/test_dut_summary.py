from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.dependencies.authz import get_current_user
from app.main import app
from app.routers.external_api_client import get_user_dut_client


class _MockDUTClient:
    def __init__(
        self,
        payload,
        *,
        station_records=None,
        sites=None,
        models=None,
        stations=None,
    ):
        self._payload = payload
        self._station_records = station_records or {}
        self._sites = sites or []
        self._models = models or {}
        self._stations = stations or {}
        self.base_url = f"mock-summary-{id(self)}"

    async def get_dut_records(self, dut_isn: str):
        return self._payload

    async def get_station_records(self, station_id: int, dut_id: int):
        return self._station_records.get((station_id, dut_id), {"record": [], "data": []})

    async def get_sites(self):
        return self._sites

    async def get_models_by_site(self, site_id: int):
        return self._models.get(site_id, [])

    async def get_stations_by_model(self, model_id: int):
        return self._stations.get(model_id, [])


@pytest.fixture(autouse=True)
def reset_overrides():
    yield
    app.dependency_overrides.clear()


def _override_dependencies(payload, **kwargs):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(username="tester")
    app.dependency_overrides[get_user_dut_client] = lambda: _MockDUTClient(payload, **kwargs)


BASE_PAYLOAD = {
    "site_name": "ABS",
    "model_name": "ABST",
    "record_data": [
        {
            "id": 142,
            "name": "BP3 Download",
            "status": 1,
            "order": 10,
            "model_id": 44,
            "model_name": "ABST",
            "svn_name": "T0_T1",
            "svn_url": "http://example.com/svn",
            "dut_isn": "260884980003907",
            "dut_id": 9410441,
            "data": [
                {
                    "id": 26337392,
                    "test_date": "2025-09-15T12:20:42Z",
                    "test_duration": 136.0,
                    "test_result": 1,
                    "error_item": "",
                    "device_id": 1669,
                    "device_id__name": "614606",
                    "device_id__station_id": 142,
                    "device_id__station_id__model_id__site_id__name": "ABS",
                    "dut_id": 9410441,
                    "dut_id__isn": "260884980003907",
                    "site_name": "ABS",
                },
                {
                    "id": 26337393,
                    "test_date": "2025-09-16T12:20:42Z",
                    "test_duration": 100.0,
                    "test_result": 0,
                    "error_item": "ERROR",
                    "device_id": 1669,
                    "device_id__name": "614606",
                    "device_id__station_id": 142,
                    "device_id__station_id__model_id__site_id__name": "ABS",
                    "dut_id": 9410441,
                    "dut_id__isn": "260884980003907",
                    "site_name": "ABS",
                },
            ],
        },
        {
            "id": 200,
            "name": "Skip Station",
            "status": 1,
            "order": 11,
            "model_id": 44,
            "model_name": "ABST",
            "svn_name": None,
            "svn_url": None,
            "dut_isn": "260884980003907",
            "dut_id": 999999,
            "data": [],
        },
    ],
}


def test_dut_summary_aggregates_records():
    payload = BASE_PAYLOAD
    _override_dependencies(payload)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/summary",
        params={
            "dut_isn": "260884980003907",
            "site_identifier": "ABS",
            "model_identifier": "ABST",
        },
    )
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    assert data["dut_isn"] == "260884980003907"
    assert data["site_name"] == "ABS"
    assert data["model_name"] == "ABST"
    assert data["station_count"] == 2

    stations = {s["station_name"]: s for s in data["stations"]}
    station = stations["BP3 Download"]
    assert station["test_runs"] == 2
    assert station["dut_id"] == 9410441
    assert station["dut_isn"] == "260884980003907"
    assert len(station["devices"]) == 1

    device = station["devices"][0]
    assert device["device_name"] == "614606"
    assert device["total_runs"] == 2
    assert device["pass_runs"] == 1
    assert device["fail_runs"] == 1
    assert {result["status"] for result in device["results"]} == {"PASS", "FAIL"}


def test_dut_summary_returns_404_when_no_records():
    payload = {"site_name": "ABS", "model_name": "ABST", "record_data": []}
    _override_dependencies(payload)

    client = TestClient(app)
    resp = client.get("/api/dut/summary", params={"dut_isn": "UNKNOWN"})
    assert resp.status_code == 404


def test_dut_identifiers_endpoint_lists_unique_pairs():
    _override_dependencies(BASE_PAYLOAD)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/history/identifiers",
        params={
            "dut_isn": "260884980003907",
            "site_identifier": "ABS",
            "model_identifier": "ABST",
        },
    )
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    assert data["dut_isn"] == "260884980003907"
    ident_list = data["identifiers"]
    assert any(entry["dut_id"] == 9410441 for entry in ident_list)
    assert all(entry["station_name"] for entry in ident_list)
    assert all(entry["dut_isn"] == "260884980003907" for entry in ident_list if entry.get("dut_isn"))


def test_dut_progression_marks_stations_tested():
    _override_dependencies(BASE_PAYLOAD)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/history/progression",
        params={
            "dut_isn": "260884980003907",
            "site_identifier": "ABS",
            "model_identifier": "ABST",
        },
    )
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    stations = {s["station_name"]: s for s in data["stations"]}
    assert stations["BP3 Download"]["tested"] is True
    assert stations["BP3 Download"]["test_runs"] == 2
    assert stations["Skip Station"]["tested"] is False
    assert stations["Skip Station"]["test_runs"] == 0


def test_dut_results_reports_pass_fail_counts():
    _override_dependencies(BASE_PAYLOAD)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/history/results",
        params={
            "dut_isn": "260884980003907",
            "site_identifier": "ABS",
            "model_identifier": "ABST",
        },
    )
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    station = next(s for s in data["stations"] if s["station_name"] == "BP3 Download")
    assert station["test_runs"] == 2
    assert station["pass_runs"] == 1
    assert station["fail_runs"] == 1
    statuses = {result["status"] for result in station["results"]}
    assert statuses == {"PASS", "FAIL"}


def test_dut_isn_variants_endpoint_returns_unique_isns():
    _override_dependencies(BASE_PAYLOAD)

    client = TestClient(app)
    resp = client.get("/api/dut/history/isns", params={"dut_isn": "260884980003907"})
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    assert data["dut_isn"] == "260884980003907"
    assert data["isns"] == ["260884980003907"]


def test_dut_isn_variants_preserves_discovery_order():
    payload = {
        "site_name": "PTB",
        "model_name": "HH5K",
        "record_data": [
            {
                "id": 1,
                "name": "Station A",
                "status": 1,
                "order": 1,
                "model_id": 10,
                "site_name": "PTB",
                "dut_isn": "DM2527470003581",
                "data": [
                    {"dut_id__isn": "DM2527470003581"},
                    {"dut_id__isn": "BCD5EDF112CF"},
                ],
            }
        ],
    }
    _override_dependencies(payload)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/history/isns",
        params={
            "dut_isn": "262034740002573",
            "site_identifier": "PTB",
            "model_identifier": "HH5K",
        },
    )
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    assert data["isns"] == [
        "262034740002573",
        "DM2527470003581",
        "BCD5EDF112CF",
    ]


def test_dut_summary_exposes_single_site_parameter():
    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/api/dut/summary")
    query_param_aliases = [param.alias for param in route.dependant.query_params]
    assert "site_identifier" in query_param_aliases
    assert "site_id" not in query_param_aliases
    assert "site_name" not in query_param_aliases


def test_models_endpoint_accepts_site_name():
    sites = [{"id": 2, "name": "TY"}]
    models = {2: [{"id": 10, "name": "MODEL_A", "site_id": 2}]}
    _override_dependencies(BASE_PAYLOAD, sites=sites, models=models)

    client = TestClient(app)
    resp = client.get("/api/dut/sites/TY/models")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data == models[2]


def test_stations_endpoint_accepts_model_name():
    sites = [{"id": 2, "name": "TY"}]
    models = {2: [{"id": 10, "name": "MODEL_A", "site_id": 2}]}
    stations = {10: [{"id": 100, "name": "StationA", "status": 1, "order": 1, "model_id": 10}]}
    _override_dependencies(BASE_PAYLOAD, sites=sites, models=models, stations=stations)

    client = TestClient(app)
    resp = client.get("/api/dut/models/MODEL_A/stations")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data[0]["id"] == 100
    assert data[0]["name"] == "StationA"


def test_station_records_endpoint_returns_payload():
    station_payload = {
        "record": [
            {
                "test_date": "2025-09-16T13:32:25Z",
                "device": "614621",
                "station": "ZIGBEE_ZWAVE",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "PTB",
            }
        ],
        "data": [["ITEM", 23.0, 15.0, "0.26", "18.18", "18.36"]],
    }
    station_map = {(142, 9410441): station_payload}
    _override_dependencies(BASE_PAYLOAD, station_records=station_map)

    client = TestClient(app)
    resp = client.get("/api/dut/records/142/9410441")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data == station_payload


def test_station_records_endpoint_supports_name_and_isn():
    station_payload = {
        "record": [
            {
                "test_date": "2025-09-16T13:32:25Z",
                "device": "614621",
                "station": "BP3 Download",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "ABS",
            }
        ],
        "data": [["ITEM", 23.0, 15.0, "0.26", "18.18", "18.36"]],
    }
    station_map = {(142, 9410441): station_payload}
    _override_dependencies(BASE_PAYLOAD, station_records=station_map)

    client = TestClient(app)
    resp = client.get("/api/dut/records/BP3%20Download/260884980003907")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data == station_payload


def test_latest_station_record_endpoint_returns_latest_entry():
    station_payload = {
        "record": [
            {
                "test_date": "2025-09-15T13:32:25Z",
                "device": "614621",
                "station": "BP3 Download",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "ABS",
            },
            {
                "test_date": "2025-09-16T13:32:25Z",
                "device": "614622",
                "station": "BP3 Download",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "ABS",
            },
        ],
        "data": [["ITEM", 23.0, 15.0, "0.26", "18.18", "18.36"]],
    }
    station_map = {(142, 9410441): station_payload}
    _override_dependencies(BASE_PAYLOAD, station_records=station_map)

    client = TestClient(app)
    resp = client.get("/api/dut/records/latest/BP3%20Download/260884980003907")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert len(data["record"]) == 1
    assert data["record"][0]["device"] == "614622"


def test_latest_tests_endpoint_returns_latest_per_station():
    station_payload = {
        "record": [
            {
                "test_date": "2025-09-15T13:32:25Z",
                "device": "614621",
                "station": "BP3 Download",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "ABS",
            },
            {
                "test_date": "2025-09-16T13:32:25Z",
                "device": "614606",
                "station": "BP3 Download",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "ABS",
            },
        ],
        "data": [
            ["ITEM", 23.0, 15.0, "18.36"],
            ["ITEM2", 23.0, 15.0, "17.45"],
        ],
    }
    station_map = {(142, 9410441): station_payload}
    _override_dependencies(BASE_PAYLOAD, station_records=station_map)

    client = TestClient(app)
    resp = client.get("/api/dut/history/latest-tests", params={"dut_isn": "260884980003907"})
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["dut_isn"] == "260884980003907"
    assert data["station_count"] == 1
    assert data["record_data"]
    entry = data["record_data"][0]
    assert entry["station_name"] == "BP3 Download"
    assert entry["device"] == "614606"
    assert entry["test_date"].startswith("2025-09-16")
    assert entry["data"] == station_payload["data"]


def test_latest_tests_endpoint_filters_by_station_identifier():
    station_payload = {
        "record": [
            {
                "test_date": "2025-09-16T13:32:25Z",
                "device": "614606",
                "station": "BP3 Download",
                "isn": "260884980003907",
                "error_item": "",
                "site_name": "ABS",
            },
        ],
        "data": [["ITEM", 23.0, 15.0, "18.36"]],
    }
    station_map = {(142, 9410441): station_payload}
    _override_dependencies(BASE_PAYLOAD, station_records=station_map)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/history/latest-tests",
        params=[("dut_isn", "260884980003907"), ("stations", "142")],
    )
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["station_count"] == 1
    assert len(data["record_data"]) == 1
    assert data["record_data"][0]["data"] == station_payload["data"]


def test_latest_tests_endpoint_returns_404_when_no_matching_stations():
    _override_dependencies(BASE_PAYLOAD)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/history/latest-tests",
        params=[("dut_isn", "260884980003907"), ("stations", "Skip Station")],
    )
    assert resp.status_code == 404
