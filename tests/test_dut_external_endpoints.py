from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.dependencies.authz import get_current_user
from app.main import app
from app.routers.external_api_client import get_user_dut_client


class _MockDUTClient:
    def __init__(
        self,
        *,
        devices=None,
        period_devices=None,
        test_items=None,
        device_results=None,
        model_summary=None,
        nonvalue_records=None,
        nonvalue_bin_records=None,
        dut_records=None,
        latest_nonvalue=None,
        latest_nonvalue_bin=None,
        station_records=None,
        latest_records=None,
        sites=None,
        models=None,
        stations=None,
    ):
        self._devices = devices or {}
        self._period_devices = period_devices or {}
        self._test_items = test_items or {}
        self._device_results = device_results or []
        self._model_summary = model_summary or {}
        self._nonvalue_records = nonvalue_records or {}
        self._nonvalue_bin_records = nonvalue_bin_records or {}
        self._dut_records = dut_records or {"record_data": []}
        self._latest_nonvalue = latest_nonvalue or {}
        self._latest_nonvalue_bin = latest_nonvalue_bin or {}
        self._station_records = station_records or {}
        self._latest_records = latest_records or {}
        self._sites = sites or []
        self._models = models or {}
        self._stations = stations or {}
        self.base_url = f"mock-{id(self)}"
        self.last_period_query = None
        self.last_device_payload = None
        self.last_summary_payload = None

    async def get_devices_by_station(self, station_id: int):
        return self._devices.get(station_id, [])

    async def get_devices_by_period(self, station_id, start_time, end_time, test_result):
        self.last_period_query = (station_id, start_time, end_time, test_result)
        return self._period_devices.get(station_id, [])

    async def get_test_items_by_station(self, station_id: int):
        return self._test_items.get(station_id, [])

    async def get_station_records(self, station_id: int, dut_id: int):
        return self._station_records.get((station_id, dut_id), {"record": [], "data": []})

    async def get_test_results_by_device(self, payload):
        self.last_device_payload = payload
        return self._device_results

    async def get_model_summary(self, payload):
        self.last_summary_payload = payload
        return self._model_summary

    async def get_sites(self):
        return self._sites

    async def get_models_by_site(self, site_id: int):
        return self._models.get(site_id, [])

    async def get_stations_by_model(self, model_id: int):
        return self._stations.get(model_id, [])

    async def get_station_nonvalue_records(self, station_id: int, dut_id: int):
        return self._nonvalue_records.get((station_id, dut_id), {"record": [], "data": []})

    async def get_station_nonvalue_bin_records(self, station_id: int, dut_id: int):
        return self._nonvalue_bin_records.get((station_id, dut_id), {"record": [], "data": []})

    async def get_dut_records(self, dut_isn: str):
        return self._dut_records

    async def get_latest_station_records(self, station_id: int, dut_id: int):
        if (station_id, dut_id) in self._latest_records:
            return self._latest_records[(station_id, dut_id)]
        return self._station_records.get((station_id, dut_id), {"record": [], "data": []})

    async def get_latest_nonvalue_record(self, station_id: int, dut_id: int):
        return self._latest_nonvalue.get((station_id, dut_id), {"record": [], "data": []})

    async def get_latest_nonvalue_bin_record(self, station_id: int, dut_id: int):
        return self._latest_nonvalue_bin.get((station_id, dut_id), {"record": [], "data": []})


@pytest.fixture(autouse=True)
def reset_overrides():
    """Clear dependency overrides and metadata cache before and after each test."""
    # Clear BEFORE test to ensure clean state
    from app.services import dut_metadata_cache

    dut_metadata_cache._local_cache.clear()
    dut_metadata_cache._locks.clear()
    app.dependency_overrides.clear()

    yield

    # Clear AFTER test to prevent pollution
    app.dependency_overrides.clear()
    dut_metadata_cache._local_cache.clear()
    dut_metadata_cache._locks.clear()


def _override_client(mock_client: _MockDUTClient):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(username="tester")
    app.dependency_overrides[get_user_dut_client] = lambda: mock_client


def test_devices_by_station_accepts_name_resolution():
    mock = _MockDUTClient(
        devices={148: [{"id": 2302, "name": "614754"}]},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "BP3 Download"}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/stations/BP3%20Download/devices")
    assert resp.status_code == 200, resp.json()

    payload = resp.json()
    assert payload["station_name"] == "BP3 Download"
    assert payload["site_name"] == "PTB"
    assert payload["model_name"] == "HH5K"
    assert len(payload["data"]) == 1
    entry = payload["data"][0]
    assert entry["id"] == 2302
    assert entry["device_name"] == "614754"


def test_devices_by_station_period_returns_payload_and_normalises_result():
    mock = _MockDUTClient(
        period_devices={148: [{"id": 1255, "name": "614750", "status": 2, "station_id": 148}]},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "BP3 Download"}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get(
        "/api/dut/stations/148/devices/period",
        params={
            "start_time": "2025-10-12T00:00:00",
            "end_time": "2025-10-12T01:00:00",
            "test_result": "fail",
        },
    )
    assert resp.status_code == 200, resp.json()
    payload = resp.json()
    assert payload["station_id"] == 148
    assert payload["site_name"] == "PTB"
    assert payload["model_name"] == "HH5K"
    assert len(payload["data"]) == 1
    entry = payload["data"][0]
    assert entry["id"] == 1255
    assert mock.last_period_query is not None
    assert mock.last_period_query[0] == 148
    # ensure result filter was upper-cased
    assert mock.last_period_query[3] == "FAIL"


def test_devices_by_station_period_enforces_time_order():
    mock = _MockDUTClient()
    _override_client(mock)
    client = TestClient(app)
    resp = client.get(
        "/api/dut/stations/148/devices/period",
        params={
            "start_time": "2025-10-12T02:00:00",
            "end_time": "2025-10-12T01:00:00",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "end_time must be greater than start_time"


def test_test_items_by_station_returns_payload():
    mock = _MockDUTClient(
        test_items={148: [{"id": 204025, "name": "BRCM_DEFAULT_RXGAINERR", "upperlimit": 0.0, "lowerlimit": 0.0}]},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "BP3 Download"}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/stations/148/test-items")
    assert resp.status_code == 200, resp.json()
    payload = resp.json()
    assert payload["station_id"] == 148
    assert payload["station_name"] == "BP3 Download"
    assert payload["site_name"] == "PTB"
    assert payload["model_name"] == "HH5K"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["name"] == "BRCM_DEFAULT_RXGAINERR"


def test_device_results_endpoint_requires_payload():
    mock = _MockDUTClient()
    _override_client(mock)
    client = TestClient(app)
    # Send incomplete form data (missing required fields)
    resp = client.post("/api/dut/history/device-results", data={"test_result": "ALL"})
    # FastAPI returns 422 for missing required Form fields
    assert resp.status_code == 422
    # Check that error mentions missing fields
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    required_fields = {"site_id", "model_id", "device_id", "start_time", "end_time"}
    missing_fields = {item["loc"][-1] for item in detail if item["type"] == "missing"}
    assert required_fields.issubset(missing_fields)


def test_device_results_endpoint_returns_data():
    device_results = [
        {
            "id": 30405525,
            "device": "614660",
            "device_id__name": "614660",
            "dut_id__isn": "DM2524470055736",
            "dut_id": 10607995,
            "test_date": "2025-10-11T01:24:27Z",
            "test_result": 1,
            "station_id__name": "Wireless_Test_2_5G",
            "test_duration": 412.0,
            "error_item": "",
            "status": "Pass",
        }
    ]
    mock = _MockDUTClient(device_results=device_results)
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "start_time": "2025-10-11T01:00:47.000Z",
        "end_time": "2025-10-14T06:30:56.000Z",
        "site_id": "PTB",
        "model_id": "HH5K",
        "device_id": "2324",
        "test_result": "ALL",
        "model": "",
    }
    resp = client.post("/api/dut/history/device-results", data=form_payload)
    assert resp.status_code == 200, resp.json()

    data = resp.json()
    assert len(data) == 1
    for key, value in device_results[0].items():
        assert data[0][key] == value
    assert mock.last_device_payload == {
        "start_time": "2025-10-11T01:00:47.000Z",
        "end_time": "2025-10-14T06:30:56.000Z",
        "device_id": "2324",
        "test_result": "ALL",
    }


def test_device_results_endpoint_resolves_numeric_device_name():
    device_results = [
        {
            "id": 999,
            "device": "614650",
            "device_id__name": "614650",
            "dut_id__isn": "DM2600000000000",
            "dut_id": 10411910,
            "test_date": "2025-10-20T02:05:22Z",
            "test_result": 0,
            "station_id__name": "Wireless_Test_6G",
            "test_duration": 69.0,
            "error_item": "",
            "status": "Retest",
        }
    ]
    mock = _MockDUTClient(
        device_results=device_results,
        devices={148: [{"id": 1351, "name": "614650"}]},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "Wireless_Test_6G"}]},
    )
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "start_time": "2025-10-20T01:00:00.000Z",
        "end_time": "2025-10-21T01:00:00.000Z",
        "site_id": "PTB",
        "model_id": "HH5K",
        "device_id": "614650",
        "test_result": "ALL",
        "model": "",
    }
    resp = client.post("/api/dut/history/device-results", data=form_payload)
    assert resp.status_code == 200, resp.json()

    assert mock.last_device_payload == {
        "start_time": "2025-10-20T01:00:00.000Z",
        "end_time": "2025-10-21T01:00:00.000Z",
        "device_id": "1351",
        "test_result": "ALL",
    }

    data = resp.json()
    assert len(data) == 1
    assert data[0]["device_id__name"] == "614650"


def test_device_results_endpoint_includes_model_when_provided():
    device_results = [
        {
            "id": 555,
            "device": "614601",
            "device_id__name": "614601",
            "dut_id__isn": "DM2600000001111",
            "dut_id": 10411911,
            "test_date": "2025-10-20T03:05:22Z",
            "test_result": 1,
            "station_id__name": "Wireless_Test_6G",
            "test_duration": 70.0,
            "error_item": "",
            "status": "Pass",
        }
    ]
    mock = _MockDUTClient(device_results=device_results)
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "site_id": "PTB",
        "model_id": "HH5K",
        "device_id": "2324",
        "test_result": "FAIL",
        "model": "HH5K",
    }
    resp = client.post("/api/dut/history/device-results", data=form_payload)
    assert resp.status_code == 200, resp.json()

    assert mock.last_device_payload == {
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "device_id": "2324",
        "test_result": "FAIL",
        "model": "HH5K",
    }


def test_device_results_endpoint_uses_station_hint_to_avoid_site_scan():
    class HintMock(_MockDUTClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.get_sites_called = False

        async def get_sites(self):
            self.get_sites_called = True
            return await super().get_sites()

    mock = HintMock(
        device_results=[
            {
                "id": 123,
                "device": "614754",
                "device_id__name": "614754",
                "dut_id__isn": "DM2600000002222",
                "dut_id": 108,
                "test_date": "2025-10-20T05:05:22Z",
                "test_result": 0,
                "station_id__name": "Wireless_Test_6G",
                "test_duration": 45.0,
                "error_item": "",
                "status": "Retest",
            }
        ],
        devices={148: [{"id": 1351, "name": "614754"}]},
        stations={44: [{"id": 148, "name": "Wireless_Test_6G"}]},
        models={2: [{"id": 44, "name": "HH5K"}]},
    )
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "site_id": "PTB",
        "model_id": "HH5K",
        "device_id": "614754",
        "test_result": "ALL",
        "model": "",
        "station_identifier": "148",
    }
    resp = client.post("/api/dut/history/device-results", data=form_payload)
    assert resp.status_code == 200, resp.json()
    assert mock.last_device_payload["device_id"] == "1351"
    assert mock.get_sites_called is False


def test_device_results_endpoint_handles_multiple_devices():
    device_results_a = [
        {"id": 1, "device": "614650", "device_id__name": "614650", "test_result": 1},
    ]
    device_results_b = [
        {"id": 2, "device": "614651", "device_id__name": "614651", "test_result": 0},
    ]

    class RecordingMock(_MockDUTClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.device_payloads = []
            self._results_by_device = {"1351": device_results_a, "1352": device_results_b}
            # Override base_url to a fixed value to avoid cache issues
            self.base_url = "mock-test-handles-multiple-devices"

        async def get_test_results_by_device(self, payload):
            self.last_device_payload = payload
            self.device_payloads.append(payload)
            return self._results_by_device.get(payload["device_id"], [])

    mock = RecordingMock(
        devices={148: [{"id": 1351, "name": "614650"}, {"id": 1352, "name": "614651"}]},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "Wireless_Test_6G"}]},
    )
    # Set up overrides BEFORE creating client
    _override_client(mock)
    client = TestClient(app)
    # Use data parameter with dict containing list for device_id field
    form_data = {
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "site_id": "PTB",
        "model_id": "HH5K",
        "device_id": ["614650", "614651"],
        "test_result": "ALL",
        "model": "",
    }

    resp = client.post(
        "/api/dut/history/device-results",
        data=form_data,
    )
    assert resp.status_code == 200, resp.json()

    payload = resp.json()
    assert len(payload) == 2, f"Expected 2 results but got {len(payload)}: {payload}"
    names = sorted(entry["device_id__name"] for entry in payload)
    assert names == ["614650", "614651"]
    assert len(mock.device_payloads) == 2
    assert {p["device_id"] for p in mock.device_payloads} == {"1351", "1352"}


def test_model_summary_endpoint_validates_response():
    summary_payload = {
        "model_id": 44,
        "model_name": "HH5K",
        "site_name": "PTB",
        "total_yield_rate": 98.89,
        "total_retest_rate": 2.8,
        "total_UPH": 1490,
        "station_info": [
            {
                "station_id": 220,
                "station_order": 80,
                "station_name": "NOISE_FLOOR",
                "test_times": 1191,
                "pass_count": 1190,
                "fail_count": 1,
                "retest_count": 0,
                "yield_rate": 99.92,
                "retest_rate": 3.61,
                "device_summary": [
                    {
                        "device": "615920",
                        "device_id": 2283,
                        "pass_count": 252,
                        "fail_count": 2,
                        "line": "6PTB3F20A",
                        "error_items": [{"error_item": "Check_Noise", "fail_count": 1}],
                        "retest_dut": ["A039F92CD2A7"],
                        "retest_dut2": [{"dut_id": 10610988, "dut_id__isn": "A039F92CA827"}],
                        "retest_ratio": 1.0,
                        "retest_count": 2,
                        "retest_count_unique": 2,
                    }
                ],
            }
        ],
        "model_type_list": [""],
    }
    mock = _MockDUTClient(model_summary=summary_payload)
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "model_id": "44",
        "start_time": "2025-10-11T01:00:47.000Z",
        "end_time": "2025-10-11T12:10:47.000Z",
        "model": "",
        "site_id": "PTB",
    }
    resp = client.post("/api/dut/history/model-summary", data=form_payload)
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["model_id"] == 44
    assert data["station_info"][0]["station_name"] == "NOISE_FLOOR"
    assert mock.last_summary_payload == {
        "model_id": 44,
        "start_time": "2025-10-11T01:00:47.000Z",
        "end_time": "2025-10-11T12:10:47.000Z",
    }
    assert "model" not in mock.last_summary_payload


def test_model_summary_endpoint_accepts_model_name():
    summary_payload = {
        "model_id": 44,
        "model_name": "HH5K",
        "site_name": "PTB",
        "total_yield_rate": 90.52,
        "total_retest_rate": 7.33,
        "total_UPH": 191,
        "station_info": [],
    }
    mock = _MockDUTClient(
        model_summary=summary_payload,
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
    )
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "model_id": "HH5K",
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "model": "",
        "site_id": "PTB",
    }
    resp = client.post("/api/dut/history/model-summary", data=form_payload)
    assert resp.status_code == 200, resp.json()

    assert mock.last_summary_payload == {
        "model_id": 44,
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
    }
    assert "model" not in mock.last_summary_payload

    payload = resp.json()
    assert payload["model_id"] == 44
    assert payload["model_name"] == "HH5K"


def test_model_summary_endpoint_keeps_model_value():
    summary_payload = {
        "model_id": 44,
        "model_name": "HH5K",
        "site_name": "PTB",
        "total_yield_rate": 90.52,
        "total_retest_rate": 7.33,
        "total_UPH": 191,
        "station_info": [],
    }
    mock = _MockDUTClient(model_summary=summary_payload)
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "model_id": "44",
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "model": "HH5K",
        "site_id": "PTB",
    }
    resp = client.post("/api/dut/history/model-summary", data=form_payload)
    assert resp.status_code == 200, resp.json()

    assert mock.last_summary_payload == {
        "model_id": 44,
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "model": "HH5K",
    }

    payload = resp.json()
    assert payload["model_id"] == 44
    assert payload["model_name"] == "HH5K"


def test_model_summary_endpoint_respects_site_hint():
    summary_payload = {
        "model_id": 44,
        "model_name": "HH5K",
        "site_name": "PTB",
        "total_yield_rate": 90.52,
        "total_retest_rate": 7.33,
        "total_UPH": 191,
        "station_info": [],
    }

    class HintMock(_MockDUTClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.get_sites_called = False

        async def get_sites(self):
            self.get_sites_called = True
            return await super().get_sites()

    mock = HintMock(
        model_summary=summary_payload,
        models={2: [{"id": 44, "name": "HH5K", "site_id": 2}]},
        sites=[{"id": 2, "name": "PTB"}],
    )
    _override_client(mock)

    client = TestClient(app)
    form_payload = {
        "model_id": "HH5K",
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
        "model": "",
        "site_id": "2",
    }
    resp = client.post("/api/dut/history/model-summary", data=form_payload)
    assert resp.status_code == 200, resp.json()
    assert mock.last_summary_payload == {
        "model_id": 44,
        "start_time": "2025-10-21T01:00:00.000Z",
        "end_time": "2025-10-21T02:00:00.000Z",
    }
    assert mock.get_sites_called is False


def test_latest_test_items_batch_returns_structured_payload():
    value_payload = {
        "record": [{"test_date": "2025-01-02T00:00:00Z"}],
        "data": [
            ["WiFi_TX1_POW", "25", "17", "20.1", "21.0"],
            ["WiFi_TX2_POW", 26, 16, "20.4", "21.0"],
        ],
    }
    nonvalue_payload = {
        "record": [],
        "data": [
            ["TEST_RESULT", "", "", "PASS"],
            ["TEMPERATURE_CHECK", None, None, "OK"],
        ],
    }
    nonvalue_bin_payload = {
        "record": [],
        "data": [
            ["WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "0x237d", "0x244c", "0x235g"],
            ["WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "0x244c", "0x244c", "0x247e"],
        ],
    }
    dut_records = {
        "site_name": "PTB",
        "model_name": "CALIX",
        "record_data": [
            {
                "id": 148,
                "name": "StationX",
                "dut_isn": "ISN_B",
                "dut_id": 333,
                "data": [
                    {
                        "test_date": "2025-01-02T00:00:00Z",
                        "test_result": 1,
                        "device_id": 2,
                        "device_id__name": "DeviceB",
                        "dut_id": 333,
                        "dut_id__isn": "ISN_B",
                    }
                ],
            }
        ],
    }
    mock = _MockDUTClient(
        dut_records=dut_records,
        latest_records={(148, 333): value_payload},
        nonvalue_records={(148, 333): nonvalue_payload},
        nonvalue_bin_records={(148, 333): nonvalue_bin_payload},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "CALIX", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "StationX", "site_id": 2, "model_id": 44, "status": 1, "order": 10}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.post(
        "/api/dut/test-items/latest/batch",
        json={
            "dut_isn": "ISN_B",
            "station_identifiers": ["StationX"],
            "site_identifier": "PTB",
            "model_identifier": "CALIX",
        },
    )
    assert resp.status_code == 200, resp.json()
    payload = resp.json()
    assert payload["dut_isn"] == "ISN_B"
    assert len(payload["stations"]) == 1
    station = payload["stations"][0]
    assert station["station_name"] == "StationX"
    assert [item["name"] for item in station["value_test_items"]] == ["WiFi_TX1_POW", "WiFi_TX2_POW"]
    assert station["value_test_items"][0]["usl"] == 25.0
    assert station["value_test_items"][0]["lsl"] == 17.0
    assert [item["name"] for item in station["nonvalue_test_items"]] == ["TEMPERATURE_CHECK", "TEST_RESULT"]
    assert "status" not in station["nonvalue_test_items"][0]
    assert "status" not in station["nonvalue_test_items"][1]
    assert [item["name"] for item in station["nonvalue_bin_test_items"]] == [
        "WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20",
        "WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20",
    ]
    assert "status" not in station["nonvalue_bin_test_items"][0]
    assert station["error"] is None


def test_latest_station_records_accepts_station_name_and_isn():
    record_payload = {
        "record": [
            {
                "test_date": "2025-01-02T00:00:00Z",
                "device": "DeviceB",
                "isn": "ISN_B",
            }
        ],
        "data": [
            ["WiFi_TX1_POW", 25.0, 17.0, "20.1", "21.0", "PASS"],
            ["WiFi_TX2_POW", 26.0, 16.0, "20.4", "21.0", "PASS"],
        ],
    }
    dut_records = {
        "site_name": "PTB",
        "model_name": "CALIX",
        "record_data": [
            {
                "id": 148,
                "name": "StationX",
                "dut_isn": "ISN_B",
                "dut_id": 333,
                "data": [
                    {
                        "test_date": "2025-01-02T00:00:00Z",
                        "test_result": 1,
                        "device_id": 2,
                        "device_id__name": "DeviceB",
                        "dut_id": 333,
                        "dut_id__isn": "ISN_B",
                    }
                ],
            }
        ],
    }
    mock = _MockDUTClient(
        dut_records=dut_records,
        station_records={(148, 333): record_payload},
        latest_records={(148, 333): record_payload},
        sites=[{"id": 2, "name": "PTB"}],
        models={2: [{"id": 44, "name": "CALIX", "site_id": 2}]},
        stations={44: [{"id": 148, "name": "StationX", "site_id": 2, "model_id": 44, "status": 1, "order": 10}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/records/latest/StationX/ISN_B")
    assert resp.status_code == 200, resp.json()
    payload = resp.json()
    assert len(payload["record"]) == 1
    assert payload["record"][0]["device"] == "DeviceB"
    assert len(payload["data"]) == 2


def test_station_nonvalue_records_route():
    station_payload = {
        "record": [
            {"test_date": "2025-10-11T01:41:38Z", "device": "614638", "error_item": ""},
            {"test_date": "2025-10-14T03:50:57Z", "device": "614660", "error_item": "CHECK_WiFi_6G_FW_VER"},
        ],
        "data": [
            ["WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "0x237d", "0x244c", "0x235g"],
            ["WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "0x244c", "0x244c", "0x247e"],
        ],
    }
    mock = _MockDUTClient(
        nonvalue_records={(142, 9410441): station_payload},
        dut_records={
            "site_name": "ABS",
            "model_name": "ABST",
            "record_data": [
                {
                    "id": 142,
                    "name": "BP3 Download",
                    "dut_id": 9410441,
                    "dut_isn": "260884980003907",
                    "data": [
                        {
                            "dut_id": 9410441,
                            "dut_id__isn": "260884980003907",
                            "device_id": 1669,
                            "device_id__name": "614638",
                        }
                    ],
                }
            ],
        },
        sites=[{"id": 2, "name": "ABS"}],
        models={2: [{"id": 44, "name": "ABST", "site_id": 2}]},
        stations={44: [{"id": 142, "name": "BP3 Download", "site_id": 2, "model_id": 44}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/records/nonvalue/BP3%20Download/260884980003907")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["data"] == station_payload["data"]
    assert len(data["record"]) == len(station_payload["record"])


def test_latest_station_nonvalue_records_route():
    station_payload = {
        "record": [
            {"test_date": "2025-10-15T04:05:00Z", "device": "614661", "error_item": ""},
        ],
        "data": [
            ["WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "0x247e"],
        ],
    }
    mock = _MockDUTClient(
        latest_nonvalue={(142, 9410441): station_payload},
        dut_records={
            "site_name": "ABS",
            "model_name": "ABST",
            "record_data": [
                {
                    "id": 142,
                    "name": "BP3 Download",
                    "dut_id": 9410441,
                    "dut_isn": "260884980003907",
                    "data": [
                        {
                            "dut_id": 9410441,
                            "dut_id__isn": "260884980003907",
                            "device_id": 1669,
                            "device_id__name": "614638",
                        }
                    ],
                }
            ],
        },
        sites=[{"id": 2, "name": "ABS"}],
        models={2: [{"id": 44, "name": "ABST", "site_id": 2}]},
        stations={44: [{"id": 142, "name": "BP3 Download", "site_id": 2, "model_id": 44}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/records/nonvalue/latest/BP3%20Download/260884980003907")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["data"] == [["WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", "0x247e"]]
    assert len(data["record"]) == 1
    assert data["record"][0]["device"] == "614661"


def test_station_nonvalue_bin_records_route():
    station_payload = {
        "record": [
            {"test_date": "2025-10-11T01:41:38Z", "device": "614638", "error_item": ""},
            {"test_date": "2025-10-14T03:50:57Z", "device": "614660", "error_item": "CHECK_WiFi_6G_FW_VER"},
        ],
        "data": [
            ["SET_IPLAS_INFO_WIRELESS_1", "PASS", "PASS", "PASS"],
            ["CHECK_CAL_PATHLOSS_MD5", "PASS", "PASS", "PASS"],
        ],
    }
    mock = _MockDUTClient(
        nonvalue_bin_records={(142, 9410441): station_payload},
        dut_records={
            "site_name": "ABS",
            "model_name": "ABST",
            "record_data": [
                {
                    "id": 142,
                    "name": "BP3 Download",
                    "dut_id": 9410441,
                    "dut_isn": "260884980003907",
                    "data": [
                        {
                            "dut_id": 9410441,
                            "dut_id__isn": "260884980003907",
                            "device_id": 1669,
                            "device_id__name": "614638",
                        }
                    ],
                }
            ],
        },
        sites=[{"id": 2, "name": "ABS"}],
        models={2: [{"id": 44, "name": "ABST", "site_id": 2}]},
        stations={44: [{"id": 142, "name": "BP3 Download", "site_id": 2, "model_id": 44}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/records/nonvalueBin/BP3%20Download/260884980003907")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert [row[0] for row in data["data"]] == ["SET_IPLAS_INFO_WIRELESS_1", "CHECK_CAL_PATHLOSS_MD5"]
    assert len(data["record"]) == 2


def test_latest_station_nonvalue_bin_records_route():
    station_payload = {
        "record": [
            {"test_date": "2025-10-15T04:05:00Z", "device": "614661", "error_item": ""},
        ],
        "data": [
            ["CHECK_CAL_PATHLOSS_MD5", "PASS"],
        ],
    }
    mock = _MockDUTClient(
        latest_nonvalue_bin={(142, 9410441): station_payload},
        dut_records={
            "site_name": "ABS",
            "model_name": "ABST",
            "record_data": [
                {
                    "id": 142,
                    "name": "BP3 Download",
                    "dut_id": 9410441,
                    "dut_isn": "260884980003907",
                    "data": [
                        {
                            "dut_id": 9410441,
                            "dut_id__isn": "260884980003907",
                            "device_id": 1669,
                            "device_id__name": "614638",
                        }
                    ],
                }
            ],
        },
        sites=[{"id": 2, "name": "ABS"}],
        models={2: [{"id": 44, "name": "ABST", "site_id": 2}]},
        stations={44: [{"id": 142, "name": "BP3 Download", "site_id": 2, "model_id": 44}]},
    )
    _override_client(mock)

    client = TestClient(app)
    resp = client.get("/api/dut/records/nonvalueBin/latest/BP3%20Download/260884980003907")
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0][0] == "CHECK_CAL_PATHLOSS_MD5"
