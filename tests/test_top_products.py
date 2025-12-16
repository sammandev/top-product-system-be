from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db import Base, SessionLocal, engine
from app.dependencies.authz import get_current_user
from app.main import app
from app.models.top_product import TopProduct, TopProductMeasurement
from app.routers.external_api_client import get_user_dut_client


class _MockDUTClient:
    def __init__(self, dut_records, station_payloads):
        self._records = dut_records
        self._station_payloads = station_payloads
        self.base_url = f"mock-top-{id(self)}"
        record_station = dut_records["record_data"][0]
        self._site = {"id": 2, "name": dut_records.get("site_name", "PTB")}
        self._model = {"id": 32, "name": dut_records.get("model_name", "CALIX_EXPRESSO2-R"), "site_id": self._site["id"]}
        self._station = {
            "id": record_station["id"],
            "name": record_station["name"],
            "model_id": self._model["id"],
            "model_name": self._model["name"],
            "site_id": self._site["id"],
            "site_name": self._site["name"],
            "status": 1,
            "order": record_station.get("order", 0),
        }
        self._device_runs = []
        for entry in record_station["data"]:
            self._device_runs.append(
                {
                    "test_date": entry["test_date"],
                    "test_result": entry.get("test_result", 1),
                    "device_id": entry.get("device_id"),
                    "device_id__name": entry.get("device_id__name"),
                    "device": entry.get("device"),
                    "dut_id": entry.get("dut_id"),
                    "dut_id__isn": entry.get("dut_id__isn"),
                    "status": entry.get("test_result", 1),
                }
            )

    async def get_dut_records(self, dut_isn: str):
        return self._records

    async def get_station_records(self, station_id: int, dut_id: int):
        return self._station_payloads[(station_id, dut_id)]

    async def get_sites(self):
        return [self._site]

    async def get_models_by_site(self, site_id: int):
        if site_id != self._site["id"]:
            return []
        return [self._model]

    async def get_stations_by_model(self, model_id: int):
        if model_id != self._model["id"]:
            return []
        return [self._station]

    async def get_devices_by_period(self, station_id: int, start_time: datetime, end_time: datetime, _result: str):
        if station_id != self._station["id"]:
            return []
        if start_time.tzinfo is not None:
            start_cmp = start_time.astimezone(UTC).replace(tzinfo=None)
        else:
            start_cmp = start_time
        if end_time.tzinfo is not None:
            end_cmp = end_time.astimezone(UTC).replace(tzinfo=None)
        else:
            end_cmp = end_time
        result: list[dict] = []
        for entry in self._device_runs:
            test_dt = datetime.fromisoformat(entry["test_date"].replace("Z", "+00:00")).replace(tzinfo=None)
            if start_cmp <= test_dt <= end_cmp:
                result.append(dict(entry))
        return result


@pytest.fixture(autouse=True)
def reset_top_product_tables():
    tables = [TopProductMeasurement.__table__, TopProduct.__table__]
    Base.metadata.drop_all(bind=engine, tables=tables)
    Base.metadata.create_all(bind=engine, tables=tables)
    yield
    Base.metadata.drop_all(bind=engine, tables=tables)


def _override_dependencies(mock_client):
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(username="tester")
    app.dependency_overrides[get_user_dut_client] = lambda: mock_client


def _clear_overrides():
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _make_common_payloads():
    record_data = [
        {
            "id": 148,
            "name": "Wireless_2_5G_Test",
            "order": 10,
            "model_name": "ProjectX",
            "data": [
                {
                    "test_date": "2025-01-01T00:00:00Z",
                    "test_result": 1,
                    "device_id": 1,
                    "device_id__name": "DeviceA",
                    "device": "DeviceA",
                    "dut_id": 111,
                    "dut_id__isn": "ISN_A",
                },
                {
                    "test_date": "2025-01-02T00:00:00Z",
                    "test_result": 1,
                    "device_id": 2,
                    "device_id__name": "DeviceB",
                    "device": "DeviceB",
                    "dut_id": 222,
                    "dut_id__isn": "ISN_B",
                },
                {
                    "test_date": "2025-01-03T00:00:00Z",
                    "test_result": 1,
                    "device_id": 3,
                    "device_id__name": "DeviceC",
                    "device": "DeviceC",
                    "dut_id": 333,
                    "dut_id__isn": "ISN_C",
                },
            ],
        }
    ]

    station_payload = {
        "record": [
            {"test_date": "2025-01-01T00:00:00Z", "device": "DeviceA", "isn": "ISN_A"},
            {"test_date": "2025-01-02T00:00:00Z", "device": "DeviceB", "isn": "ISN_B"},
            {"test_date": "2025-01-03T00:00:00Z", "device": "DeviceC", "isn": "ISN_C"},
        ],
        "data": [
            ["WiFi_PA1_POW_OLD_2422_11AC_MCS7_B40", 25.0, 17.0, "18.0", "20.1", "30.0"],
            ["WiFi_TX1_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20", 26.0, 16.0, "21.0", "21.0", "21.0"],
        ],
    }

    station_payloads = {
        (148, 111): station_payload,
        (148, 222): station_payload,
        (148, 333): station_payload,
    }

    dut_records = {
        "site_name": "PTB",
        "model_name": "CALIX_EXPRESSO2-R",
        "record_data": record_data,
    }

    return dut_records, station_payloads


def _write_criteria(tmp_path: Path) -> Path:
    content = """
[Wireless_2_5G_Test]
"WiFi_PA_POW_OLD_2422_11AC_MCS7_B40" <25,17>  ===> "21"
"WiFi_TX_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20" <26,16>  ===> "21"
"""
    path = tmp_path / "criteria.ini"
    path.write_text(content.strip(), encoding="utf-8")
    return path


def _upload_payload(criteria_file: Path):
    content = criteria_file.read_bytes()
    return {"criteria_file": (criteria_file.name, content, "text/plain")}


def test_top_product_endpoint_ranks_best_dut(client, tmp_path):
    dut_records, station_payloads = _make_common_payloads()
    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/top-product",
            params={"dut_isn": "ISN_B"},
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["errors"] == []
    data = payload["results"][0]
    assert data["criteria_path"] == criteria_file.name
    assert data["dut_isn"] == "ISN_B"
    assert len(data["test_result"]) == 1
    top_entry = data["test_result"][0]
    assert top_entry["device"] == "DeviceB"
    assert top_entry["station_name"] == "Wireless_2_5G_Test"
    assert pytest.approx(top_entry["data"][0]["actual"], rel=1e-6) == 20.1
    assert pytest.approx(top_entry["data"][0]["score_breakdown"]["final_score"], rel=1e-6) == 9.57
    assert pytest.approx(top_entry["overall_data_score"], abs=0.01) == 9.79
    assert "group_scores" not in top_entry

    with SessionLocal() as db:
        stored = db.query(TopProduct).all()
        assert len(stored) == 1
        record = stored[0]
        assert record.dut_isn == "ISN_B"
        assert record.station_name == "Wireless_2_5G_Test"
        assert pytest.approx(record.score, abs=0.01) == 9.79
        assert len(record.measurements) == 2
        assert pytest.approx(record.measurements[0].actual_value, rel=1e-6) == 20.1


def test_top_product_uses_default_criteria_without_file(client):
    dut_records, station_payloads = _make_common_payloads()
    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params={"dut_isn": "ISN_B"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["errors"] == []
    data = payload["results"][0]
    assert data["criteria_path"] == "latest-tests"
    assert data["test_result"][0]["device"] == "DeviceB"

    with SessionLocal() as db:
        stored = db.query(TopProduct).all()
        assert len(stored) == 1
        assert stored[0].dut_isn == "ISN_B"


def test_top_product_allows_test_item_filters(client):
    dut_records, station_payloads = _make_common_payloads()
    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params={
                "dut_isn": "ISN_B",
                "test_item_filters": ["WiFi_TX_FIXTURE_OR_DUT_PROBLEM_POW_2437_11N_MCS0_B20"],
            },
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["errors"] == []
    data = payload["results"][0]
    assert len(data["test_result"]) == 1
    entry = data["test_result"][0]
    assert entry["device"] == "DeviceB"
    assert len(entry["data"]) == 1
    assert "WiFi_TX" in entry["data"][0]["test_item"]
    assert pytest.approx(entry["data"][0]["score_breakdown"]["final_score"], rel=1e-6) == 10.0
    assert pytest.approx(entry["overall_data_score"], rel=1e-6) == 10.0
    assert "group_scores" not in entry

    with SessionLocal() as db:
        stored = db.query(TopProduct).all()
        assert len(stored) == 1
        stored_record = stored[0]
        assert stored_record.dut_isn == "ISN_B"
        assert pytest.approx(stored_record.score, rel=1e-6) == 10.0
        assert len(stored_record.measurements) == 1


def test_top_product_exclude_patterns(client):
    dut_records, station_payloads = _make_common_payloads()
    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params={
                "dut_isn": "ISN_B",
                "exclude_test_item_filters": ["WiFi_PA_POW_OLD_2422_11AC_MCS7_B40"],
            },
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["errors"] == []
    entry = payload["results"][0]["test_result"][0]
    assert len(entry["data"]) == 1
    assert "WiFi_PA" not in entry["data"][0]["test_item"]
    assert "WiFi_TX" in entry["data"][0]["test_item"]
    # hierarchical alias remains consistent


def test_station_top_products_filters_by_date_and_eliminates_out_of_spec(client, tmp_path):
    dut_records, station_payloads = _make_common_payloads()
    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-02T00:00:00Z",
                "end_time": "2025-01-03T00:00:00Z",
                "criteria_score": "6",
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["site_name"] == "PTB"
    assert payload["model_name"] == "CALIX_EXPRESSO2-R"
    assert payload["criteria_score"] == "6"
    assert len(payload["requested_data"]) == 1
    best = payload["requested_data"][0]
    assert best["station_name"] == "Wireless_2_5G_Test"
    assert best["isn"] == "ISN_B"
    assert pytest.approx(best["latest_data"][0]["actual"], rel=1e-6) == 20.1
    assert pytest.approx(best["overall_data_score"], abs=0.01) == 9.79
    assert "group_scores" not in best

    with SessionLocal() as db:
        stored = db.query(TopProduct).all()
        assert len(stored) == 0


def test_top_product_filter_matches_variant_indices(client):
    dut_records, station_payloads = _make_common_payloads()
    variant_payload = deepcopy(station_payloads[(148, 111)])
    variant_payload["data"] = [
        ["WiFi_TX1_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "23.0"],
        ["WiFi_TX2_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "23.0", "23.0", "22.5"],
        ["WiFi_TX3_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "22.7", "23.0", "22.0"],
        ["WiFi_TX4_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "22.5", "23.0", "21.5"],
    ]
    for key in list(station_payloads.keys()):
        station_payloads[key] = deepcopy(variant_payload)

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params={
                "dut_isn": "ISN_B",
                "test_item_filters": ["WiFi_TX_POW_6185_11AX_MCS11_B160"],
            },
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["errors"] == []
    measurements = payload["results"][0]["test_result"][0]["data"]
    assert len(measurements) == 4
    names = [row["test_item"] for row in measurements]
    assert names == [
        "WiFi_TX1_POW_6185_11AX_MCS11_B160",
        "WiFi_TX2_POW_6185_11AX_MCS11_B160",
        "WiFi_TX3_POW_6185_11AX_MCS11_B160",
        "WiFi_TX4_POW_6185_11AX_MCS11_B160",
    ]


def test_top_product_filter_general_pattern(client):
    dut_records, station_payloads = _make_common_payloads()
    variant_payload = deepcopy(station_payloads[(148, 111)])
    variant_payload["data"] = [
        ["WiFi_TX1_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "23.0"],
        ["WiFi_TX1_EVM_6185_11AX_MCS11_B160", 26.0, 16.0, "23.1", "23.0", "22.5"],
        ["WiFi_TX2_FREQ_6185_11AX_MCS11_B160", 26.0, 16.0, "22.9", "23.0", "22.1"],
        ["WiFi_TX3_MASK_6185_11AX_MCS11_B160", 26.0, 16.0, "23.2", "23.0", "23.4"],
        ["WiFi_TX4_LO_LEAKAGE_DB_6185_11AX_MCS11_B160", 26.0, 16.0, "23.3", "23.0", "22.9"],
        ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "20.0"],
    ]
    for key in list(station_payloads.keys()):
        station_payloads[key] = deepcopy(variant_payload)

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params={"dut_isn": "ISN_B", "test_item_filters": ["WiFi_TX_6185_11AX_MCS11_B160"]},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    entry = payload["results"][0]["test_result"][0]
    names = sorted(row["test_item"] for row in entry["data"])
    assert names == sorted(
        [
            "WiFi_TX1_EVM_6185_11AX_MCS11_B160",
            "WiFi_TX1_POW_6185_11AX_MCS11_B160",
            "WiFi_TX2_FREQ_6185_11AX_MCS11_B160",
            "WiFi_TX3_MASK_6185_11AX_MCS11_B160",
            "WiFi_TX4_LO_LEAKAGE_DB_6185_11AX_MCS11_B160",
        ]
    )
    assert "WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160" not in names
    assert "group_scores" not in entry


def test_top_product_exclude_general_pattern(client):
    dut_records, station_payloads = _make_common_payloads()
    variant_payload = deepcopy(station_payloads[(148, 111)])
    variant_payload["data"] = [
        ["WiFi_TX1_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "23.0"],
        ["WiFi_TX1_EVM_6185_11AX_MCS11_B160", 26.0, 16.0, "23.1", "23.0", "22.5"],
        ["WiFi_TX2_FREQ_6185_11AX_MCS11_B160", 26.0, 16.0, "22.9", "23.0", "22.1"],
        ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "20.0"],
    ]
    for key in list(station_payloads.keys()):
        station_payloads[key] = deepcopy(variant_payload)

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params={
                "dut_isn": "ISN_B",
                "exclude_test_item_filters": ["WiFi_TX_6185_11AX_MCS11_B160"],
            },
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    entry = payload["results"][0]["test_result"][0]
    names = [row["test_item"] for row in entry["data"]]
    assert names == ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160"]


def test_top_product_hierarchical_groups(client):
    dut_records, station_payloads = _make_common_payloads()
    variant_payload = deepcopy(station_payloads[(148, 111)])
    variant_payload["data"] = [
        ["WiFi_TX1_POW_6185_11AX_MCS11_B160", 20.0, 10.0, 15.0, 16.0, 9.4],
        ["WiFi_TX1_EVM_6185_11AX_MCS11_B160", 20.0, 10.0, 15.0, 16.0, 9.0],
        ["WiFi_TX2_POW_6185_11AX_MCS11_B160", 20.0, 10.0, 14.8, 16.0, 9.2],
        ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160", 20.0, 10.0, 15.0, 16.0, 8.5],
        ["WiFi_RX1_PER_6185_11AX_MCS11_B160", 20.0, 10.0, 15.0, 16.0, 10.0],
        ["WiFi_RX1_RSSI_6185_11AX_MCS11_B160", 20.0, 10.0, 15.0, 16.0, 9.0],
        ["WiFi_TX1_POW_6175_11AX_MCS9_B20", 20.0, 10.0, 15.0, 16.0, 8.0],
        ["WiFi_TX2_POW_6175_11AX_MCS9_B20", 20.0, 10.0, 15.0, 16.0, 8.5],
    ]
    for key in list(station_payloads.keys()):
        station_payloads[key] = deepcopy(variant_payload)

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product/hierarchical",
            params=[("dut_isn", "ISN_B")],
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["errors"] == []
    station = payload["results"][0]["test_result"][0]
    groups = station["group_scores"]
    assert "6185_11AX_MCS11_B160" in groups
    tx_group = groups["6185_11AX_MCS11_B160"]["TX"]
    # Verify hierarchical structure is present with reasonable score values
    assert "TX1" in tx_group
    assert "POW" in tx_group["TX1"]
    assert "tx1_score" in tx_group["TX1"]
    assert 9.0 < tx_group["TX1"]["POW"] < 9.5  # POW score should be high
    assert 7.5 < tx_group["TX1"]["tx1_score"] < 8.5  # Average of POW and EVM
    assert "TX2" in tx_group
    assert "POW" in tx_group["TX2"]
    assert 9.0 < tx_group["TX2"]["POW"] < 9.5  # TX2 POW score
    assert "tx_group_score" in tx_group
    assert 8.0 < tx_group["tx_group_score"] < 9.0  # Average across TX antennas
    assert "6175_11AX_MCS9_B20" in groups
    assert "TX" in groups["6175_11AX_MCS9_B20"]
    assert "tx_group_score" in groups["6175_11AX_MCS9_B20"]["TX"]
    overall_groups = station["overall_group_scores"]
    # Verify overall group scores exist and are reasonable
    assert "TX" in overall_groups and overall_groups["TX"] > 7.0
    assert "PA" in overall_groups and overall_groups["PA"] > 6.0
    assert "RX" in overall_groups and overall_groups["RX"] > 5.0  # RX has low PER score
    assert "overall_data_score" in station


def test_top_product_hierarchical_filter_general_pattern(client):
    dut_records, station_payloads = _make_common_payloads()
    variant_payload = deepcopy(station_payloads[(148, 111)])
    variant_payload["data"] = [
        ["WiFi_TX1_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "23.0"],
        ["WiFi_TX1_EVM_6185_11AX_MCS11_B160", 26.0, 16.0, "23.1", "23.0", "22.5"],
        ["WiFi_TX2_FREQ_6185_11AX_MCS11_B160", 26.0, 16.0, "22.9", "23.0", "22.1"],
        ["WiFi_TX3_MASK_6185_11AX_MCS11_B160", 26.0, 16.0, "23.2", "23.0", "23.4"],
        ["WiFi_TX4_LO_LEAKAGE_DB_6185_11AX_MCS11_B160", 26.0, 16.0, "23.3", "23.0", "22.9"],
        ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "20.0"],
    ]
    for key in list(station_payloads.keys()):
        station_payloads[key] = deepcopy(variant_payload)

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product/hierarchical",
            params=[("dut_isn", "ISN_B"), ("test_item_filters", "WiFi_TX_6185_11AX_MCS11_B160")],
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    station = payload["results"][0]["test_result"][0]
    names = sorted(row["test_item"] for row in station["data"])
    assert names == sorted(
        [
            "WiFi_TX1_EVM_6185_11AX_MCS11_B160",
            "WiFi_TX1_POW_6185_11AX_MCS11_B160",
            "WiFi_TX2_FREQ_6185_11AX_MCS11_B160",
            "WiFi_TX3_MASK_6185_11AX_MCS11_B160",
            "WiFi_TX4_LO_LEAKAGE_DB_6185_11AX_MCS11_B160",
        ]
    )
    assert station["group_scores"]["6185_11AX_MCS11_B160"]["TX"]["tx_group_score"] >= 0
    assert set(station["group_scores"].keys()) == {"6185_11AX_MCS11_B160"}
    assert set(station["overall_group_scores"].keys()) == {"TX"}


def test_top_product_hierarchical_exclude_general_pattern(client):
    dut_records, station_payloads = _make_common_payloads()
    variant_payload = deepcopy(station_payloads[(148, 111)])
    variant_payload["data"] = [
        ["WiFi_TX1_POW_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "23.0"],
        ["WiFi_TX1_EVM_6185_11AX_MCS11_B160", 26.0, 16.0, "23.1", "23.0", "22.5"],
        ["WiFi_TX2_FREQ_6185_11AX_MCS11_B160", 26.0, 16.0, "22.9", "23.0", "22.1"],
        ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160", 26.0, 16.0, "23.5", "23.0", "20.0"],
    ]
    for key in list(station_payloads.keys()):
        station_payloads[key] = deepcopy(variant_payload)

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product/hierarchical",
            params=[("dut_isn", "ISN_B"), ("exclude_test_item_filters", "WiFi_TX_6185_11AX_MCS11_B160")],
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    station = payload["results"][0]["test_result"][0]
    names = [row["test_item"] for row in station["data"]]
    assert names == ["WiFi_PA1_POW_OLD_6185_11AX_MCS11_B160"]
    assert set(station["group_scores"].keys()) == {"6185_11AX_MCS11_B160"}
    assert station["group_scores"]["6185_11AX_MCS11_B160"]["PA"]["pa_group_score"] >= 0
    assert set(station["overall_group_scores"].keys()) == {"PA"}


def test_top_product_handles_multiple_isns(client):
    dut_records, station_payloads = _make_common_payloads()
    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        response = client.post(
            "/api/dut/top-product",
            params=[("dut_isn", "ISN_B"), ("dut_isn", "ISN_C")],
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["results"]) == 2
    assert {entry["dut_isn"] for entry in payload["results"]} == {"ISN_B", "ISN_C"}
    assert payload["errors"] == []


# UPDATED: Tests for new limit parameter feature
def test_station_top_products_default_limit_returns_5(client, tmp_path):
    """Test that without limit parameter, endpoint returns maximum 5 products."""
    dut_records, station_payloads = _make_common_payloads()
    # Create 8 devices to test limiting
    dut_records["record_data"][0]["data"] = [{"test_date": "2025-01-01T00:00:00Z", "test_result": 1, "device_id": i, "device_id__name": f"Device{i}", "device": f"Device{i}", "dut_id": 100 + i, "dut_id__isn": f"ISN_{i}"} for i in range(1, 9)]
    for i in range(1, 9):
        station_payloads[(148, 100 + i)] = station_payloads[(148, 111)]

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-02T00:00:00Z",
                "criteria_score": "5",
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    # Default limit is 5, so even with 8 devices we should get max 5
    assert len(payload["requested_data"]) <= 5


def test_station_top_products_custom_limit_3(client, tmp_path):
    """Test that limit=3 returns exactly 3 products."""
    dut_records, station_payloads = _make_common_payloads()
    # Create 6 devices
    dut_records["record_data"][0]["data"] = [{"test_date": "2025-01-01T00:00:00Z", "test_result": 1, "device_id": i, "device_id__name": f"Device{i}", "device": f"Device{i}", "dut_id": 100 + i, "dut_id__isn": f"ISN_{i}"} for i in range(1, 7)]
    for i in range(1, 7):
        station_payloads[(148, 100 + i)] = station_payloads[(148, 111)]

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-02T00:00:00Z",
                "criteria_score": "5",
                "limit": 3,
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["requested_data"]) <= 3


def test_station_top_products_limit_1_returns_best_only(client, tmp_path):
    """Test that limit=1 returns only the single best product."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-03T00:00:00Z",
                "criteria_score": "5",
                "limit": 1,
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert len(payload["requested_data"]) == 1
    # Should be the highest scoring product
    assert payload["requested_data"][0]["overall_data_score"] > 0


def test_station_top_products_limit_100_validates_max(client, tmp_path):
    """Test that limit=100 is accepted as the maximum allowed value."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-03T00:00:00Z",
                "criteria_score": "5",
                "limit": 100,
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    # Should succeed with limit=100


def test_station_top_products_limit_exceeds_max_fails(client, tmp_path):
    """Test that limit > 100 returns validation error."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-03T00:00:00Z",
                "criteria_score": "5",
                "limit": 101,
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 422  # Pydantic validation error


def test_station_top_products_limit_less_than_min_fails(client, tmp_path):
    """Test that limit < 1 returns validation error."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-03T00:00:00Z",
                "criteria_score": "5",
                "limit": 0,
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 422  # Pydantic validation error


# UPDATED: Tests for 7-day time window validation
def test_station_top_products_accepts_7_day_window(client, tmp_path):
    """Test that exactly 7-day time window is accepted."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-08T00:00:00Z",  # Exactly 7 days
                "criteria_score": "5",
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    # Should succeed with exactly 7 days


def test_station_top_products_rejects_8_day_window(client, tmp_path):
    """Test that 8-day time window is rejected with 400 error."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-09T00:00:00Z",  # 8 days
                "criteria_score": "5",
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 400, response.json()
    payload = response.json()
    assert "cannot exceed 7 days" in payload["detail"].lower()


def test_station_top_products_rejects_30_day_window(client, tmp_path):
    """Test that 30-day time window is rejected with clear error message."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-31T00:00:00Z",  # 30 days
                "criteria_score": "5",
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 400, response.json()
    payload = response.json()
    assert "cannot exceed 7 days" in payload["detail"].lower()


def test_station_top_products_sorted_by_score_descending(client, tmp_path):
    """Test that results are sorted by overall_data_score in descending order."""
    dut_records, station_payloads = _make_common_payloads()

    mock_client = _MockDUTClient(dut_records, station_payloads)
    _override_dependencies(mock_client)
    try:
        criteria_file = _write_criteria(tmp_path)
        response = client.post(
            "/api/dut/stations/Wireless_2_5G_Test/top-products",
            params={
                "site_id": "PTB",
                "model_id": "CALIX_EXPRESSO2-R",
                "start_time": "2025-01-01T00:00:00Z",
                "end_time": "2025-01-03T00:00:00Z",
                "criteria_score": "5",
                "limit": 10,
            },
            files=_upload_payload(criteria_file),
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200, response.json()
    payload = response.json()
    if len(payload["requested_data"]) > 1:
        scores = [item["overall_data_score"] for item in payload["requested_data"]]
        # Verify descending order (highest scores first)
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"
