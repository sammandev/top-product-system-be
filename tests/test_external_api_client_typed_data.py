from app.routers.external_api_client import (
    StationEvaluation,
    _build_station_result_schema,
    _process_pa_srom_enhanced,
)
from app.schemas.dut_schemas import (
    PAAdjustedPowerDataItemSchema,
    PASROMDataItemSchema,
    StationRecordResponseSchema,
)


def test_pa_srom_enhanced_pairs_typed_values() -> None:
    payload = StationRecordResponseSchema(
        data=[
            ["WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "0x0100"],
            ["WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80", "0x0200"],
        ]
    )

    items, has_pa_srom = _process_pa_srom_enhanced(payload)

    assert has_pa_srom is True
    assert sum(isinstance(item, PASROMDataItemSchema) for item in items) == 2
    adjusted_items = [item for item in items if isinstance(item, PAAdjustedPowerDataItemSchema)]
    assert len(adjusted_items) == 1
    assert adjusted_items[0].adjusted_value == 1.0


def test_station_summary_converts_legacy_rows_to_response_objects() -> None:
    evaluation = StationEvaluation(
        station_id=1,
        station_name="Wireless_Test",
        dut_isn="ISN-1",
        dut_id=2,
        device_id=3,
        device_name="Device",
        test_date=None,
        test_duration=None,
        error_item=None,
        pass_count=1,
        fail_count=0,
        retest_count=0,
        test_count=1,
        score=8.5,
        status=1,
        order=1,
        data=[["TX_POW_5985", 10.0, 0.0, 7.5]],
        measurements=[],
        site_name="PTB",
        project_name="Model",
        model_name="Model",
    )

    summary = _build_station_result_schema(evaluation)

    assert summary.data == [
        {
            "test_item": "TX_POW_5985",
            "usl": 10.0,
            "lsl": 0.0,
            "actual": 7.5,
            "score_breakdown": None,
        }
    ]
