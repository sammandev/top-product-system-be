import json

from fastapi.testclient import TestClient

from app.main import app
from app.routers.test_log import _apply_universal_scoring_to_parse_result, _build_scoring_config_map


client = TestClient(app)


def test_test_log_rescore_honors_disabled_and_min_score_filters_aggregate() -> None:
    response = client.post(
        '/api/test-log/rescore',
        json={
            'test_items': [
                {
                    'test_item': 'ITEM_HIGH',
                    'value': '5',
                    'usl': 10,
                    'lsl': 0,
                    'status': 'PASS',
                },
                {
                    'test_item': 'ITEM_LOW',
                    'value': '0',
                    'usl': 10,
                    'lsl': 0,
                    'status': 'PASS',
                },
                {
                    'test_item': 'ITEM_DISABLED',
                    'value': '5',
                    'usl': 10,
                    'lsl': 0,
                    'status': 'PASS',
                },
            ],
            'scoring_configs': [
                {
                    'test_item_name': 'ITEM_LOW',
                    'scoring_type': 'symmetrical',
                    'enabled': True,
                    'weight': 1,
                    'min_score': 0.8,
                },
                {
                    'test_item_name': 'ITEM_DISABLED',
                    'scoring_type': 'symmetrical',
                    'enabled': False,
                    'weight': 1,
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    score_map = {item['test_item']: item for item in payload['test_item_scores']}

    assert score_map['ITEM_HIGH']['score'] == 10.0
    assert score_map['ITEM_LOW']['score'] == 1.0
    assert score_map['ITEM_DISABLED']['score'] is None
    assert payload['overall_score'] == 10.0


def test_test_log_rescore_preserves_non_numeric_non_criteria_value() -> None:
    response = client.post(
        '/api/test-log/rescore',
        json={
            'test_items': [
                {
                    'test_item': 'SERIAL_NUMBER',
                    'value': 'QS2627174001924',
                    'usl': None,
                    'lsl': None,
                    'status': 'PASS',
                }
            ],
            'scoring_configs': [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    [item_score] = payload['test_item_scores']
    assert item_score['test_item'] == 'SERIAL_NUMBER'
    assert item_score['value'] == 'QS2627174001924'
    assert item_score['score'] is None
    assert payload['overall_score'] == 0.0


def test_test_log_parse_honors_explicit_include_scope() -> None:
    result = {
        'parsed_items_enhanced': [
            {
                'test_item': 'ITEM_SELECTED',
                'value': '5',
                'usl': 10,
                'lsl': 0,
                'is_value_type': True,
                'numeric_value': 5,
            },
            {
                'test_item': 'ITEM_NOT_SELECTED',
                'value': '5',
                'usl': 10,
                'lsl': 0,
                'is_value_type': True,
                'numeric_value': 5,
            },
        ],
    }
    config_map = _build_scoring_config_map(
        json.dumps(
            [
                {
                    'test_item_name': 'ITEM_SELECTED',
                    'scoring_type': 'symmetrical',
                    'enabled': True,
                    'weight': 1,
                },
                {
                    'test_item_name': 'ITEM_NOT_SELECTED',
                    'scoring_type': 'symmetrical',
                    'enabled': False,
                    'weight': 1,
                },
            ],
        ),
    )

    _apply_universal_scoring_to_parse_result(result, config_map)

    selected_item, not_selected_item = result['parsed_items_enhanced']
    assert selected_item['score'] == 10.0
    assert not_selected_item['score'] is None
    assert result['avg_score'] == 10.0
    assert result['median_score'] == 10.0


def test_extract_station_from_filename_patterns() -> None:
    from app.services.test_log_parser import _extract_station_from_filename

    # Pattern 1: [Station]_[ISN]_[Date]
    assert _extract_station_from_filename("Wireless_Test_6G_DM2524470075517_2025_11_20_105056100.txt") == "Wireless_Test_6G"
    assert _extract_station_from_filename("2G_CAL_ISN1234567890_2025_11_20_105056100.txt") == "2G_CAL"
    assert _extract_station_from_filename("PT1_ISN1234567890_2025_11_20_105056100.txt") == "PT1"

    # Pattern 3: [ISN]_[Station]_[Date]
    assert _extract_station_from_filename("DM2524470075517_Wireless_Test_6G_2025_11_20_105056100.txt") == "Wireless_Test_6G"

    # Pattern 2: [ISN]_[Date] - no station
    assert _extract_station_from_filename("DM2527470012971_2025_11_20_172659650.txt") == "Unknown_Station"


def test_test_log_scoring_special_types() -> None:
    from app.services.scoring_service import score_test_item

    # b1) RX PER and TX MASK: 10 if 0 or close to 0
    per_item = {'NAME': 'WiFi_RX_PER', 'STATUS': 'PASS', 'VALUE': '0', 'UCL': '10', 'LCL': '0'}
    res_per = score_test_item(per_item)
    assert res_per.scoring_type.value == 'per_mask'
    assert res_per.score == 1.0  # normalized 0-1 (i.e. 10.0 on 0-10 scale)

    mask_item = {'NAME': 'TX_MASK_ERR', 'STATUS': 'PASS', 'VALUE': '0.0', 'UCL': '5', 'LCL': ''}
    res_mask = score_test_item(mask_item)
    assert res_mask.scoring_type.value == 'per_mask'
    assert res_mask.score == 1.0

    # b2) TX EVM: linear, good (~9.0) if value close to -40
    evm_item_good = {'NAME': 'WiFi_TX_EVM', 'STATUS': 'PASS', 'VALUE': '-40', 'UCL': '-25', 'LCL': '-50'}
    res_evm = score_test_item(evm_item_good)
    assert res_evm.scoring_type.value == 'evm'
    evm_score_0_10 = res_evm.score * 10.0
    assert 8.5 <= evm_score_0_10 <= 9.2

    # b3) Throughput: 10 if value approaches or exceeds UCL
    tput_item = {'NAME': 'ETH_THROUGHPUT', 'STATUS': 'PASS', 'VALUE': '1000', 'UCL': '1000', 'LCL': '500'}
    res_tput = score_test_item(tput_item)
    assert res_tput.scoring_type.value == 'throughput'
    assert res_tput.score == 1.0  # 10.0 on 0-10 scale


def test_zero_limit_is_treated_as_criteria() -> None:
    from app.routers.test_log import _has_meaningful_limit, _should_score_upload_item

    assert _has_meaningful_limit(0.0) is True
    assert _has_meaningful_limit(0) is True
    assert _has_meaningful_limit(None) is False
    assert _should_score_upload_item('PER_ITEM', 0.0, None, {}) is True
    assert _should_score_upload_item('NO_LIMIT_ITEM', None, None, {}) is False
