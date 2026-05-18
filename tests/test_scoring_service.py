from app.schemas.scoring_schemas import ScoringConfig, ScoringType
from app.services.scoring_service import score_record, score_test_item


def make_value_item(value: str, *, name: str = 'TX_EVM_SAMPLE', status: str = 'PASS') -> dict:
    return {
        'NAME': name,
        'STATUS': status,
        'VALUE': value,
        'UCL': '10',
        'LCL': '0',
    }


def test_score_test_item_sets_exceeds_max_deviation_flag() -> None:
    config = ScoringConfig(
        test_item_name='TX_EVM_SAMPLE',
        scoring_type=ScoringType.SYMMETRICAL,
        max_deviation=3.0,
    )

    result = score_test_item(make_value_item('9'), config)

    assert result.deviation == 4.0
    assert result.max_deviation == 3.0
    assert result.exceeds_max_deviation is True


def test_score_test_item_keeps_exceeds_max_deviation_false_within_limit() -> None:
    config = ScoringConfig(
        test_item_name='TX_EVM_SAMPLE',
        scoring_type=ScoringType.SYMMETRICAL,
        max_deviation=3.0,
    )

    result = score_test_item(make_value_item('7.5'), config)

    assert result.deviation == 2.5
    assert result.exceeds_max_deviation is False


def test_score_record_counts_deviation_fail_as_failed_item() -> None:
    config = ScoringConfig(
        test_item_name='TX_EVM_SAMPLE',
        scoring_type=ScoringType.SYMMETRICAL,
        max_deviation=2.0,
    )
    record = {
        'ISN': 'ISN-001',
        'DeviceId': 'DEVICE-001',
        'station': 'STATION-A',
        'Test Start Time': '2026-04-27 08:00:00',
        'Test Status': 'PASS',
        'TestItem': [make_value_item('9')],
    }

    result = score_record(record, {'TX_EVM_SAMPLE': config})

    assert result.failed_items == 1
    assert result.test_item_scores[0].exceeds_max_deviation is True


def test_score_test_item_zero_only_limit_falls_back_to_binary_fail() -> None:
    config = ScoringConfig(
        test_item_name='ZERO_LIMIT_ITEM',
        scoring_type=ScoringType.SYMMETRICAL,
    )

    result = score_test_item({
        'NAME': 'ZERO_LIMIT_ITEM',
        'STATUS': 'FAIL',
        'VALUE': '0',
        'UCL': '0',
        'LCL': '',
    }, config)

    assert result.scoring_type == ScoringType.BINARY
    assert result.score == 0.0


def test_score_test_item_with_real_limit_and_zero_boundary_still_scores() -> None:
    config = ScoringConfig(
        test_item_name='ONE_REAL_LIMIT',
        scoring_type=ScoringType.SYMMETRICAL,
    )

    result = score_test_item({
        'NAME': 'ONE_REAL_LIMIT',
        'STATUS': 'PASS',
        'VALUE': '5',
        'UCL': '10',
        'LCL': '0',
    }, config)

    assert result.scoring_type == ScoringType.SYMMETRICAL
    assert result.score == 1.0
