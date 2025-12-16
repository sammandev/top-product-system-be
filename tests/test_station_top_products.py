from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from app.routers.external_api_client import (
    _calculate_measurement_metrics,
    _determine_target_value,
    _parse_score_criteria,
    _validate_time_window,
)


def test_parse_score_criteria_single_value() -> None:
    lower, upper = _parse_score_criteria("8.5")
    assert lower == pytest.approx(8.5)
    assert upper is None


def test_parse_score_criteria_range() -> None:
    lower, upper = _parse_score_criteria("8-9.5")
    assert lower == pytest.approx(8.0)
    assert upper == pytest.approx(9.5)


@pytest.mark.parametrize("raw", ["", "abc", "9-5", "8--9"])
def test_parse_score_criteria_invalid(raw: str) -> None:
    with pytest.raises(HTTPException):
        _parse_score_criteria(raw)


def test_validate_time_window_accepts_seven_days() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=7)
    _validate_time_window(start, end)


@pytest.mark.parametrize(
    "end_offset",
    [
        timedelta(seconds=0),
        timedelta(days=8),
    ],
)
def test_validate_time_window_rejects_invalid_ranges(end_offset: timedelta) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + end_offset
    with pytest.raises(HTTPException):
        _validate_time_window(start, end)


def test_score_prefers_values_above_lsl_when_no_usl():
    target = _determine_target_value(None, None, 10.0, 12.0)
    deviation, score, _ = _calculate_measurement_metrics(None, 10.0, target, 12.0)
    assert deviation == 0.0
    assert score == 10.0
    _, low_score, _ = _calculate_measurement_metrics(None, 10.0, target, 8.0)
    assert low_score < 10.0


def test_score_prefers_values_below_usl_when_no_lsl():
    target = _determine_target_value(None, 20.0, None, 18.0)
    deviation, score, _ = _calculate_measurement_metrics(20.0, None, target, 18.0)
    assert deviation == 0.0
    assert score == 10.0
    _, high_score, _ = _calculate_measurement_metrics(20.0, None, target, 25.0)
    assert high_score < 10.0


def test_target_zero_when_lower_bound_zero():
    target = _determine_target_value(None, 5.0, 0.0, 2.0)
    assert target == 0.0
    deviation, score, _ = _calculate_measurement_metrics(5.0, 0.0, target, 1.0)
    assert deviation == 1.0
    assert score > 0.0
    _, distant_score, _ = _calculate_measurement_metrics(5.0, 0.0, target, 5.0)
    assert distant_score < score
