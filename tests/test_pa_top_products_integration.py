"""
Integration tests for PA adjusted power in top-products endpoints.

Tests verify that PA SROM test items are detected, adjusted power is calculated,
and scores are properly included in top-products evaluation.
"""

import pytest

from app.routers.external_api_client import (
    MeasurementRow,
    _calculate_pa_adjusted_power_score,
    _calculate_pa_pow_dif_abs_score,
    _detect_measurement_category,
    _extract_pa_test_items_from_measurements,
    _is_pa_srom_test_item,
)


class TestPADetection:
    """Test PA SROM test item detection."""

    def test_is_pa_srom_test_item_old(self):
        """Should detect PA SROM OLD test items."""
        assert _is_pa_srom_test_item("WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80") is True
        assert _is_pa_srom_test_item("WiFi_PA2_SROM_OLD_6275_11AC_VHT40_MCS9") is True

    def test_is_pa_srom_test_item_new(self):
        """Should detect PA SROM NEW test items."""
        assert _is_pa_srom_test_item("WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80") is True
        assert _is_pa_srom_test_item("WiFi_PA3_SROM_NEW_2412_11B_CCK11_B20") is True

    def test_is_pa_srom_test_item_non_pa(self):
        """Should not detect non-PA test items."""
        assert _is_pa_srom_test_item("WiFi_TX1_POW_2412_11B_CCK11_B20") is False
        assert _is_pa_srom_test_item("WiFi_RX_SENSITIVITY_2412") is False
        assert _is_pa_srom_test_item("WiFi_EVM_5300_11AC_MCS8_B20") is False

    def test_is_pa_srom_test_item_case_insensitive(self):
        """Should work case-insensitively."""
        assert _is_pa_srom_test_item("wifi_pa1_srom_old_5985_11ax_mcs9_b80") is True
        assert _is_pa_srom_test_item("WIFI_PA2_SROM_NEW_6275_11AC_VHT40_MCS9") is True


class TestPAMeasurementExtraction:
    """Test extracting PA test items from measurements."""

    def test_extract_pa_test_items_with_pairs(self):
        """Should extract PA test item pairs."""
        measurements = [
            MeasurementRow(name="WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", usl=None, lsl=None, latest=11219.0),
            MeasurementRow(name="WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80", usl=None, lsl=None, latest=11313.0),
            MeasurementRow(name="WiFi_TX1_POW_2412_11B_CCK11_B20", usl=23.0, lsl=15.0, latest=18.5),
        ]

        pa_items, base_mapping = _extract_pa_test_items_from_measurements(measurements)

        assert len(pa_items) == 2
        assert "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80" in pa_items
        assert "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80" in pa_items
        assert "WiFi_PA1_5985_11AX_MCS9_B80" in base_mapping

    def test_extract_pa_test_items_no_pa(self):
        """Should return empty when no PA items present."""
        measurements = [
            MeasurementRow(name="WiFi_TX1_POW_2412_11B_CCK11_B20", usl=23.0, lsl=15.0, latest=18.5),
            MeasurementRow(name="WiFi_RX_SENSITIVITY_2412", usl=-65.0, lsl=-90.0, latest=-75.0),
        ]

        pa_items, base_mapping = _extract_pa_test_items_from_measurements(measurements)

        assert len(pa_items) == 0
        assert len(base_mapping) == 0

    def test_extract_multiple_pa_pairs(self):
        """Should handle multiple PA test item pairs."""
        measurements = [
            MeasurementRow(name="WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", usl=None, lsl=None, latest=11219.0),
            MeasurementRow(name="WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80", usl=None, lsl=None, latest=11313.0),
            MeasurementRow(name="WiFi_PA2_SROM_OLD_6275_11AC_VHT40_MCS9", usl=None, lsl=None, latest=10500.0),
            MeasurementRow(name="WiFi_PA2_SROM_NEW_6275_11AC_VHT40_MCS9", usl=None, lsl=None, latest=10608.0),
        ]

        pa_items, base_mapping = _extract_pa_test_items_from_measurements(measurements)

        assert len(pa_items) == 4
        assert "WiFi_PA1_5985_11AX_MCS9_B80" in base_mapping
        assert "WiFi_PA2_6275_11AC_VHT40_MCS9" in base_mapping


class TestPAScoring:
    """Test PA adjusted power scoring logic."""

    def test_perfect_score_zero_value(self):
        """Perfect adjusted power (0) should score 10.0."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.0)
        assert deviation == 0.0
        assert score == 10.0

    def test_excellent_score_small_deviation(self):
        """Small deviations should score high with linear formula."""
        deviation1, score1, _ = _calculate_pa_adjusted_power_score(0.1)
        # Linear: 10 * (1 - 0.1/5.0) = 10 * 0.98 = 9.8
        assert 9.7 <= score1 <= 9.9
        assert score1 < 10.0

        deviation2, score2, _ = _calculate_pa_adjusted_power_score(0.2)
        # Linear: 10 * (1 - 0.2/5.0) = 10 * 0.96 = 9.6
        assert 9.5 <= score2 <= 9.7
        assert score2 < score1  # Higher deviation = lower score

    def test_good_score_medium_deviation(self):
        """Medium deviations should score with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.4)
        # Linear: 10 * (1 - 0.4/5.0) = 10 * 0.92 = 9.2
        assert 9.1 <= score <= 9.3

    def test_acceptable_score_larger_deviation(self):
        """Larger deviations should score with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.7)
        # Linear: 10 * (1 - 0.7/5.0) = 10 * 0.86 = 8.6
        assert 8.5 <= score <= 8.7

    def test_poor_score_high_deviation(self):
        """High deviations should score lower with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(1.8)
        # Linear: 10 * (1 - 1.8/5.0) = 10 * 0.64 = 6.4
        assert 6.3 <= score <= 6.5

    def test_critical_score_extreme_deviation(self):
        """Extreme deviations should score lower with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(3.0)
        # Linear: 10 * (1 - 3.0/5.0) = 10 * 0.4 = 4.0
        assert 3.9 <= score <= 4.1

    def test_negative_values_treated_as_absolute(self):
        """Negative adjusted power should be scored by absolute value."""
        dev_pos, score_pos, _ = _calculate_pa_adjusted_power_score(0.5)
        dev_neg, score_neg, _ = _calculate_pa_adjusted_power_score(-0.5)

        assert dev_pos == dev_neg == 0.5
        assert score_pos == score_neg

    def test_example_from_spec(self):
        """Test with example from spec: (11313 - 11219) / 256 = 0.37."""
        adjusted_power = (11313.0 - 11219.0) / 256
        assert adjusted_power == pytest.approx(0.37, abs=0.01)

        deviation, score, _ = _calculate_pa_adjusted_power_score(adjusted_power)
        # Linear: 10 * (1 - 0.37/5.0) = 10 * 0.926 = 9.26
        assert 9.2 <= score <= 9.3


class TestPACategoryDetection:
    """Test measurement category detection for PA adjusted power."""

    def test_detect_pa_adjusted_power_mid_category(self):
        """Should detect PA_ADJUSTED_POWER category from MID virtual test item names."""
        # Virtual test item format: WiFi_PA{n}_ADJUSTED_POWER_MID_<rest>
        assert _detect_measurement_category("WiFi_PA1_ADJUSTED_POWER_MID_5985_11AX_MCS9_B80") == "PA_ADJUSTED_POWER"
        assert _detect_measurement_category("WiFi_PA2_ADJUSTED_POWER_MID_6275_11AC_VHT40_MCS9") == "PA_ADJUSTED_POWER"

    def test_detect_pa_adjusted_power_mean_category(self):
        """Should detect PA_ADJUSTED_POWER category from MEAN virtual test item names."""
        # Virtual test item format: WiFi_PA{n}_ADJUSTED_POWER_MEAN_<rest>
        assert _detect_measurement_category("WiFi_PA1_ADJUSTED_POWER_MEAN_5985_11AX_MCS9_B80") == "PA_ADJUSTED_POWER"
        assert _detect_measurement_category("WiFi_PA2_ADJUSTED_POWER_MEAN_6275_11AC_VHT40_MCS9") == "PA_ADJUSTED_POWER"

    def test_detect_pa_pow_dif_abs_category(self):
        """Should detect PA_POW_DIF_ABS category from POW_DIF_ABS test item names."""
        # Format: WiFi_PA{n}_POW_DIF_ABS_<rest>
        assert _detect_measurement_category("WiFi_PA1_POW_DIF_ABS_5985_11AX_MCS9_B80") == "PA_POW_DIF_ABS"
        assert _detect_measurement_category("WiFi_PA2_POW_DIF_ABS_6275_11AC_VHT40_MCS9") == "PA_POW_DIF_ABS"
        assert _detect_measurement_category("WiFi_PA3_POW_DIF_ABS_2412_11B_CCK11_B20") == "PA_POW_DIF_ABS"

    def test_not_detect_regular_pa_items(self):
        """Should not detect regular PA SROM items as PA_ADJUSTED_POWER."""
        # These should be detected as other categories or None
        result1 = _detect_measurement_category("WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80")
        result2 = _detect_measurement_category("WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80")

        # Should not be PA_ADJUSTED_POWER (might be None or POW)
        assert result1 != "PA_ADJUSTED_POWER"
        assert result2 != "PA_ADJUSTED_POWER"

    def test_detect_other_categories(self):
        """Should still detect other measurement categories correctly."""
        assert _detect_measurement_category("WiFi_TX1_EVM_5300_11AC_MCS8_B20") == "EVM"
        assert _detect_measurement_category("WiFi_TX1_PER_2412_11B_CCK11_B20") == "PER"
        assert _detect_measurement_category("WiFi_TX1_FREQ_2462") == "FREQ"
        assert _detect_measurement_category("WiFi_TX1_POW_6185_11AX_MCS11_B160") == "POW"


class TestPAPowDifAbsScoring:
    """Test PA POW_DIF_ABS linear scoring logic."""

    def test_perfect_score_zero_difference(self):
        """Perfect match (0 difference) should score 10.0."""
        deviation, score, _ = _calculate_pa_pow_dif_abs_score(0.0)
        assert deviation == 0.0
        assert score == 10.0

    def test_linear_scoring_with_default_threshold(self):
        """Should use linear interpolation with default threshold of 5.0."""
        # Default threshold = 5.0
        # Score = 10 * (1 - deviation / 5.0)

        # 10% of threshold
        dev1, score1, _ = _calculate_pa_pow_dif_abs_score(0.5)
        assert dev1 == 0.5
        assert score1 == 9.0  # 10 * (1 - 0.5/5.0) = 9.0

        # 20% of threshold
        dev2, score2, _ = _calculate_pa_pow_dif_abs_score(1.0)
        assert dev2 == 1.0
        assert score2 == 8.0  # 10 * (1 - 1.0/5.0) = 8.0

        # 50% of threshold
        dev3, score3, _ = _calculate_pa_pow_dif_abs_score(2.5)
        assert dev3 == 2.5
        assert score3 == 5.0  # 10 * (1 - 2.5/5.0) = 5.0

    def test_at_threshold_scores_zero(self):
        """At threshold should score 0.0."""
        deviation, score, _ = _calculate_pa_pow_dif_abs_score(5.0)
        assert deviation == 5.0
        assert score == 0.0

    def test_exceeds_threshold_scores_zero(self):
        """Exceeding threshold should score 0.0."""
        deviation, score, _ = _calculate_pa_pow_dif_abs_score(6.0)
        assert deviation == 6.0
        assert score == 0.0

    def test_linear_scoring_with_custom_usl(self):
        """Should use custom USL for linear interpolation."""
        # Custom threshold = 1.5
        # Score = 10 * (1 - deviation / 1.5)

        dev1, score1, _ = _calculate_pa_pow_dif_abs_score(0.5, usl=1.5)
        assert score1 == pytest.approx(6.67, abs=0.01)  # 10 * (1 - 0.5/1.5)

        dev2, score2, _ = _calculate_pa_pow_dif_abs_score(1.0, usl=1.5)
        assert score2 == pytest.approx(3.33, abs=0.01)  # 10 * (1 - 1.0/1.5)

        dev3, score3, _ = _calculate_pa_pow_dif_abs_score(1.5, usl=1.5)
        assert score3 == 0.0  # At threshold

    def test_negative_values_treated_as_absolute(self):
        """Negative values should be scored by absolute value."""
        dev_pos, score_pos, _ = _calculate_pa_pow_dif_abs_score(0.8)
        dev_neg, score_neg, _ = _calculate_pa_pow_dif_abs_score(-0.8)

        assert dev_pos == dev_neg == 0.8
        assert score_pos == score_neg

    def test_small_differences_high_score(self):
        """Small differences should score very high."""
        deviation, score, _ = _calculate_pa_pow_dif_abs_score(0.1)
        assert score == 9.8  # 10 * (1 - 0.1/5.0) = 9.8
        assert score > 9.0

    def test_score_decreases_linearly(self):
        """Score should decrease linearly with increasing deviation."""
        _, score1, _ = _calculate_pa_pow_dif_abs_score(0.2)
        _, score2, _ = _calculate_pa_pow_dif_abs_score(0.4)
        _, score3, _ = _calculate_pa_pow_dif_abs_score(0.6)
        _, score4, _ = _calculate_pa_pow_dif_abs_score(0.8)

        # Verify linear decrease
        assert score1 > score2 > score3 > score4

        # Verify equal intervals (linear)
        diff1 = score1 - score2
        diff2 = score2 - score3
        diff3 = score3 - score4
        assert diff1 == pytest.approx(diff2, abs=0.01)
        assert diff2 == pytest.approx(diff3, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
