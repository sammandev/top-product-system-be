"""
Test PA Adjusted Power Integration into Top Products

This test verifies that PA adjusted power measurements are correctly
integrated into the top product filtering and scoring system.
"""

import re

import pytest

from src.app.routers.external_api_client import (
    _calculate_pa_adjusted_power_score,
    _detect_measurement_category,
)


class TestPAScoringThresholds:
    """Test updated PA adjusted power scoring thresholds."""

    def test_perfect_score_zero_deviation(self):
        """Test that deviation = 0 gives score = 10.0 (perfect)."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.0)
        assert deviation == 0.0
        assert score == 10.0

    def test_very_good_score_deviation_0_25(self):
        """Test that deviation = 0.25 gives score 9.5 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.25)
        assert deviation == 0.25
        assert score == 9.5  # Linear: 10 * (1 - 0.25/5.0) = 9.5

    def test_very_good_score_deviation_0_5(self):
        """Test that deviation = 0.5 gives score 9.0 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.5)
        assert deviation == 0.5
        assert score == 9.0  # Linear: 10 * (1 - 0.5/5.0) = 9.0

    def test_good_score_deviation_0_75(self):
        """Test that deviation = 0.75 gives score 8.5 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(0.75)
        assert deviation == 0.75
        assert score == 8.5  # Linear: 10 * (1 - 0.75/5.0) = 8.5

    def test_good_score_deviation_1_0(self):
        """Test that deviation = 1.0 gives score 8.0 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(1.0)
        assert deviation == 1.0
        assert score == 8.0  # Linear: 10 * (1 - 1.0/5.0) = 8.0

    def test_acceptable_score_deviation_1_25(self):
        """Test that deviation = 1.25 gives score 7.5 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(1.25)
        assert deviation == 1.25
        assert score == 7.5  # Linear: 10 * (1 - 1.25/5.0) = 7.5

    def test_acceptable_score_deviation_1_5(self):
        """Test that deviation = 1.5 gives score 7.0 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(1.5)
        assert deviation == 1.5
        assert score == 7.0  # Linear: 10 * (1 - 1.5/5.0) = 7.0

    def test_not_so_good_score_deviation_1_75(self):
        """Test that deviation = 1.75 gives score 6.5 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(1.75)
        assert deviation == 1.75
        assert score == 6.5  # Linear: 10 * (1 - 1.75/5.0) = 6.5

    def test_not_so_good_score_deviation_2_0(self):
        """Test that deviation = 2.0 gives score 6.0 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(2.0)
        assert deviation == 2.0
        assert score == 6.0  # Linear: 10 * (1 - 2.0/5.0) = 6.0

    def test_not_good_score_deviation_2_5(self):
        """Test that deviation = 2.5 gives score 5.0 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(2.5)
        assert deviation == 2.5
        assert score == 5.0  # Linear: 10 * (1 - 2.5/5.0) = 5.0

    def test_not_good_score_deviation_3_0(self):
        """Test that deviation = 3.0 gives score 4.0 with linear formula."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(3.0)
        assert deviation == 3.0
        assert score == 4.0  # Linear: 10 * (1 - 3.0/5.0) = 4.0

    def test_negative_values_treated_as_absolute(self):
        """Test that negative values are treated as absolute deviation."""
        deviation_pos, score_pos, _ = _calculate_pa_adjusted_power_score(0.5)
        deviation_neg, score_neg, _ = _calculate_pa_adjusted_power_score(-0.5)

        assert deviation_pos == deviation_neg == 0.5
        assert score_pos == score_neg == 9.0  # Linear: 10 * (1 - 0.5/5.0) = 9.0

    def test_large_negative_deviation(self):
        """Test that large negative values still score properly."""
        deviation, score, _ = _calculate_pa_adjusted_power_score(-1.5)
        assert deviation == 1.5
        assert score == 7.0  # Linear: 10 * (1 - 1.5/5.0) = 7.0


class TestPACategoryDetection:
    """Test PA adjusted power category detection."""

    def test_detect_pa_adjusted_power_mid(self):
        """Test detection of PA adjusted power MID test items."""
        test_item = "WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID"
        category = _detect_measurement_category(test_item)
        assert category == "PA_ADJUSTED_POWER"

    def test_detect_pa_adjusted_power_mean(self):
        """Test detection of PA adjusted power MEAN test items."""
        test_item = "WiFi_PA2_6275_11AC_VHT40_MCS9_ADJUSTED_POW_MEAN"
        category = _detect_measurement_category(test_item)
        assert category == "PA_ADJUSTED_POWER"

    def test_detect_pa_adjusted_power_case_insensitive(self):
        """Test case-insensitive detection."""
        test_item = "WiFi_PA3_2462_11G_54M_adjusted_power_mid"
        category = _detect_measurement_category(test_item)
        assert category == "PA_ADJUSTED_POWER"

    def test_not_detect_regular_pa_srom(self):
        """Test that regular PA SROM items are not detected as ADJUSTED_POWER."""
        test_item = "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80"
        category = _detect_measurement_category(test_item)
        assert category != "PA_ADJUSTED_POWER"

    def test_not_detect_regular_pa_pow(self):
        """Test that regular PA power items are not detected as ADJUSTED_POWER."""
        test_item = "WiFi_PA1_POW_5985_11AX_MCS9_B80"
        category = _detect_measurement_category(test_item)
        assert category != "PA_ADJUSTED_POWER"


class TestPAFilteringPatterns:
    """Test PA regex filtering patterns."""

    @pytest.mark.parametrize(
        "test_item,pattern,should_match",
        [
            # All PA adjusted power items
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID", r".*ADJUSTED_POW.*", True),
            ("WiFi_PA2_6275_11AC_VHT40_MCS9_ADJUSTED_POW_MEAN", r".*ADJUSTED_POW.*", True),
            # Only MID
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID", r".*ADJUSTED_POW_MID.*", True),
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MEAN", r".*ADJUSTED_POW_MID.*", False),
            # Only MEAN
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MEAN", r".*ADJUSTED_POW_MEAN.*", True),
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID", r".*ADJUSTED_POW_MEAN.*", False),
            # Specific PA number
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID", r"WiFi_PA1_.*ADJUSTED_POW.*", True),
            ("WiFi_PA2_5985_11AX_MCS9_B80_ADJUSTED_POW_MID", r"WiFi_PA1_.*ADJUSTED_POW.*", False),
            # Specific frequency
            ("WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID", r".*5985.*ADJUSTED_POW.*", True),
            ("WiFi_PA1_6275_11AC_VHT40_MCS9_ADJUSTED_POW_MID", r".*5985.*ADJUSTED_POW.*", False),
            # Not PA adjusted power
            ("WiFi_TX1_POW_5985_11AX_MCS9_B80", r".*ADJUSTED_POW.*", False),
            ("WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", r".*ADJUSTED_POW.*", False),
        ],
    )
    def test_pa_pattern_matching(self, test_item: str, pattern: str, should_match: bool):
        """Test that PA regex patterns match correctly."""
        regex = re.compile(pattern, re.IGNORECASE)
        matches = bool(regex.search(test_item))
        assert matches == should_match


class TestPAIntegrationScenarios:
    """Test complete PA integration scenarios."""

    def test_pa_measurement_row_format(self):
        """Test that PA measurements have correct format for measurement matrix."""
        # Expected format: [test_item_name, usl, lsl, actual, target, score]
        test_item_name = "WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID"
        usl = None  # PA has no USL
        lsl = None  # PA has no LSL
        actual = 0.37  # Example adjusted power value
        target = 0.0  # Target is always 0 for PA
        _, score, _ = _calculate_pa_adjusted_power_score(actual)

        measurement_row = [test_item_name, usl, lsl, actual, target, score]

        assert len(measurement_row) == 6
        assert isinstance(measurement_row[0], str)
        assert measurement_row[1] is None  # usl
        assert measurement_row[2] is None  # lsl
        assert isinstance(measurement_row[3], float)  # actual
        assert measurement_row[4] == 0.0  # target
        assert isinstance(measurement_row[5], float)  # score
        assert 0.0 <= measurement_row[5] <= 10.0

    def test_pa_score_calculation_example(self):
        """Test PA score calculation with real-world example."""
        # Example: PA SROM difference = 94 (from docs example)
        # Adjusted power = 94 / 256 = 0.37 (rounded to 2 decimals)
        actual = 0.37
        deviation, score, _ = _calculate_pa_adjusted_power_score(actual)

        assert deviation == 0.37
        # Linear formula: 10 * (1 - 0.37/5.0) = 9.26
        assert score == 9.26
        assert 9.0 < score < 10.0

    def test_pa_score_boundary_cases(self):
        """Test PA score calculation at boundary values."""
        test_cases = [
            (0.0, 10.0),  # Perfect: 10 * (1 - 0/5) = 10.0
            (0.5, 9.0),   # 10 * (1 - 0.5/5) = 9.0
            (1.0, 8.0),   # 10 * (1 - 1.0/5) = 8.0
            (1.5, 7.0),   # 10 * (1 - 1.5/5) = 7.0
            (2.0, 6.0),   # 10 * (1 - 2.0/5) = 6.0
        ]

        for actual, expected_score in test_cases:
            _, score, _ = _calculate_pa_adjusted_power_score(actual)
            assert score == expected_score, f"Failed for actual={actual}: got {score}, expected {expected_score}"

    def test_pa_naming_convention(self):
        """Test that PA test item names follow expected convention."""
        examples = [
            "WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MID",
            "WiFi_PA1_5985_11AX_MCS9_B80_ADJUSTED_POW_MEAN",
            "WiFi_PA2_6275_11AC_VHT40_MCS9_ADJUSTED_POW_MID",
            "WiFi_PA3_2462_11G_54M_ADJUSTED_POW_MEAN",
            "WiFi_PA4_5200_11N_HT40_MCS15_ADJUSTED_POW_MID",
        ]

        for test_item in examples:
            # Verify naming components
            assert "WiFi_PA" in test_item
            assert "ADJUSTED_POW" in test_item
            assert test_item.endswith("_MID") or test_item.endswith("_MEAN")

            # Verify PA number (1-4)
            pa_match = re.search(r"PA([1-4])", test_item)
            assert pa_match is not None
            pa_num = int(pa_match.group(1))
            assert 1 <= pa_num <= 4

            # Verify category detection
            category = _detect_measurement_category(test_item)
            assert category == "PA_ADJUSTED_POWER"

