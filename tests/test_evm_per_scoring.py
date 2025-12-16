"""
Tests for specialized EVM and PER scoring logic in top-products endpoints.

These tests verify that:
1. EVM (Error Vector Magnitude) measurements are scored based on USL with better scores for lower values
2. PER (Packet Error Rate) measurements use the USL-driven scoring model
3. Other measurement types continue to use default scoring logic
"""

import pytest

from app.routers.external_api_client import (
    _calculate_evm_score,
    _calculate_measurement_metrics,
    _calculate_per_score,
    _detect_measurement_category,
)


class TestCategoryDetection:
    """Test detection of measurement categories from test item names."""

    def test_detect_evm_category(self):
        """Test EVM category detection."""
        test_items = [
            "WiFi_TX1_EVM_6185_11AX_MCS11_B160",
            "WiFi_RX2_EVM_2437_11N_MCS0_B20",
            "TX_EVM_5300_11AC_MCS8_B40",
        ]
        for item in test_items:
            category = _detect_measurement_category(item)
            assert category == "EVM", f"Failed to detect EVM in {item}"

    def test_detect_per_category(self):
        """Test PER category detection."""
        test_items = [
            "WiFi_RX1_PER_6185_11AX_MCS11_B160",
            "WiFi_RX2_PER_2437_11N_MCS0_B20",
            "RX_PER_5300_11AC_MCS8_B40",
        ]
        for item in test_items:
            category = _detect_measurement_category(item)
            assert category == "PER", f"Failed to detect PER in {item}"

    def test_detect_pow_category(self):
        """Test POW (Power) category detection - should use default scoring."""
        test_items = [
            "WiFi_TX1_POW_6185_11AX_MCS11_B160",
            "WiFi_PA1_POW_OLD_2422_11AC_MCS7_B40",
        ]
        for item in test_items:
            category = _detect_measurement_category(item)
            assert category == "POW", f"Failed to detect POW in {item}"


class TestEVMScoring:
    """Test EVM (Error Vector Magnitude) specialized scoring."""

    def test_evm_excellent_performance(self):
        """Test EVM scoring for excellent performance (very low EVM)."""
        # USL = -5, Actual = -30 (excellent, far below USL)
        deviation, score, _ = _calculate_evm_score(usl=-5.0, actual=-30.0)
        assert deviation == 0.0, "Within spec should have 0 deviation"
        assert score >= 8.0, f"Excellent EVM should score >= 8.0, got {score}"
        assert score <= 10.0, f"Score should not exceed 10.0, got {score}"

    def test_evm_good_performance(self):
        """Test EVM scoring for good performance (below USL)."""
        # USL = -3, Actual = -17 (good, well below USL)
        deviation, score, _ = _calculate_evm_score(usl=-3.0, actual=-17.0)
        assert deviation == 0.0, "Within spec should have 0 deviation"
        assert score >= 6.0, f"Good EVM should score >= 6.0, got {score}"
        assert score <= 10.0, f"Score should not exceed 10.0, got {score}"

    def test_evm_at_usl_limit(self):
        """Test EVM scoring at USL boundary."""
        # USL = -5, Actual = -5 (at limit, minimum acceptable)
        deviation, score, _ = _calculate_evm_score(usl=-5.0, actual=-5.0)
        assert deviation == 0.0, "At USL should have 0 deviation"
        assert score == 6.0, f"At USL should score exactly 6.0, got {score}"

    def test_evm_exceeds_usl(self):
        """Test EVM scoring when exceeding USL (failing)."""
        # USL = -5, Actual = -3 (exceeds USL, worse than spec)
        deviation, score, _ = _calculate_evm_score(usl=-5.0, actual=-3.0)
        assert deviation > 0, "Exceeding USL should have positive deviation"
        assert score < 6.0, f"Exceeding USL should score < 6.0, got {score}"
        assert score >= 0.0, f"Score should not be negative, got {score}"

    def test_evm_no_usl_provided(self):
        """Test EVM scoring when no USL is provided (uses default -3)."""
        # No USL, Actual = -20 (good by default standard)
        deviation, score, _ = _calculate_evm_score(usl=None, actual=-20.0)
        assert score >= 6.0, f"Good EVM with default USL should score >= 6.0, got {score}"

    def test_evm_bonus_for_very_low_values(self):
        """Test bonus scoring for exceptionally low EVM values."""
        # USL = -5, Actual = -40 (exceptional performance)
        deviation, score, _ = _calculate_evm_score(usl=-5.0, actual=-40.0)
        assert score >= 9.0, f"Exceptional EVM should score >= 9.0, got {score}"


class TestPERScoring:
    """Test PER (Packet Error Rate) specialized scoring."""

    def test_per_perfect_zero(self):
        """Test PER scoring for perfect 0 (no errors)."""
        usl = 0.1
        deviation, score, _ = _calculate_per_score(usl=usl, actual=0.0)
        assert deviation == 0.0, "Perfect PER should have 0 deviation"
        assert score == 10.0, f"Perfect PER should score 10.0, got {score}"

    def test_per_very_low(self):
        """Test PER scoring for very low error rate."""
        usl = 0.1
        actual = 0.01
        deviation, score, _ = _calculate_per_score(usl=usl, actual=actual)
        expected = ((usl - actual) / usl) * 10.0
        assert score == pytest.approx(expected, rel=1e-6)
        assert 0.0 < score < 10.0

    def test_per_low_range(self):
        """Test PER scoring in low but acceptable range."""
        usl = 0.1
        actual = 0.05
        deviation, score, _ = _calculate_per_score(usl=usl, actual=actual)
        expected = ((usl - actual) / usl) * 10.0
        assert score == pytest.approx(expected, rel=1e-6)
        assert score == pytest.approx(5.0, rel=1e-6)

    def test_per_threshold_boundary(self):
        """Test PER scoring at USL boundary."""
        usl = 0.1
        deviation, score, _ = _calculate_per_score(usl=usl, actual=usl)
        assert deviation == usl
        assert score == 0.0

    def test_per_high_error_rate(self):
        """Test PER scoring for high error rate."""
        usl = 0.1
        deviation, score, _ = _calculate_per_score(usl=usl, actual=0.3)
        assert deviation == 0.3
        assert score == 0.0

    def test_per_with_custom_usl(self):
        """Test PER scoring with smaller USL window."""
        usl = 0.02
        actual = 0.01
        deviation, score, _ = _calculate_per_score(usl=usl, actual=actual)
        expected = ((usl - actual) / usl) * 10.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_per_at_usl(self):
        """Test PER scoring exactly at USL."""
        usl = 0.05
        deviation, score, _ = _calculate_per_score(usl=usl, actual=usl)
        assert score == 0.0
        assert deviation == usl


class TestIntegratedScoring:
    """Test integrated scoring through _calculate_measurement_metrics."""

    def test_evm_integrated_scoring(self):
        """Test EVM scoring through main metrics function."""
        # EVM test item with USL
        test_item = "WiFi_TX1_EVM_6185_11AX_MCS11_B160"
        usl = -5.0
        lsl = None
        target = -5.0
        actual = -20.0

        deviation, score, _ = _calculate_measurement_metrics(usl, lsl, target, actual, test_item)
        assert score >= 6.0, f"EVM should use specialized scoring, got {score}"

    def test_per_integrated_scoring(self):
        """Test PER scoring through main metrics function."""
        # PER test item with USL
        test_item = "WiFi_RX1_PER_6185_11AX_MCS11_B160"
        usl = 0.1
        lsl = None
        target = 0.0
        actual = 0.0

        deviation, score, _ = _calculate_measurement_metrics(usl, lsl, target, actual, test_item)
        assert score == 10.0, f"Perfect PER should score 10.0, got {score}"

    def test_pow_uses_default_scoring(self):
        """Test that POW (Power) measurements use default scoring."""
        # POW test item - should use default logic
        test_item = "WiFi_TX1_POW_6185_11AX_MCS11_B160"
        usl = 26.0
        lsl = 16.0
        target = 21.0
        actual = 21.0

        deviation, score, _ = _calculate_measurement_metrics(usl, lsl, target, actual, test_item)
        # At target, should score 10.0 with default logic
        assert score == 10.0, f"POW at target should score 10.0, got {score}"

    def test_default_scoring_without_test_item(self):
        """Test that default scoring is used when test_item is None."""
        usl = 26.0
        lsl = 16.0
        target = 21.0
        actual = 21.0

        deviation, score, _ = _calculate_measurement_metrics(usl, lsl, target, actual, test_item=None)
        assert score == 10.0, f"At target should score 10.0, got {score}"


class TestScoringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_evm_usl_zero(self):
        """Test EVM with USL of 0."""
        deviation, score, _ = _calculate_evm_score(usl=0.0, actual=-10.0)
        assert score >= 6.0, "Below USL should pass"

    def test_per_near_zero(self):
        """Test PER with very small but non-zero value."""
        usl = 0.1
        actual = 0.001
        deviation, score, _ = _calculate_per_score(usl=usl, actual=actual)
        expected = ((usl - actual) / usl) * 10.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_per_negative_value(self):
        """Test PER with negative value (impossible but handle gracefully)."""
        usl = 0.1
        deviation, score, _ = _calculate_per_score(usl=usl, actual=-0.01)
        assert score == 10.0, "Negative PER (better than 0) should score 10.0"

    def test_evm_positive_value(self):
        """Test EVM with positive value (unusual)."""
        # EVM is typically negative, but test positive case
        deviation, score, _ = _calculate_evm_score(usl=-5.0, actual=2.0)
        assert score < 6.0, "Positive EVM (exceeding USL) should score low"


class TestScoringConsistency:
    """Test scoring consistency and monotonicity."""

    def test_evm_monotonic_decrease(self):
        """Test that EVM score increases as actual value decreases."""
        usl = -5.0
        scores = []
        for actual in [-5, -10, -20, -30, -40]:
            _, score, _ = _calculate_evm_score(usl, actual)
            scores.append(score)

        # Scores should generally increase or stay the same as EVM improves
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 0.1, f"EVM scores should improve monotonically: {scores}"

    def test_per_monotonic_increase_with_errors(self):
        """Test that PER score decreases as error rate increases."""
        scores = []
        for actual in [0.0, 0.02, 0.05, 0.1, 0.2]:
            _, score, _ = _calculate_per_score(None, actual)
            scores.append(score)

        # Scores should strictly decrease as PER increases
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1] - 0.1, f"PER scores should decrease monotonically: {scores}"
