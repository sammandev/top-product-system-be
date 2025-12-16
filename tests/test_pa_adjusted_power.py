"""Tests for PA adjusted power calculation."""

import pytest

from app.routers.external_api_client import _calculate_adjusted_power, _extract_pa_base_name


class TestPABaseName:
    """Test extraction of PA base names from test item names."""

    def test_extract_old_pattern(self):
        """Test extracting base name from SROM_OLD pattern."""
        result = _extract_pa_base_name("WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80")
        assert result == "WiFi_PA1_5985_11AX_MCS9_B80"

    def test_extract_new_pattern(self):
        """Test extracting base name from SROM_NEW pattern."""
        result = _extract_pa_base_name("WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80")
        assert result == "WiFi_PA1_5985_11AX_MCS9_B80"

    def test_extract_pa2_old(self):
        """Test PA2 SROM_OLD pattern."""
        result = _extract_pa_base_name("WiFi_PA2_SROM_OLD_6275_11AC_VHT40_MCS9")
        assert result == "WiFi_PA2_6275_11AC_VHT40_MCS9"

    def test_extract_pa3_new(self):
        """Test PA3 SROM_NEW pattern."""
        result = _extract_pa_base_name("WiFi_PA3_SROM_NEW_7891_11N_HT20_MCS7")
        assert result == "WiFi_PA3_7891_11N_HT20_MCS7"

    def test_extract_pa4_old(self):
        """Test PA4 SROM_OLD pattern."""
        result = _extract_pa_base_name("WiFi_PA4_SROM_OLD_8888_11B_CCK_11M")
        assert result == "WiFi_PA4_8888_11B_CCK_11M"

    def test_invalid_pattern(self):
        """Test non-PA test item returns None."""
        result = _extract_pa_base_name("Some_Other_Test_Item")
        assert result is None

    def test_empty_string(self):
        """Test empty string returns None."""
        result = _extract_pa_base_name("")
        assert result is None

    def test_none_input(self):
        """Test None input returns None."""
        result = _extract_pa_base_name(None)
        assert result is None

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        result = _extract_pa_base_name("WiFi_pa1_srom_old_5985_11AX_MCS9_B80")
        assert result == "WiFi_pa1_5985_11AX_MCS9_B80"


class TestAdjustedPowerCalculation:
    """Test adjusted power calculation logic."""

    def test_example_from_spec(self):
        """Test the exact example from the specification."""
        # From spec:
        # OLD: mid=11219.0, mean=11227
        # NEW: mid=11313.0, mean=11308
        # Expected: adjusted_mid=0.37, adjusted_mean=0.32
        result = _calculate_adjusted_power(
            old_mid=11219.0,
            old_mean=11227,
            new_mid=11313.0,
            new_mean=11308,
        )

        assert result["adjusted_mid"] == 0.37
        assert result["adjusted_mean"] == 0.32
        assert result["raw_mid_difference"] == 94.0
        assert result["raw_mean_difference"] == 81

    def test_positive_difference(self):
        """Test positive difference (NEW > OLD)."""
        result = _calculate_adjusted_power(
            old_mid=1000.0,
            old_mean=1000.0,
            new_mid=1256.0,
            new_mean=1256.0,
        )

        assert result["adjusted_mid"] == 1.0  # 256/256
        assert result["adjusted_mean"] == 1.0
        assert result["raw_mid_difference"] == 256.0
        assert result["raw_mean_difference"] == 256.0

    def test_negative_difference(self):
        """Test negative difference (OLD > NEW)."""
        result = _calculate_adjusted_power(
            old_mid=2000.0,
            old_mean=2000.0,
            new_mid=1744.0,
            new_mean=1744.0,
        )

        assert result["adjusted_mid"] == -1.0  # -256/256
        assert result["adjusted_mean"] == -1.0
        assert result["raw_mid_difference"] == -256.0
        assert result["raw_mean_difference"] == -256.0

    def test_zero_difference(self):
        """Test zero difference (NEW == OLD)."""
        result = _calculate_adjusted_power(
            old_mid=5000.0,
            old_mean=5000.0,
            new_mid=5000.0,
            new_mean=5000.0,
        )

        assert result["adjusted_mid"] == 0.0
        assert result["adjusted_mean"] == 0.0
        assert result["raw_mid_difference"] == 0.0
        assert result["raw_mean_difference"] == 0.0

    def test_fractional_rounding(self):
        """Test rounding to 2 decimal places."""
        result = _calculate_adjusted_power(
            old_mid=1000.0,
            old_mean=1000.0,
            new_mid=1001.0,  # difference = 1
            new_mean=1001.0,
        )

        # 1 / 256 = 0.00390625 -> rounds to 0.0
        assert result["adjusted_mid"] == 0.0
        assert result["adjusted_mean"] == 0.0

        result2 = _calculate_adjusted_power(
            old_mid=1000.0,
            old_mean=1000.0,
            new_mid=1100.0,  # difference = 100
            new_mean=1100.0,
        )

        # 100 / 256 = 0.390625 -> rounds to 0.39
        assert result2["adjusted_mid"] == 0.39
        assert result2["adjusted_mean"] == 0.39

    def test_missing_old_mid(self):
        """Test when OLD mid value is None."""
        result = _calculate_adjusted_power(
            old_mid=None,
            old_mean=1000.0,
            new_mid=1256.0,
            new_mean=1256.0,
        )

        assert result["adjusted_mid"] is None
        assert result["adjusted_mean"] == 1.0
        assert result["raw_mid_difference"] is None
        assert result["raw_mean_difference"] == 256.0

    def test_missing_new_mid(self):
        """Test when NEW mid value is None."""
        result = _calculate_adjusted_power(
            old_mid=1000.0,
            old_mean=1000.0,
            new_mid=None,
            new_mean=1256.0,
        )

        assert result["adjusted_mid"] is None
        assert result["adjusted_mean"] == 1.0
        assert result["raw_mid_difference"] is None

    def test_missing_all_mid_values(self):
        """Test when both mid values are None."""
        result = _calculate_adjusted_power(
            old_mid=None,
            old_mean=1000.0,
            new_mid=None,
            new_mean=1256.0,
        )

        assert result["adjusted_mid"] is None
        assert result["adjusted_mean"] == 1.0
        assert result["raw_mid_difference"] is None

    def test_missing_all_mean_values(self):
        """Test when both mean values are None."""
        result = _calculate_adjusted_power(
            old_mid=1000.0,
            old_mean=None,
            new_mid=1256.0,
            new_mean=None,
        )

        assert result["adjusted_mid"] == 1.0
        assert result["adjusted_mean"] is None
        assert result["raw_mean_difference"] is None

    def test_all_none_values(self):
        """Test when all values are None."""
        result = _calculate_adjusted_power(
            old_mid=None,
            old_mean=None,
            new_mid=None,
            new_mean=None,
        )

        assert result["adjusted_mid"] is None
        assert result["adjusted_mean"] is None
        assert result["raw_mid_difference"] is None
        assert result["raw_mean_difference"] is None

    def test_large_values(self):
        """Test with large integer values."""
        result = _calculate_adjusted_power(
            old_mid=50000.0,
            old_mean=50000.0,
            new_mid=51280.0,
            new_mean=51280.0,
        )

        # 1280 / 256 = 5.0
        assert result["adjusted_mid"] == 5.0
        assert result["adjusted_mean"] == 5.0
        assert result["raw_mid_difference"] == 1280.0

    def test_decimal_precision(self):
        """Test rounding behavior at edge cases."""
        result = _calculate_adjusted_power(
            old_mid=1000.0,
            old_mean=1000.0,
            new_mid=1128.0,  # 128 / 256 = 0.5
            new_mean=1128.0,
        )

        assert result["adjusted_mid"] == 0.5
        assert result["adjusted_mean"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
