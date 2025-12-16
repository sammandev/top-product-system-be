"""
Test PA endpoint renaming and response structure changes.
"""

from app.schemas.dut_schemas import (
    CombinedPAAdjustedPowerSchema,
    PAAdjustedPowerTrendItemSchema,
)


class TestPAEndpointStructure:
    """Test the new PA endpoint response structures."""

    def test_pa_adjusted_power_trend_item_schema(self):
        """Test PAAdjustedPowerTrendItemSchema structure."""
        item = PAAdjustedPowerTrendItemSchema(
            test_pattern="WiFi_PA1_5985_11AX_MCS9_B80",
            adjusted_power_test_items={
                "WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80": 0.37,
                "WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80": 0.35,
            },
        )

        assert item.test_pattern == "WiFi_PA1_5985_11AX_MCS9_B80"
        assert item.adjusted_power_test_items["WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80"] == 0.37
        assert item.adjusted_power_test_items["WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80"] == 0.35

    def test_pa_adjusted_power_trend_item_with_empty_items(self):
        """Test PAAdjustedPowerTrendItemSchema with empty adjusted_power_test_items."""
        item = PAAdjustedPowerTrendItemSchema(
            test_pattern="WiFi_PA1_5985_11AX_MCS9_B80",
            adjusted_power_test_items={},
        )

        assert item.test_pattern == "WiFi_PA1_5985_11AX_MCS9_B80"
        assert len(item.adjusted_power_test_items) == 0

    def test_combined_pa_response_structure(self):
        """Test CombinedPAAdjustedPowerSchema structure."""
        from datetime import datetime

        response = CombinedPAAdjustedPowerSchema(
            dut_isn="261534750003154",
            site_name="PTB",
            model_name="HH5K",
            stations=[],
            adjusted_power_trend=[
                PAAdjustedPowerTrendItemSchema(
                    test_pattern="WiFi_PA1_5985_11AX_MCS9_B80",
                    adjusted_power_test_items={
                        "WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80": 0.37,
                        "WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80": 0.35,
                    },
                    error=None,
                )
            ],
            time_window_start=datetime(2025, 11, 23, 10, 30, 0),
            time_window_end=datetime(2025, 11, 24, 10, 30, 0),
            unpaired_items=[],
        )

        assert response.dut_isn == "261534750003154"
        assert response.site_name == "PTB"
        assert response.model_name == "HH5K"
        assert len(response.adjusted_power_trend) == 1
        assert response.adjusted_power_trend[0].test_pattern == "WiFi_PA1_5985_11AX_MCS9_B80"
        assert len(response.unpaired_items) == 0

    def test_combined_pa_response_json_serialization(self):
        """Test that the response can be serialized to JSON."""
        from datetime import datetime

        response = CombinedPAAdjustedPowerSchema(
            dut_isn="261534750003154",
            site_name="PTB",
            model_name="HH5K",
            stations=[],
            adjusted_power_trend=[
                PAAdjustedPowerTrendItemSchema(
                    test_pattern="WiFi_PA1_5985_11AX_MCS9_B80",
                    adjusted_power_test_items={
                        "WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80": 0.37,
                        "WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80": 0.35,
                    },
                    error=None,
                )
            ],
            time_window_start=datetime(2025, 11, 23, 10, 30, 0),
            time_window_end=datetime(2025, 11, 24, 10, 30, 0),
            unpaired_items=[],
        )

        # Convert to dict (simulates JSON serialization)
        response_dict = response.model_dump()

        assert "dut_isn" in response_dict
        assert "adjusted_power_trend" in response_dict
        assert isinstance(response_dict["adjusted_power_trend"], list)
        assert len(response_dict["adjusted_power_trend"]) == 1
        assert "test_pattern" in response_dict["adjusted_power_trend"][0]
        assert "adjusted_power_test_items" in response_dict["adjusted_power_trend"][0]

    def test_multiple_pa_patterns_in_trend(self):
        """Test response with multiple PA patterns."""
        response = CombinedPAAdjustedPowerSchema(
            dut_isn="261534750003154",
            site_name="PTB",
            model_name="HH5K",
            stations=[],
            adjusted_power_trend=[
                PAAdjustedPowerTrendItemSchema(
                    test_pattern="WiFi_PA1_5985_11AX_MCS9_B80",
                    adjusted_power_test_items={
                        "WiFi_PA1_ADJUSTED_POW_MID_5985_11AX_MCS9_B80": 0.37,
                        "WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80": 0.35,
                    },
                    error=None,
                ),
                PAAdjustedPowerTrendItemSchema(
                    test_pattern="WiFi_PA2_6275_11AC_VHT40_MCS9",
                    adjusted_power_test_items={
                        "WiFi_PA2_ADJUSTED_POW_MID_6275_11AC_VHT40_MCS9": 0.42,
                        "WiFi_PA2_ADJUSTED_POW_MEAN_6275_11AC_VHT40_MCS9": 0.40,
                    },
                    error=None,
                ),
            ],
            time_window_start=None,
            time_window_end=None,
            unpaired_items=[],
        )

        assert len(response.adjusted_power_trend) == 2
        assert response.adjusted_power_trend[0].test_pattern == "WiFi_PA1_5985_11AX_MCS9_B80"
        assert response.adjusted_power_trend[1].test_pattern == "WiFi_PA2_6275_11AC_VHT40_MCS9"
