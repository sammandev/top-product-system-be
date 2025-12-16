"""
Example script demonstrating PA adjusted power MID and MEAN calculation.

This script shows how the PA adjusted power enrichment works with both
MID and MEAN values, creating two separate virtual test items per PA base name.
"""

import re

from app.routers.external_api_client import (
    _calculate_adjusted_power,
    _calculate_pa_adjusted_power_score,
    _extract_pa_base_name,
)


def main():
    print("=" * 80)
    print("PA Adjusted Power MID and MEAN Example")
    print("=" * 80)
    print()

    # Example trend data from external DUT API
    trend_data = {
        "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80": {"mid": 11219.0, "mean": 11200.0},
        "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80": {"mid": 11313.0, "mean": 11290.0},
        "WiFi_PA2_SROM_OLD_6275_11AC_VHT40_MCS9": {"mid": 10500.0, "mean": 10480.0},
        "WiFi_PA2_SROM_NEW_6275_11AC_VHT40_MCS9": {"mid": 10608.0, "mean": 10585.0},
    }

    print("External DUT API Trend Data:")
    print("-" * 80)
    for item_name, values in trend_data.items():
        print(f"  {item_name}:")
        print(f"    mid:  {values['mid']}")
        print(f"    mean: {values['mean']}")
    print()

    # Step 1: Pair OLD and NEW items by base name
    print("Step 1: Pairing OLD and NEW items")
    print("-" * 80)

    paired_items = {}
    for item_name, item_data in trend_data.items():
        base_name = _extract_pa_base_name(item_name)
        if not base_name:
            continue

        if base_name not in paired_items:
            paired_items[base_name] = {"old_mid": None, "old_mean": None, "new_mid": None, "new_mean": None}

        if "_SROM_OLD_" in item_name.upper():
            paired_items[base_name]["old_mid"] = item_data.get("mid")
            paired_items[base_name]["old_mean"] = item_data.get("mean")
        elif "_SROM_NEW_" in item_name.upper():
            paired_items[base_name]["new_mid"] = item_data.get("mid")
            paired_items[base_name]["new_mean"] = item_data.get("mean")

    for base_name, values in paired_items.items():
        print(f"\n  Base Name: {base_name}")
        print(f"    OLD:  mid={values['old_mid']}, mean={values['old_mean']}")
        print(f"    NEW:  mid={values['new_mid']}, mean={values['new_mean']}")
    print()

    # Step 2: Calculate adjusted power for both MID and MEAN
    print("\nStep 2: Calculating Adjusted Power")
    print("-" * 80)

    adjusted_power_values = {}
    for base_name, values in paired_items.items():
        adjusted_power_result = _calculate_adjusted_power(
            old_mid=values["old_mid"],
            old_mean=values["old_mean"],
            new_mid=values["new_mid"],
            new_mean=values["new_mean"],
        )

        result_dict = {}
        if adjusted_power_result.get("adjusted_mid") is not None:
            result_dict["mid"] = adjusted_power_result["adjusted_mid"]

        if adjusted_power_result.get("adjusted_mean") is not None:
            result_dict["mean"] = adjusted_power_result["adjusted_mean"]

        if result_dict:
            adjusted_power_values[base_name] = result_dict

        print(f"\n  {base_name}:")
        print("    Formula: (NEW - OLD) / 256")
        if "mid" in result_dict:
            print(f"    MID:  ({values['new_mid']} - {values['old_mid']}) / 256 = {result_dict['mid']}")
        if "mean" in result_dict:
            print(f"    MEAN: ({values['new_mean']} - {values['old_mean']}) / 256 = {result_dict['mean']}")
    print()

    # Step 3: Create virtual test items and score them
    print("\nStep 3: Creating Virtual Test Items and Scoring")
    print("-" * 80)

    virtual_measurements = []
    for base_name, value_dict in adjusted_power_values.items():
        print(f"\n  Base Name: {base_name}")

        # Create MID virtual test item
        if "mid" in value_dict:
            # Replace PA{n}_ with PA{n}_ADJUSTED_POWER_MID_
            virtual_test_item_mid = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POWER_MID_", base_name, count=1, flags=re.IGNORECASE)
            deviation_mid, score_mid = _calculate_pa_adjusted_power_score(value_dict["mid"])

            measurement_mid = {
                "test_item": virtual_test_item_mid,
                "usl": None,
                "lsl": None,
                "target": 0.0,
                "actual": value_dict["mid"],
                "deviation": deviation_mid,
                "score_value": score_mid,
            }
            virtual_measurements.append(measurement_mid)

            print("\n    MID Virtual Test Item:")
            print(f"      Name:      {virtual_test_item_mid}")
            print(f"      Target:    {measurement_mid['target']}")
            print(f"      Actual:    {measurement_mid['actual']}")
            print(f"      Deviation: {measurement_mid['deviation']}")
            print(f"      Score:     {measurement_mid['score_value']}/10.0")

        # Create MEAN virtual test item
        if "mean" in value_dict:
            # Replace PA{n}_ with PA{n}_ADJUSTED_POWER_MEAN_
            virtual_test_item_mean = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POWER_MEAN_", base_name, count=1, flags=re.IGNORECASE)
            deviation_mean, score_mean = _calculate_pa_adjusted_power_score(value_dict["mean"])

            measurement_mean = {
                "test_item": virtual_test_item_mean,
                "usl": None,
                "lsl": None,
                "target": 0.0,
                "actual": value_dict["mean"],
                "deviation": deviation_mean,
                "score_value": score_mean,
            }
            virtual_measurements.append(measurement_mean)

            print("\n    MEAN Virtual Test Item:")
            print(f"      Name:      {virtual_test_item_mean}")
            print(f"      Target:    {measurement_mean['target']}")
            print(f"      Actual:    {measurement_mean['actual']}")
            print(f"      Deviation: {measurement_mean['deviation']}")
            print(f"      Score:     {measurement_mean['score_value']}/10.0")
    print()

    # Summary
    print("\nSummary:")
    print("-" * 80)
    print(f"Total PA base names processed: {len(adjusted_power_values)}")
    print(f"Total virtual measurements created: {len(virtual_measurements)}")
    print()
    print("Virtual Test Items Created:")
    for i, measurement in enumerate(virtual_measurements, 1):
        print(f"  {i}. {measurement['test_item']}")
        print(f"     Actual: {measurement['actual']}, Score: {measurement['score_value']}/10.0")
    print()
    print("=" * 80)
    print("✅ PA Adjusted Power MID and MEAN calculation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
