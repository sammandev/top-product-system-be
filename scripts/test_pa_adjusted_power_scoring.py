"""
Test Script: PA Adjusted Power Linear Scoring

Verifies the updated scoring formula: score = 10 × (1 - deviation/threshold)
Default threshold: 5.0 dB
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.routers.external_api_client import _calculate_pa_adjusted_power_score


def test_pa_adjusted_power_linear_scoring():
    """Test linear scoring with default threshold of 5.0"""
    print("=" * 70)
    print("PA ADJUSTED POWER - Linear Scoring Test (threshold=5.0)")
    print("=" * 70)
    print()

    test_cases = [
        (0.0, 10.0, "Perfect - no deviation"),
        (1.0, 8.0, "Small deviation"),
        (2.5, 5.0, "Half threshold"),
        (5.0, 0.0, "At threshold"),
        (7.5, 0.0, "Beyond threshold"),
        (-3.0, 4.0, "Negative value (abs applied)"),
    ]

    all_passed = True
    for actual, expected_score, description in test_cases:
        deviation, score = _calculate_pa_adjusted_power_score(actual, threshold=5.0)

        status = "✓ PASS" if abs(score - expected_score) < 0.01 else "✗ FAIL"
        if status == "✗ FAIL":
            all_passed = False

        print(f"{status} {description}")
        print(f"      Actual: {actual}, Deviation: {deviation:.2f}")
        print(f"      Expected Score: {expected_score}, Got: {score}")
        print()

    return all_passed


def test_pa_adjusted_power_custom_threshold():
    """Test linear scoring with custom threshold"""
    print("=" * 70)
    print("PA ADJUSTED POWER - Linear Scoring Test (custom threshold=10.0)")
    print("=" * 70)
    print()

    test_cases = [
        (0.0, 10.0, "Perfect - no deviation"),
        (2.5, 7.5, "Quarter threshold"),
        (5.0, 5.0, "Half threshold"),
        (10.0, 0.0, "At threshold"),
        (15.0, 0.0, "Beyond threshold"),
    ]

    all_passed = True
    for actual, expected_score, description in test_cases:
        deviation, score = _calculate_pa_adjusted_power_score(actual, threshold=10.0)

        status = "✓ PASS" if abs(score - expected_score) < 0.01 else "✗ FAIL"
        if status == "✗ FAIL":
            all_passed = False

        print(f"{status} {description}")
        print(f"      Actual: {actual}, Deviation: {deviation:.2f}")
        print(f"      Expected Score: {expected_score}, Got: {score}")
        print()

    return all_passed


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "PA ADJUSTED POWER SCORING VERIFICATION" + " " * 16 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    result1 = test_pa_adjusted_power_linear_scoring()
    result2 = test_pa_adjusted_power_custom_threshold()

    print("=" * 70)
    if result1 and result2:
        print("✓ ALL TESTS PASSED")
        print()
        print("Scoring formula: score = 10 × (1 - deviation/threshold)")
        print("Default threshold: 5.0 dB")
        print("Custom threshold: Configurable via API parameter")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
