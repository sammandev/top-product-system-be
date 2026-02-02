"""
Scoring Service

Business logic for calculating test item scores using various algorithms.
"""

import logging
import math
from statistics import mean, median, stdev
from typing import Optional

from ..schemas.scoring_schemas import (
    CalculateScoresRequest,
    CalculateScoresResponse,
    RecordScoreResult,
    ScoreSummary,
    ScoringConfig,
    ScoringType,
    SCORING_TYPE_DEFAULTS,
    TestItemScoreResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _parse_float(value: str | float | None) -> Optional[float]:
    """Safely parse a float value."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _has_control_limits(test_item: dict) -> bool:
    """Check if test item has UCL or LCL control limits."""
    ucl_str = str(test_item.get("UCL", "")).strip()
    lcl_str = str(test_item.get("LCL", "")).strip()
    has_ucl = False
    has_lcl = False
    if ucl_str:
        try:
            float(ucl_str)
            has_ucl = True
        except (ValueError, TypeError):
            pass
    if lcl_str:
        try:
            float(lcl_str)
            has_lcl = True
        except (ValueError, TypeError):
            pass
    return has_ucl or has_lcl


def _is_value_item(test_item: dict) -> bool:
    """
    Check if test item is a value item (CRITERIA or NON-CRITERIA).

    A test item is considered a value item if:
    - It has UCL or LCL control limits (CRITERIA), OR
    - It has a numeric VALUE that is not PASS/FAIL/1/0/-1/-999 (NON-CRITERIA)
    """
    # If it has control limits, it's a value item (CRITERIA)
    if _has_control_limits(test_item):
        return True

    # Otherwise, check if VALUE is a non-binary numeric value (NON-CRITERIA)
    value = str(test_item.get("VALUE", "")).upper().strip()
    if value in ("PASS", "FAIL", "1", "0", "-1", "-999", ""):
        return False
    return _parse_float(test_item.get("VALUE")) is not None


def _is_bin_item(test_item: dict) -> bool:
    """
    Check if test item is binary (PASS/FAIL/1/0/-1/-999 without control limits).

    Note: If an item has UCL/LCL, it's treated as CRITERIA even if value is 1/0/-999.
    """
    # If it has control limits, it's NOT a binary item
    if _has_control_limits(test_item):
        return False

    value = str(test_item.get("VALUE", "")).upper().strip()
    return value in ("PASS", "FAIL", "1", "0", "-1", "-999")


def _get_config_param(config: ScoringConfig | None, param: str, scoring_type: ScoringType) -> float | None:
    """Get parameter from config or use default."""
    if config and getattr(config, param, None) is not None:
        return getattr(config, param)
    defaults = SCORING_TYPE_DEFAULTS.get(scoring_type, {})
    return defaults.get(param)


# ============================================================================
# Auto-Detection
# ============================================================================


def detect_scoring_type_by_name(item_name: str) -> ScoringType | None:
    """
    Detect scoring type based on test item NAME patterns.

    Returns:
        ScoringType if a name-based pattern is matched, None otherwise.
    """
    upper_name = item_name.upper()

    # EVM scoring for test items containing "EVM" in their name
    if "EVM" in upper_name:
        return ScoringType.EVM

    # PER/MASK scoring for test items containing "PER" or "MASK" in their name
    if "PER" in upper_name or "MASK" in upper_name:
        return ScoringType.PER_MASK

    return None  # No name-based detection, use value-based detection


def get_default_weight_by_name(item_name: str) -> float:
    """
    Get default weight based on test item NAME patterns.

    Returns:
        Weight value (default 1.0 if no pattern matches).
    """
    upper_name = item_name.upper()

    # Weight 3 for test items containing "POW_OLD"
    if "POW_OLD" in upper_name:
        return 3.0

    # Weight 2 for test items containing "FIXTURE_OR_DUT_PROBLEM_POW"
    if "FIXTURE_OR_DUT_PROBLEM_POW" in upper_name:
        return 2.0

    return 1.0  # Default weight


def detect_scoring_type(test_item: dict) -> ScoringType:
    """
    Auto-detect the appropriate scoring type based on test item characteristics.

    Rules:
    1. NAME-based detection (takes precedence):
       - "EVM" in name -> EVM
       - "PER" or "MASK" in name -> PER_MASK
    2. Binary (PASS/FAIL value) -> BINARY
    3. Only UCL, typically negative values -> EVM
    4. Only LCL (lower limit only) -> THROUGHPUT
    5. UCL and LCL, symmetric -> SYMMETRICAL
    6. Only UCL, values near 0 -> PER_MASK
    """
    # UPDATED: First check name-based detection (takes precedence)
    item_name = test_item.get("NAME", "")
    if item_name:
        name_based_type = detect_scoring_type_by_name(item_name)
        if name_based_type is not None:
            return name_based_type

    if not _is_value_item(test_item):
        return ScoringType.BINARY

    ucl = _parse_float(test_item.get("UCL"))
    lcl = _parse_float(test_item.get("LCL"))
    value = _parse_float(test_item.get("VALUE"))

    # Only UCL defined (no LCL)
    if ucl is not None and lcl is None:
        # Check if values are typically negative (EVM-like)
        if value is not None and value < 0:
            return ScoringType.EVM
        # Check if values are near 0 (PER/MASK-like)
        if value is not None and abs(value) < ucl * 0.5:
            return ScoringType.PER_MASK
        return ScoringType.EVM  # Default for UCL-only

    # Only LCL defined (throughput-like)
    if ucl is None and lcl is not None:
        return ScoringType.THROUGHPUT

    # Both limits defined
    if ucl is not None and lcl is not None:
        return ScoringType.SYMMETRICAL

    # No limits - treat as binary
    return ScoringType.BINARY


# ============================================================================
# Scoring Algorithms
# ============================================================================


def score_symmetrical(value: float, ucl: float, lcl: float, alpha: float = 0.8) -> tuple[float, float]:
    """
    Calculate symmetrical score with centered target.

    Args:
        value: Measured value
        ucl: Upper Control Limit
        lcl: Lower Control Limit
        alpha: Minimum score at limit boundary (default: 0.8)

    Returns:
        Tuple of (score, deviation)
    """
    target = (ucl + lcl) / 2
    limit = (ucl - lcl) / 2

    if limit <= 0:
        return 1.0, 0.0

    deviation = abs(value - target)

    if deviation >= limit:
        return alpha, deviation

    score = alpha + (1 - alpha) * (limit - deviation) / limit
    return min(1.0, max(0.0, score)), deviation


def score_symmetrical_nl(value: float, ucl: float, lcl: float, target_score: float = 0.8, target_deviation: float = 2.5) -> tuple[float, float]:
    """
    Calculate non-linear (Gaussian) symmetrical score.

    Args:
        value: Measured value
        ucl: Upper Control Limit
        lcl: Lower Control Limit
        target_score: Score at target_deviation (default: 0.8)
        target_deviation: Deviation where score = target_score (default: 2.5)

    Returns:
        Tuple of (score, deviation)
    """
    target = (ucl + lcl) / 2
    deviation = abs(value - target)

    # Calculate lambda for Gaussian decay
    # score = exp(-λ × dev²), solve for λ: λ = -ln(target_score) / target_dev²
    if target_deviation <= 0 or target_score <= 0:
        return 1.0, deviation

    lmbda = -math.log(target_score + 1e-9) / (target_deviation**2)
    score = math.exp(-lmbda * (deviation**2))

    return min(1.0, max(0.0, score)), deviation


def score_evm(value: float, target_evm: float = -30.0, target_score: float = 0.9) -> tuple[float, float]:
    """
    Calculate score for EVM-type test items (lower/more negative is better).

    Args:
        value: Measured EVM value (negative dB)
        target_evm: Target EVM value for target_score (default: -30.0)
        target_score: Score at target_evm (default: 0.9)

    Returns:
        Tuple of (score, deviation_from_0)
    """
    # EVM: more negative is better, 0 is worst
    # score = 1 - exp(-λ × measured²)
    if target_score >= 1 or target_evm >= 0:
        return 0.5, abs(value)

    lmbda = -math.log(1 - target_score + 1e-9) / (target_evm**2)
    score = 1 - math.exp(-lmbda * (value**2))

    return min(1.0, max(0.0, score)), abs(value)


def score_throughput(value: float, lcl: float, target: float | None = None, min_score: float = 0.4, target_score: float = 0.9) -> tuple[float, float]:
    """
    Calculate score for throughput-type test items (higher is better).

    Args:
        value: Measured throughput value
        lcl: Lower Control Limit (minimum acceptable)
        target: Target throughput (if None, uses 1.5 × LCL)
        min_score: Score at LCL (default: 0.4)
        target_score: Score at target (default: 0.9)

    Returns:
        Tuple of (score, deviation_from_target)
    """
    if target is None or target <= lcl:
        target = lcl * 1.5 if lcl > 0 else lcl + 100

    if value < lcl:
        return 0.0, target - value

    if value < target:
        # Linear region: LCL to target
        m = (target_score - min_score) / (target - lcl)
        score = m * value + (min_score - m * lcl)
    else:
        # Exponential region: above target
        lmbda = -math.log(1 - target_score + 1e-9) / (target**2)
        score = 1 - math.exp(-lmbda * (value**2))

    return min(1.0, max(0.0, score)), abs(target - value)


def score_asymmetrical(value: float, ucl: float, lcl: float, target: float, alpha: float = 0.4) -> tuple[float, float]:
    """
    Calculate asymmetrical score with user-defined target.

    Args:
        value: Measured value
        ucl: Upper Control Limit
        lcl: Lower Control Limit
        target: User-defined target (between LCL and UCL)
        alpha: Minimum score at limit boundary (default: 0.4)

    Returns:
        Tuple of (score, deviation)
    """
    if value >= target:
        limit = ucl - target
        deviation = value - target
    else:
        limit = target - lcl
        deviation = target - value

    if limit <= 0:
        return 1.0, 0.0

    if deviation >= limit:
        return alpha, deviation

    score = alpha + (1 - alpha) * (limit - deviation) / limit
    return min(1.0, max(0.0, score)), deviation


def score_per_mask(value: float, target: float = 0.0, max_deviation: float = 10.0) -> tuple[float, float]:
    """
    Calculate score for PER/MASK test items (0 is best, linear decay).

    Args:
        value: Measured value
        target: Target value (default: 0)
        max_deviation: Maximum acceptable deviation (default: 10)

    Returns:
        Tuple of (score, deviation)
    """
    deviation = abs(value - target)

    if max_deviation <= 0:
        return 1.0 if deviation == 0 else 0.0, deviation

    gradient = 1.0 / max_deviation
    score = max(0.0, 1.0 - gradient * deviation)

    return score, deviation


def score_binary(status: str) -> float:
    """
    Calculate binary score (PASS = 1.0, FAIL = 0.0).

    Args:
        status: Test status ("PASS" or "FAIL")

    Returns:
        Score (1.0 or 0.0)
    """
    return 1.0 if status.upper() == "PASS" else 0.0


# ============================================================================
# Main Scoring Function
# ============================================================================


def score_test_item(test_item: dict, config: ScoringConfig | None = None) -> TestItemScoreResult:
    """
    Calculate score for a single test item.

    Args:
        test_item: Test item dictionary from iPLAS
        config: Optional scoring configuration

    Returns:
        TestItemScoreResult with score and metadata
    """
    name = test_item.get("NAME", "")
    status = test_item.get("STATUS", "FAIL")
    value_str = test_item.get("VALUE", "")
    ucl_str = test_item.get("UCL", "")
    lcl_str = test_item.get("LCL", "")

    value = _parse_float(value_str)
    ucl = _parse_float(ucl_str)
    lcl = _parse_float(lcl_str)

    # Determine scoring type
    scoring_type = config.scoring_type if config else detect_scoring_type(test_item)

    # Calculate score based on type
    score = 0.0
    deviation = None

    if scoring_type == ScoringType.BINARY or value is None:
        score = score_binary(status)
        scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.SYMMETRICAL:
        if ucl is not None and lcl is not None:
            alpha = _get_config_param(config, "alpha", scoring_type) or 0.8
            score, deviation = score_symmetrical(value, ucl, lcl, alpha)
        else:
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.SYMMETRICAL_NL:
        if ucl is not None and lcl is not None:
            target_score = _get_config_param(config, "target_score", scoring_type) or 0.8
            target_deviation = _get_config_param(config, "target_deviation", scoring_type) or 2.5
            score, deviation = score_symmetrical_nl(value, ucl, lcl, target_score, target_deviation)
        else:
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.EVM:
        target_evm = _get_config_param(config, "target", scoring_type) or -30.0
        target_score = _get_config_param(config, "target_score", scoring_type) or 0.9
        score, deviation = score_evm(value, target_evm, target_score)

    elif scoring_type == ScoringType.THROUGHPUT:
        if lcl is not None:
            target = _get_config_param(config, "target", scoring_type)
            min_score = _get_config_param(config, "min_score", scoring_type) or 0.4
            target_score = _get_config_param(config, "target_score", scoring_type) or 0.9
            score, deviation = score_throughput(value, lcl, target, min_score, target_score)
        else:
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.ASYMMETRICAL:
        if ucl is not None and lcl is not None:
            target = _get_config_param(config, "target", scoring_type)
            if target is None:
                target = (ucl + lcl) / 2  # Fall back to center
            alpha = _get_config_param(config, "alpha", scoring_type) or 0.4
            score, deviation = score_asymmetrical(value, ucl, lcl, target, alpha)
        else:
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.PER_MASK:
        max_dev = _get_config_param(config, "max_deviation", scoring_type)
        if max_dev is None:
            max_dev = ucl if ucl is not None else 10.0
        score, deviation = score_per_mask(value, 0.0, max_dev)

    return TestItemScoreResult(test_item_name=name, value=value, ucl=ucl, lcl=lcl, status=status, scoring_type=scoring_type, score=score, deviation=deviation)


def score_record(record: dict, config_map: dict[str, ScoringConfig], include_binary_in_overall: bool = True) -> RecordScoreResult:
    """
    Calculate scores for all test items in a record.

    Args:
        record: Test record from iPLAS API
        config_map: Map of test item name -> ScoringConfig
        include_binary_in_overall: Include binary items in overall score

    Returns:
        RecordScoreResult with individual and aggregate scores
    """
    isn = record.get("ISN", "")
    device_id = record.get("DeviceId", "")
    station = record.get("station", "")
    test_start_time = record.get("Test Start Time", "")
    test_status = record.get("Test Status", "")

    test_items = record.get("TestItem", [])

    item_scores: list[TestItemScoreResult] = []
    # Track weighted scores and their effective weights for proper weighted average
    # Using squared weights (weight^2) for more pronounced impact:
    # - weight=1 → effective=1 (normal)
    # - weight=2 → effective=4 (4x impact)
    # - weight=3 → effective=9 (9x impact)
    # - weight=0.5 → effective=0.25 (reduced impact)
    value_weighted_scores: list[tuple[float, float]] = []  # (score * effective_weight, effective_weight)
    bin_weighted_scores: list[tuple[float, float]] = []  # (score * effective_weight, effective_weight)
    all_weighted_scores: list[tuple[float, float]] = []  # (score * effective_weight, effective_weight)
    failed_count = 0

    for item in test_items:
        name = item.get("NAME", "")
        config = config_map.get(name)

        # Skip if disabled
        if config and not config.enabled:
            continue

        result = score_test_item(item, config)
        item_scores.append(result)

        # Use squared weight for more pronounced impact
        # UPDATED: Use name-based default weight when no config exists
        weight = config.weight if config else get_default_weight_by_name(name)
        effective_weight = weight * weight  # Squared for more impact
        weighted_score = result.score * effective_weight

        if result.scoring_type == ScoringType.BINARY:
            bin_weighted_scores.append((weighted_score, effective_weight))
            if include_binary_in_overall:
                all_weighted_scores.append((weighted_score, effective_weight))
        else:
            value_weighted_scores.append((weighted_score, effective_weight))
            all_weighted_scores.append((weighted_score, effective_weight))

        if result.status.upper() == "FAIL":
            failed_count += 1

    # Calculate aggregate scores using proper weighted average (sum of weighted scores / sum of weights)
    def weighted_average(scores: list[tuple[float, float]]) -> float:
        if not scores:
            return 0.0
        total_weighted = sum(ws for ws, _ in scores)
        total_weight = sum(w for _, w in scores)
        return total_weighted / total_weight if total_weight > 0 else 0.0

    overall_score = weighted_average(all_weighted_scores)
    value_items_score = weighted_average(value_weighted_scores) if value_weighted_scores else None
    bin_items_score = weighted_average(bin_weighted_scores) if bin_weighted_scores else None

    return RecordScoreResult(
        isn=isn,
        device_id=device_id,
        station=station,
        test_start_time=test_start_time,
        test_status=test_status,
        overall_score=overall_score,
        value_items_score=value_items_score,
        bin_items_score=bin_items_score,
        test_item_scores=item_scores,
        total_items=len(test_items),
        scored_items=len(item_scores),
        failed_items=failed_count,
    )


def calculate_scores(request: CalculateScoresRequest) -> CalculateScoresResponse:
    """
    Calculate scores for multiple records.

    Args:
        request: CalculateScoresRequest with records and configs

    Returns:
        CalculateScoresResponse with scored records and summary
    """
    # Build config map for quick lookup
    config_map: dict[str, ScoringConfig] = {config.test_item_name: config for config in request.scoring_configs}

    scored_records: list[RecordScoreResult] = []

    for record in request.records:
        result = score_record(record, config_map, request.include_binary_in_overall)
        scored_records.append(result)

    # Calculate summary statistics
    all_scores = [r.overall_score for r in scored_records]
    pass_count = sum(1 for r in scored_records if r.test_status.upper() == "PASS")
    fail_count = len(scored_records) - pass_count

    if all_scores:
        summary = ScoreSummary(
            average_score=mean(all_scores),
            min_score=min(all_scores),
            max_score=max(all_scores),
            median_score=median(all_scores),
            std_deviation=stdev(all_scores) if len(all_scores) > 1 else 0.0,
            total_records=len(scored_records),
            pass_records=pass_count,
            fail_records=fail_count,
        )
    else:
        summary = ScoreSummary(average_score=0.0, min_score=0.0, max_score=0.0, median_score=0.0, std_deviation=0.0, total_records=0, pass_records=0, fail_records=0)

    return CalculateScoresResponse(scored_records=scored_records, summary=summary)
