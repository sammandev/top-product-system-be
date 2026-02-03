"""
Scoring Service

Business logic for calculating test item scores using the Universal 0-10 Scoring System.

Score Range: 0.00 - 10.00
- Outside limits: 0.00
- Exactly at boundary (LCL/UCL): limit_score (default 1.00)
- At target: 10.00
- Linear interpolation between boundary and target

Scoring Types:
- Symmetrical: Target = midpoint of (UCL + LCL) / 2
- Asymmetrical: User-defined custom target with Policy (symmetrical/higher/lower)
- PER/MASK: UCL-only scoring, lower is better (best=0)
- EVM: UCL-only scoring, lower is better (best=-35 dB, gentle decay)
"""

import logging
import re
from statistics import mean, median, stdev

from ..schemas.scoring_schemas import (
    CalculateScoresRequest,
    CalculateScoresResponse,
    RecordScoreResult,
    ScoreSummary,
    ScoringConfig,
    ScoringPolicy,
    ScoringType,
    SCORING_TYPE_DEFAULTS,
    TestItemScoreResult,
)

logger = logging.getLogger(__name__)

# Score scale constants (0-10 scale as per new_scoring_top_product_v2.ipynb)
SCORE_MAX = 10.0
SCORE_MIN = 0.0
DEFAULT_LIMIT_SCORE = 1.0  # Score at UCL/LCL boundaries


# ============================================================================
# Helper Functions
# ============================================================================


def _parse_float(value: str | float | None) -> float | None:
    """Safely parse a float value."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi."""
    return lo if x < lo else hi if x > hi else x


def _round2(x: float) -> float:
    """Round to 2 decimal places."""
    return round(x, 2)


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


def _is_per_mask_item(test_item: dict) -> bool:
    """
    Check if test item is a PER or MASK item based on name pattern.

    Rules:
    - Contains "PER" (case insensitive)
    - Contains "MASK" but NOT "MASK MARGIN" or "MASKING" (strict match)
    - Must have UCL only (no LCL) for proper PER/MASK scoring

    Args:
        test_item: Test item dictionary

    Returns:
        True if this is a PER/MASK item suitable for per_mask scoring
    """
    name = str(test_item.get("NAME", "")).upper()

    # Check for PER
    if "PER" in name:
        return True

    # Check for MASK but not "MASK MARGIN" or compound words
    if "MASK" in name:
        # Exclude "MASK MARGIN", "MASKING", "MASKED", etc.
        excluded_patterns = ["MASK MARGIN", "MASKING", "MASKED", "MASK_MARGIN"]
        for pattern in excluded_patterns:
            if pattern in name:
                return False
        return True

    return False


def _is_evm_item(test_item: dict) -> bool:
    """
    Check if test item is an EVM item based on name pattern.

    Rules:
    - Contains "EVM" (case insensitive)
    - Must have UCL only (no LCL) - if LCL exists, use symmetrical scoring

    Args:
        test_item: Test item dictionary

    Returns:
        True if this is an EVM item suitable for EVM scoring
    """
    name = str(test_item.get("NAME", "")).upper()
    return "EVM" in name


# UPDATED: Added throughput item detection
def _is_throughput_item(test_item: dict) -> bool:
    """
    Check if test item is a Throughput item based on name pattern.

    Rules:
    - Contains "THROUGHPUT" (case insensitive)
    - Higher value is better, score increases as value approaches UCL

    Args:
        test_item: Test item dictionary

    Returns:
        True if this is a Throughput item suitable for throughput scoring
    """
    name = str(test_item.get("NAME", "")).upper()
    return "THROUGHPUT" in name


# ============================================================================
# Auto-Detection
# ============================================================================


def detect_scoring_type(test_item: dict) -> ScoringType:
    """
    Auto-detect the appropriate scoring type based on test item characteristics.

    Detection rules:
    - BINARY: for PASS/FAIL values without numeric control limits
    - PER_MASK: for items containing "PER" or "MASK" (strict) with UCL only
    - EVM: for items containing "EVM" with UCL only (no LCL)
    - SYMMETRICAL: for all other value items (UCL and/or LCL present)

    User can manually change to ASYMMETRICAL or other types if needed.
    """
    if not _is_value_item(test_item):
        return ScoringType.BINARY

    ucl = _parse_float(test_item.get("UCL"))
    lcl = _parse_float(test_item.get("LCL"))

    # Check for PER/MASK items (UCL-only, lower is better)
    if _is_per_mask_item(test_item):
        # PER/MASK typically has UCL only (upper limit for error rate)
        if ucl is not None and lcl is None:
            return ScoringType.PER_MASK

    # Check for EVM items (UCL-only, lower is better with gentle decay)
    if _is_evm_item(test_item):
        # EVM typically has UCL only - if LCL exists, use symmetrical scoring
        if ucl is not None and lcl is None:
            return ScoringType.EVM

    # UPDATED: Check for Throughput items (higher is better, target=UCL)
    if _is_throughput_item(test_item):
        # Throughput scoring: higher value is better, target is UCL
        if ucl is not None:
            return ScoringType.THROUGHPUT

    # All other value items default to SYMMETRICAL
    return ScoringType.SYMMETRICAL


def get_default_weight_by_name(item_name: str) -> float:
    """
    Get default weight based on test item NAME patterns.

    Returns:
        Weight value (default 1.0 if no pattern matches).
    """
    upper_name = item_name.upper()

    # UPDATED: Weight 3 for test items matching "TX{number}_POW" pattern
    # Matches: TX0_POW, TX1_POW, TX2_POW, TX12_POW, etc.
    if re.search(r'TX\d+_POW', upper_name):
        return 3.0

    # Weight 3 for test items containing "POW_OLD"
    if "POW_OLD" in upper_name:
        return 3.0

    # Weight 2 for test items containing "FIXTURE_OR_DUT_PROBLEM_POW"
    if "FIXTURE_OR_DUT_PROBLEM_POW" in upper_name:
        return 2.0

    return 1.0  # Default weight


# ============================================================================
# Scoring Algorithms (Universal 0-10 Scale)
# ============================================================================


def score_symmetrical(
    value: float,
    ucl: float,
    lcl: float,
    limit_score: float = DEFAULT_LIMIT_SCORE
) -> tuple[float, float]:
    """
    Calculate symmetrical score with centered target (UCL + LCL) / 2.
    
    Score range: 0.00 - 10.00
    - Outside limits: 0.00
    - At limit boundary: limit_score (default 1.00)
    - At target (center): 10.00
    - Linear interpolation between boundary and target

    Args:
        value: Measured value
        ucl: Upper Control Limit
        lcl: Lower Control Limit
        limit_score: Score at limit boundary (default: 1.0)

    Returns:
        Tuple of (score, deviation from target)
    """
    # Ensure lo < hi
    lo, hi = (lcl, ucl) if lcl <= ucl else (ucl, lcl)
    target = (lo + hi) / 2.0
    
    # Hard fail: outside limits
    if value > hi or value < lo:
        return SCORE_MIN, abs(value - target)
    
    # Exactly at boundary
    if value == lo or value == hi:
        return _round2(limit_score), abs(value - target)
    
    # At target
    if value == target:
        return SCORE_MAX, 0.0
    
    # Linear interpolation between boundary and target
    if value > target:
        # Upper half: target to UCL
        if hi == target:
            return _round2(limit_score), value - target
        frac = (hi - value) / (hi - target)
    else:
        # Lower half: LCL to target
        if target == lo:
            return _round2(limit_score), target - value
        frac = (value - lo) / (target - lo)
    
    frac = _clamp(frac, 0.0, 1.0)
    score = limit_score + (SCORE_MAX - limit_score) * frac
    return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), abs(value - target)


def score_asymmetrical(
    value: float,
    ucl: float,
    lcl: float,
    target: float,
    policy: ScoringPolicy = ScoringPolicy.SYMMETRICAL,
    limit_score: float = DEFAULT_LIMIT_SCORE
) -> tuple[float, float]:
    """
    Calculate asymmetrical score with user-defined target and policy.

    Score range: 0.00 - 10.00
    - Outside limits: 0.00
    - At limit boundary: limit_score (default 1.00)
    - At target: 10.00

    Policy determines scoring behavior:
    - SYMMETRICAL: Peak at target, linear decay to both limits
    - HIGHER: Perfect score at/above target, decay below target to LCL
    - LOWER: Perfect score at/below target, decay above target to UCL

    Args:
        value: Measured value
        ucl: Upper Control Limit
        lcl: Lower Control Limit
        target: User-defined target (between LCL and UCL)
        policy: Scoring policy (symmetrical/higher/lower)
        limit_score: Score at limit boundary (default: 1.0)

    Returns:
        Tuple of (score, deviation from target)
    """
    # Ensure lo < hi
    lo, hi = (lcl, ucl) if lcl <= ucl else (ucl, lcl)

    # Clamp target within limits
    target = _clamp(target, lo, hi)

    # Hard fail: outside limits
    if value > hi or value < lo:
        return SCORE_MIN, abs(value - target)

    # Exactly at boundary
    if value == lo or value == hi:
        return _round2(limit_score), abs(value - target)

    # At target
    if value == target:
        return SCORE_MAX, 0.0

    # Policy-based scoring
    if policy == ScoringPolicy.HIGHER:
        # Perfect score at/above target, decay below
        if value >= target:
            return SCORE_MAX, 0.0
        # Decay from target to LCL
        if target == lo:
            return _round2(limit_score), target - value
        frac = (value - lo) / (target - lo)
        frac = _clamp(frac, 0.0, 1.0)
        score = limit_score + (SCORE_MAX - limit_score) * frac
        return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), target - value

    elif policy == ScoringPolicy.LOWER:
        # Perfect score at/below target, decay above
        if value <= target:
            return SCORE_MAX, 0.0
        # Decay from target to UCL
        if hi == target:
            return _round2(limit_score), value - target
        frac = (hi - value) / (hi - target)
        frac = _clamp(frac, 0.0, 1.0)
        score = limit_score + (SCORE_MAX - limit_score) * frac
        return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), value - target

    else:  # SYMMETRICAL (default)
        # Linear interpolation with asymmetric slopes (peak at target)
        if value > target:
            # Upper half: target to UCL
            if hi == target:
                return _round2(limit_score), value - target
            frac = (hi - value) / (hi - target)
        else:
            # Lower half: LCL to target
            if target == lo:
                return _round2(limit_score), target - value
            frac = (value - lo) / (target - lo)

        frac = _clamp(frac, 0.0, 1.0)
        score = limit_score + (SCORE_MAX - limit_score) * frac
        return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), abs(value - target)


def score_per_mask(
    value: float,
    ucl: float,
    limit_score: float = DEFAULT_LIMIT_SCORE
) -> tuple[float, float]:
    """
    Calculate PER/MASK score (UCL-only, lower is better with best=0).

    Score range: 0.00 - 10.00
    - Above UCL: 0.00 (fail)
    - At UCL: limit_score (default 1.00)
    - At 0 (best): 10.00
    - Linear interpolation between 0 and UCL

    This scoring is used for Packet Error Rate (PER) and MASK test items
    where 0 is the ideal value and anything above UCL is a failure.

    Args:
        value: Measured value (PER/MASK reading)
        ucl: Upper Control Limit (failure threshold)
        limit_score: Score at UCL boundary (default: 1.0)

    Returns:
        Tuple of (score, deviation from best=0)
    """
    best = 0.0  # PER/MASK: best value is 0

    # Hard fail: above UCL
    if value > ucl:
        return SCORE_MIN, value

    # UPDATED: Check for perfect score FIRST (at or below best)
    # This handles edge case where value=0 and UCL=0
    if value <= best:
        return SCORE_MAX, 0.0

    # At UCL boundary (only reached if value > best)
    if value == ucl:
        return _round2(limit_score), ucl

    # Linear interpolation between best (0) and UCL
    # frac = 1 at best, 0 at UCL
    frac = (ucl - value) / (ucl - best)
    frac = _clamp(frac, 0.0, 1.0)
    score = limit_score + (SCORE_MAX - limit_score) * frac
    return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), value


def score_evm(
    value: float,
    ucl: float,
    limit_score: float = DEFAULT_LIMIT_SCORE,
    reference_best: float = -35.0,
    exponent: float = 0.25
) -> tuple[float, float]:
    """
    Calculate EVM score (UCL-only, lower is better with gentle decay).

    Score range: 0.00 - 10.00
    - Above UCL: 0.00 (fail)
    - At UCL: limit_score (default 1.00)
    - At reference_best (-35 dB): 10.00
    - Gentle decay using exponent (default 0.25 = gentler than linear)

    This scoring is used for Error Vector Magnitude (EVM) test items where
    lower values are better. EVM values are typically in dB (negative).
    The gentle decay (exponent < 1) means scores stay high for longer as
    values move away from reference_best.

    Algorithm from new_scoring_top_product_v2.ipynb:
        dist = (value - reference_best) / (ucl - reference_best)
        frac = (1.0 - dist) ** exponent
        score = limit_score + (10.0 - limit_score) * frac

    Args:
        value: Measured EVM value (typically negative dB)
        ucl: Upper Control Limit (failure threshold)
        limit_score: Score at UCL boundary (default: 1.0)
        reference_best: Best possible EVM value (default: -35.0 dB)
        exponent: Decay exponent (default: 0.25, gentler than linear)

    Returns:
        Tuple of (score, deviation from reference_best)
    """
    # Hard fail: above UCL
    if value > ucl:
        return SCORE_MIN, abs(value - reference_best)

    # At UCL boundary
    if value == ucl:
        return _round2(limit_score), abs(ucl - reference_best)

    # At or below reference_best (perfect score)
    if value <= reference_best:
        return SCORE_MAX, 0.0

    # Edge case: UCL equals reference_best
    if ucl == reference_best:
        return _round2(limit_score), 0.0

    # Gentle decay using exponent (from notebook algorithm)
    # dist: normalized distance from reference_best to UCL (0 to 1)
    dist = (value - reference_best) / (ucl - reference_best)
    dist = _clamp(dist, 0.0, 1.0)

    # frac: decay fraction using exponent (0.25 = gentler than linear)
    # When dist=0 (at ref), frac=1
    # When dist=1 (at UCL), frac=0
    frac = (1.0 - dist) ** exponent
    frac = _clamp(frac, 0.0, 1.0)

    score = limit_score + (SCORE_MAX - limit_score) * frac
    return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), abs(value - reference_best)


# UPDATED: Added throughput scoring function
def score_throughput(
    value: float,
    ucl: float,
    lcl: float | None = None,
    limit_score: float = DEFAULT_LIMIT_SCORE
) -> tuple[float, float]:
    """
    Calculate Throughput score (higher is better, target = UCL).

    Score range: 0.00 - 10.00
    - Above UCL (target): 10.00 (perfect, even exceeded target)
    - At UCL: 10.00 (perfect)
    - At LCL (or 0 if no LCL): limit_score (default 1.00)
    - Below LCL: 0.00 (fail)
    - Linear interpolation between LCL and UCL

    This scoring is used for Throughput test items where higher values
    are better and the target is the UCL (upper spec limit).

    Args:
        value: Measured throughput value
        ucl: Upper Control Limit (target - the ideal value)
        lcl: Lower Control Limit (optional, defaults to 0)
        limit_score: Score at LCL boundary (default: 1.0)

    Returns:
        Tuple of (score, deviation from UCL)
    """
    # Use LCL if provided, otherwise default to 0
    lower = lcl if lcl is not None else 0.0
    target = ucl  # Target is UCL for throughput

    # At or above UCL (target) - perfect score
    if value >= ucl:
        return SCORE_MAX, 0.0

    # Below LCL - fail
    if value < lower:
        return SCORE_MIN, abs(value - target)

    # At LCL boundary
    if value == lower:
        return _round2(limit_score), abs(target - lower)

    # Edge case: UCL equals LCL
    if ucl == lower:
        return _round2(limit_score), 0.0

    # Linear interpolation between LCL (limit_score) and UCL (10.0)
    # frac = 0 at LCL, 1 at UCL
    frac = (value - lower) / (ucl - lower)
    frac = _clamp(frac, 0.0, 1.0)
    score = limit_score + (SCORE_MAX - limit_score) * frac
    return _round2(_clamp(score, SCORE_MIN, SCORE_MAX)), abs(value - target)


def score_binary(status: str) -> float:
    """
    Calculate binary score (PASS = 10.0, FAIL = 0.0).

    Args:
        status: Test status ("PASS" or "FAIL")

    Returns:
        Score (10.0 or 0.0)
    """
    return SCORE_MAX if status.upper() == "PASS" else SCORE_MIN


# ============================================================================
# Main Scoring Function
# ============================================================================


def score_test_item(test_item: dict, config: ScoringConfig | None = None) -> TestItemScoreResult:
    """
    Calculate score for a single test item using the Universal 0-10 Scoring System.

    Args:
        test_item: Test item dictionary from iPLAS
        config: Optional scoring configuration

    Returns:
        TestItemScoreResult with score (0-10) and metadata
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

    # Get limit_score parameter (default 1.0 on 0-10 scale)
    limit_score = _get_config_param(config, "limit_score", scoring_type) or DEFAULT_LIMIT_SCORE

    # Get policy for asymmetrical scoring
    policy = config.policy if config else ScoringPolicy.SYMMETRICAL

    # Calculate score based on type
    score = SCORE_MIN
    deviation = None
    result_policy: ScoringPolicy | None = None

    if scoring_type == ScoringType.BINARY or value is None:
        score = score_binary(status)
        scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.PER_MASK:
        # PER/MASK scoring: UCL-only, lower is better (best=0)
        if ucl is not None:
            score, deviation = score_per_mask(value, ucl, limit_score)
        else:
            # No UCL - fall back to binary
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.EVM:
        # EVM scoring: UCL-only, lower is better with gentle decay (best=-35)
        if ucl is not None:
            # Get EVM-specific parameters from config or defaults
            defaults = SCORING_TYPE_DEFAULTS.get(ScoringType.EVM, {})
            reference_best = defaults.get("reference_best", -35.0)
            exponent = defaults.get("exponent", 0.25)
            score, deviation = score_evm(value, ucl, limit_score, reference_best, exponent)
        else:
            # No UCL - fall back to binary
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.SYMMETRICAL:
        if ucl is not None and lcl is not None:
            score, deviation = score_symmetrical(value, ucl, lcl, limit_score)
        elif ucl is not None:
            # UCL-only: treat as if LCL is far below (value item, higher is better up to UCL)
            # Use value as lower bound if value < ucl
            inferred_lcl = min(value - abs(ucl - value), value * 0.5) if value is not None else ucl - 10
            score, deviation = score_symmetrical(value, ucl, inferred_lcl, limit_score)
        elif lcl is not None:
            # LCL-only: treat as if UCL is far above (higher is better)
            inferred_ucl = max(value + abs(value - lcl), value * 1.5) if value is not None else lcl + 10
            score, deviation = score_symmetrical(value, inferred_ucl, lcl, limit_score)
        else:
            # No limits - binary fallback
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    elif scoring_type == ScoringType.ASYMMETRICAL:
        if ucl is not None and lcl is not None:
            target = _get_config_param(config, "target", scoring_type)
            if target is None:
                target = (ucl + lcl) / 2  # Fall back to center if no target specified
            score, deviation = score_asymmetrical(value, ucl, lcl, target, policy, limit_score)
            result_policy = policy  # Track policy used
        else:
            # Missing limits - fall back to binary
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    # UPDATED: Added throughput scoring case
    elif scoring_type == ScoringType.THROUGHPUT:
        # Throughput scoring: higher is better, target=UCL
        if ucl is not None:
            score, deviation = score_throughput(value, ucl, lcl, limit_score)
        else:
            # No UCL - fall back to binary
            score = score_binary(status)
            scoring_type = ScoringType.BINARY

    # Normalize score to 0-1 range for storage (divide by 10)
    normalized_score = score / SCORE_MAX

    return TestItemScoreResult(
        test_item_name=name,
        value=value,
        ucl=ucl,
        lcl=lcl,
        status=status,
        scoring_type=scoring_type,
        policy=result_policy,
        score=normalized_score,  # Store as 0-1, display as 0-10
        deviation=deviation
    )


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
