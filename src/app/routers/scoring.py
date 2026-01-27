"""
Scoring Router

API endpoints for the test item scoring system.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies.authz import get_current_user
from ..models.user import User
from ..schemas.scoring_schemas import (
    CalculateScoresRequest,
    CalculateScoresResponse,
    ScoringConfig,
    ScoringType,
    SCORING_TYPE_DEFAULTS,
)
from ..services.scoring_service import calculate_scores, detect_scoring_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scoring", tags=["Scoring"])


@router.post(
    "/calculate",
    response_model=CalculateScoresResponse,
    summary="Calculate scores for test records",
    description="""
    Calculate scores for a batch of test records using configurable scoring algorithms.
    
    Supported scoring types:
    - **symmetrical**: Linear scoring with centered target (UCL + LCL / 2)
    - **symmetrical_nl**: Non-linear Gaussian scoring curve
    - **evm**: EVM scoring (lower/more negative is better)
    - **throughput**: Throughput scoring (higher is better)
    - **asymmetrical**: User-defined off-center target
    - **per_mask**: PER/MASK scoring (0 is best)
    - **binary**: Simple PASS/FAIL scoring
    
    If no scoring configs are provided, the system will auto-detect the appropriate
    scoring type based on test item characteristics (UCL/LCL presence, value patterns).
    """,
)
async def calculate_test_scores(request: CalculateScoresRequest, current_user: Annotated[User, Depends(get_current_user)]) -> CalculateScoresResponse:
    """Calculate scores for test records."""
    try:
        logger.info(f"User {current_user.username} calculating scores for {len(request.records)} records")

        result = calculate_scores(request)

        logger.info(f"Scored {len(result.scored_records)} records, average: {result.summary.average_score:.3f}")

        return result

    except Exception as e:
        logger.exception(f"Error calculating scores: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error calculating scores: {str(e)}")


@router.post(
    "/detect-types",
    response_model=list[dict],
    summary="Auto-detect scoring types for test items",
    description="""
    Analyzes test items from a single record and returns the recommended
    scoring type for each unique test item.
    """,
)
async def detect_scoring_types(record: dict, current_user: Annotated[User, Depends(get_current_user)]) -> list[dict]:
    """Auto-detect scoring types for test items in a record."""
    try:
        test_items = record.get("TestItem", [])
        seen_names: set[str] = set()
        results: list[dict] = []

        for item in test_items:
            name = item.get("NAME", "")
            if name and name not in seen_names:
                seen_names.add(name)
                scoring_type = detect_scoring_type(item)

                results.append({"test_item_name": name, "detected_type": scoring_type.value, "value": item.get("VALUE", ""), "ucl": item.get("UCL", ""), "lcl": item.get("LCL", ""), "default_params": SCORING_TYPE_DEFAULTS.get(scoring_type, {})})

        return results

    except Exception as e:
        logger.exception(f"Error detecting scoring types: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error detecting scoring types: {str(e)}")


@router.get("/types", response_model=list[dict], summary="Get available scoring types", description="Returns list of available scoring types with descriptions and default parameters.")
async def get_scoring_types() -> list[dict]:
    """Get all available scoring types with their descriptions."""
    return [
        {
            "type": ScoringType.SYMMETRICAL.value,
            "label": "Symmetrical",
            "description": "Linear scoring with centered target (UCL + LCL / 2)",
            "use_case": "TX Power, frequency measurements with symmetric limits",
            "params": ["alpha"],
            "defaults": SCORING_TYPE_DEFAULTS[ScoringType.SYMMETRICAL],
        },
        {
            "type": ScoringType.SYMMETRICAL_NL.value,
            "label": "Symmetrical (Gaussian)",
            "description": "Non-linear Gaussian scoring curve",
            "use_case": "When you want steeper decay away from target",
            "params": ["target_score", "target_deviation"],
            "defaults": SCORING_TYPE_DEFAULTS[ScoringType.SYMMETRICAL_NL],
        },
        {
            "type": ScoringType.EVM.value,
            "label": "EVM (Lower is Better)",
            "description": "For negative dB values like EVM - more negative = better",
            "use_case": "EVM measurements (typically -20 to -60 dB)",
            "params": ["target", "target_score"],
            "defaults": SCORING_TYPE_DEFAULTS[ScoringType.EVM],
        },
        {
            "type": ScoringType.THROUGHPUT.value,
            "label": "Throughput (Higher is Better)",
            "description": "Linear to target, exponential above target",
            "use_case": "Data throughput, speed measurements",
            "params": ["min_score", "target_score", "target"],
            "defaults": SCORING_TYPE_DEFAULTS[ScoringType.THROUGHPUT],
        },
        {
            "type": ScoringType.ASYMMETRICAL.value,
            "label": "Asymmetrical",
            "description": "Custom target between UCL and LCL",
            "use_case": "When optimal value is not centered between limits",
            "params": ["alpha", "target"],
            "defaults": SCORING_TYPE_DEFAULTS[ScoringType.ASYMMETRICAL],
        },
        {
            "type": ScoringType.PER_MASK.value,
            "label": "PER/MASK (Zero is Best)",
            "description": "Linear decrease from 1.0 at 0 to 0.0 at max",
            "use_case": "Packet Error Rate, Mask margin measurements",
            "params": ["max_deviation"],
            "defaults": SCORING_TYPE_DEFAULTS[ScoringType.PER_MASK],
        },
        {"type": ScoringType.BINARY.value, "label": "Binary (PASS/FAIL)", "description": "PASS = 1.0, FAIL = 0.0", "use_case": "Non-numeric test items", "params": [], "defaults": {}},
    ]


@router.post(
    "/preview",
    response_model=dict,
    summary="Preview scoring for a single test item",
    description="""
    Calculate and preview the score for a single test item value using specified
    scoring configuration. Useful for testing scoring parameters.
    """,
)
async def preview_score(value: float, scoring_type: ScoringType, ucl: float | None = None, lcl: float | None = None, config: ScoringConfig | None = None) -> dict:
    """Preview score calculation for a single value."""
    from ..services.scoring_service import (
        score_symmetrical,
        score_symmetrical_nl,
        score_evm,
        score_throughput,
        score_asymmetrical,
        score_per_mask,
        score_binary,
    )

    score = 0.0
    deviation = None

    try:
        if scoring_type == ScoringType.SYMMETRICAL:
            if ucl is not None and lcl is not None:
                alpha = config.alpha if config and config.alpha else 0.8
                score, deviation = score_symmetrical(value, ucl, lcl, alpha)

        elif scoring_type == ScoringType.SYMMETRICAL_NL:
            if ucl is not None and lcl is not None:
                target_score = config.target_score if config and config.target_score else 0.8
                target_deviation = config.target_deviation if config and config.target_deviation else 2.5
                score, deviation = score_symmetrical_nl(value, ucl, lcl, target_score, target_deviation)

        elif scoring_type == ScoringType.EVM:
            target_evm = config.target if config and config.target else -30.0
            target_score = config.target_score if config and config.target_score else 0.9
            score, deviation = score_evm(value, target_evm, target_score)

        elif scoring_type == ScoringType.THROUGHPUT:
            if lcl is not None:
                target = config.target if config and config.target else None
                min_score = config.min_score if config and config.min_score else 0.4
                target_score = config.target_score if config and config.target_score else 0.9
                score, deviation = score_throughput(value, lcl, target, min_score, target_score)

        elif scoring_type == ScoringType.ASYMMETRICAL:
            if ucl is not None and lcl is not None:
                target = config.target if config and config.target else (ucl + lcl) / 2
                alpha = config.alpha if config and config.alpha else 0.4
                score, deviation = score_asymmetrical(value, ucl, lcl, target, alpha)

        elif scoring_type == ScoringType.PER_MASK:
            max_dev = config.max_deviation if config and config.max_deviation else (ucl if ucl else 10.0)
            score, deviation = score_per_mask(value, 0.0, max_dev)

        elif scoring_type == ScoringType.BINARY:
            score = 1.0 if value > 0 else 0.0

        return {"value": value, "scoring_type": scoring_type.value, "ucl": ucl, "lcl": lcl, "score": score, "deviation": deviation, "score_percent": f"{score * 100:.1f}%"}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error calculating preview: {str(e)}")
