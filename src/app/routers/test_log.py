"""
Router for test log parsing and comparison endpoints.

UPDATED: All scoring now uses the Universal 0-10 Scoring System from scoring_service.py.
Scoring types: symmetrical, asymmetrical, per_mask, evm, binary, throughput.
The old category-based scoring (EVM/Frequency/PER/PA Power formulas) is removed.
"""

import logging
import shutil
from pathlib import Path
from statistics import mean
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.test_log import (
    CompareResponse,
    CompareResponseEnhanced,
    TestLogParseResponse,
    TestLogParseResponseEnhanced,
)
from ..schemas.scoring_schemas import ScoringConfig, ScoringPolicy, ScoringType
from ..services.scoring_service import score_test_item
from ..services.test_log_parser import TestLogParser, parse_test_log_criteria_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test-log", tags=["Test_Log_Processing"])


def validate_test_log_pattern(content: str) -> bool:
    """
    Check if file content contains valid test log patterns.

    Valid patterns:
    1. << START TESTING >> repeated 4 times
    2. =========[Start SFIS Test Result]====== format with TEST/UCL/LCL/VALUE
    3. *** Test flow *** format with Duration/Status/Result/Counter

    Args:
        content: File content as string

    Returns:
        True if file contains at least one valid pattern, False otherwise
    """
    # Pattern 1: << START TESTING >> repeated 4 times
    if "<< START TESTING >>  << START TESTING >>  << START TESTING >>  << START TESTING >>" in content:
        return True

    # Pattern 2: SFIS Test Result format
    if "=========[Start SFIS Test Result]======" in content and '"TEST" <"UCL","LCL">  ===> "VALUE"' in content:
        return True

    # Pattern 3: Test flow format
    if "*** Test flow ***" in content and "[Result]=" in content:
        return True

    return False


# Directory for temporary file storage
UPLOAD_DIR = Path("data/uploads/test_logs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Universal Scoring Post-Processing
# ============================================================================


def _build_scoring_config_map(scoring_configs_json: str | None) -> dict[str, ScoringConfig]:
    """
    Parse scoring configs JSON string and build a name-to-config map.

    Args:
        scoring_configs_json: Optional JSON string of scoring configurations

    Returns:
        Dictionary mapping test item names to ScoringConfig objects
    """
    import json

    config_map: dict[str, ScoringConfig] = {}
    if not scoring_configs_json:
        return config_map

    try:
        configs = json.loads(scoring_configs_json)
        for cfg in configs:
            name = cfg.get("test_item_name", "")
            if name:
                config_map[name] = ScoringConfig(
                    test_item_name=name,
                    scoring_type=ScoringType(cfg.get("scoring_type", "symmetrical")),
                    enabled=cfg.get("enabled", True),
                    weight=cfg.get("weight", 1.0),
                    target=cfg.get("target"),
                    policy=ScoringPolicy(cfg.get("policy", "symmetrical")) if cfg.get("policy") else ScoringPolicy.SYMMETRICAL,
                    limit_score=cfg.get("limit_score"),
                )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse scoring_configs JSON: {e}")

    return config_map


def _score_test_item_universal(
    test_item_name: str,
    value_str: str | None,
    usl: float | None,
    lsl: float | None,
    config_map: dict[str, ScoringConfig],
) -> dict:
    """
    Score a single test item using the universal scoring system.

    Returns a dict with {score, score_breakdown} fields.
    """
    iplas_format = {
        "NAME": test_item_name,
        "VALUE": str(value_str) if value_str is not None else "",
        "UCL": str(usl) if usl is not None else "",
        "LCL": str(lsl) if lsl is not None else "",
        "STATUS": "PASS",
    }

    config = config_map.get(test_item_name)
    result = score_test_item(iplas_format, config)

    # Convert score from 0-1 internal to 0-10 display
    score_0_10 = round(result.score * 10.0, 2)

    breakdown = {
        "scoring_type": result.scoring_type.value,
        "score": score_0_10,
        "target": result.target,
        "deviation": result.deviation,
        "weight": result.weight,
        "policy": result.policy.value if result.policy else None,
        "ucl": result.ucl,
        "lcl": result.lcl,
        "actual": result.value,
    }

    return {"score": score_0_10, "score_breakdown": breakdown}


def _apply_universal_scoring_to_parse_result(result: dict, config_map: dict[str, ScoringConfig]) -> dict:
    """
    Post-process a parse result dict to replace old scoring with universal scoring.

    Modifies the result dict in-place and returns it.
    """
    scored_values: list[float] = []

    for item in result.get("parsed_items_enhanced", []):
        # Only score value-type items (not binary)
        if item.get("is_value_type") and item.get("numeric_value") is not None:
            # UPDATED: By default, only score Criteria items (those with UCL or LCL).
            # Non-Criteria items (no limits) are skipped unless user explicitly configured them.
            has_limits = item.get("usl") is not None or item.get("lsl") is not None
            has_explicit_config = item["test_item"] in config_map
            if not has_limits and not has_explicit_config:
                item["score"] = None
                item["score_breakdown"] = None
                continue

            scoring_result = _score_test_item_universal(
                item["test_item"],
                item.get("value"),
                item.get("usl"),
                item.get("lsl"),
                config_map,
            )
            item["score"] = scoring_result["score"]
            item["score_breakdown"] = scoring_result["score_breakdown"]
            scored_values.append(scoring_result["score"])
        else:
            # Non-value items: binary scoring (PASS=10, FAIL=0)
            item["score"] = None
            item["score_breakdown"] = None

    # Recalculate aggregate scores
    if scored_values:
        result["avg_score"] = round(mean(scored_values), 2)
        sorted_scores = sorted(scored_values)
        n = len(sorted_scores)
        result["median_score"] = round(
            sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2,
            2,
        )
    else:
        result["avg_score"] = None
        result["median_score"] = None

    return result


def _apply_universal_scoring_to_compare_result(result: dict, config_map: dict[str, ScoringConfig]) -> dict:
    """
    Post-process a compare result dict to replace old scoring with universal scoring.

    Modifies the result dict in-place and returns it.
    """
    # Score each item in comparison_value_items and comparison_non_value_items
    for item_list_key in ("comparison_value_items", "comparison_non_value_items"):
        for compare_item in result.get(item_list_key, []):
            item_scored_values: list[float] = []

            for per_isn in compare_item.get("per_isn_data", []):
                # Try to score if we have a numeric value
                if per_isn.get("is_value_type") and per_isn.get("numeric_value") is not None:
                    # UPDATED: By default, only score Criteria items (those with UCL or LCL).
                    # Non-Criteria items (no limits) are skipped unless user explicitly configured them.
                    has_limits = compare_item.get("usl") is not None or compare_item.get("lsl") is not None
                    has_explicit_config = compare_item["test_item"] in config_map
                    if not has_limits and not has_explicit_config:
                        per_isn["score"] = None
                        per_isn["score_breakdown"] = None
                        continue

                    scoring_result = _score_test_item_universal(
                        compare_item["test_item"],
                        per_isn.get("value"),
                        compare_item.get("usl"),
                        compare_item.get("lsl"),
                        config_map,
                    )
                    per_isn["score"] = scoring_result["score"]
                    per_isn["score_breakdown"] = scoring_result["score_breakdown"]
                    item_scored_values.append(scoring_result["score"])
                else:
                    per_isn["score"] = None
                    per_isn["score_breakdown"] = None

            # Recalculate aggregate scores for this compare item
            if item_scored_values:
                compare_item["avg_score"] = round(mean(item_scored_values), 2)
                sorted_scores = sorted(item_scored_values)
                n = len(sorted_scores)
                compare_item["median_score"] = round(
                    sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2,
                    2,
                )
            else:
                compare_item["avg_score"] = None
                compare_item["median_score"] = None

    # Recalculate per-file avg_score in file_summaries
    for file_summary in result.get("file_summaries", []):
        isn = file_summary.get("isn")
        if isn is None:
            continue

        # Collect all scores for this ISN
        isn_scores: list[float] = []
        for item_list_key in ("comparison_value_items", "comparison_non_value_items"):
            for compare_item in result.get(item_list_key, []):
                for per_isn in compare_item.get("per_isn_data", []):
                    if per_isn.get("isn") == isn and per_isn.get("score") is not None:
                        isn_scores.append(per_isn["score"])

        file_summary["avg_score"] = round(mean(isn_scores), 2) if isn_scores else None

    return result


@router.post(
    "/parse",
    response_model=None,
    summary="Parse test log file or archive (with optional criteria filtering)",
    description="""
    Upload and parse test log file (.txt) or archive (.zip, .rar, .7z).

    **Optional criteria filtering:**
    - Upload .json criteria file to filter and score test items
    - JSON format: {"criteria": [{"test_item": "...", "ucl": 20, "lcl": 10, "target": 15}]}
    - Enhanced response includes metadata, classification, scoring with LaTeX formulas

    **Parser behavior:**
    - Flexible markers (any number of '=' characters)
    - USL/LSL spec limits extracted
    - FAIL items included for tracking
    - PASS/VALUE items excluded
    - ISN extraction from filename pattern: [Station]_[ISN]_[Date]
    """,
)
async def parse_test_log(
    file: Annotated[UploadFile, File(description="Test log file (.txt) or archive (.zip, .rar, .7z) to parse")],
    criteria_file: Annotated[UploadFile | None, File(description="Optional .ini or .json criteria file for filtering")] = None,
    show_only_criteria: Annotated[bool, Form(description="If true, only show items matching criteria")] = False,
    scoring_configs: Annotated[str | None, Form(description="Optional JSON string of scoring configurations")] = None,
) -> TestLogParseResponse | TestLogParseResponseEnhanced | JSONResponse:
    """
    Parse a test log file or archive and extract test items.

    Args:
        file: Uploaded .txt file or archive
        criteria_file: Optional .ini criteria file for filtering
        show_only_criteria: If True, only return items matching criteria

    Returns:
        TestLogParseResponse (basic) or TestLogParseResponseEnhanced (with criteria)
        or JSONResponse (for archives)

    Raises:
        HTTPException 400: If file type is invalid
        HTTPException 500: If parsing fails
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    filename_lower = file.filename.lower()

    # Check if file is archive
    is_archive = filename_lower.endswith((".zip", ".rar", ".7z"))
    is_txt = filename_lower.endswith(".txt")

    if not is_archive and not is_txt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only .txt, .zip, .rar, and .7z files are accepted.")

    # Parse criteria file if provided (JSON format only)
    criteria_rules = None
    if criteria_file:
        if not criteria_file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Criteria file must have a filename.")

        criteria_filename_lower = criteria_file.filename.lower()
        if not criteria_filename_lower.endswith(".json"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Criteria file must be .json format.")

        try:
            criteria_content = await criteria_file.read()
            criteria_rules = parse_test_log_criteria_file(criteria_content, criteria_file.filename)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse criteria file: {str(e)}") from e

    # Save uploaded file temporarily
    temp_file_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if is_archive:
            # Extract archive to get .txt files
            extracted_files = TestLogParser.extract_archive(str(temp_file_path))

            if len(extracted_files) == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No .txt files found in archive.")

            # Validate patterns in extracted files
            valid_files = []
            for txt_file in extracted_files:
                try:
                    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if validate_test_log_pattern(content):
                            valid_files.append(txt_file)
                except Exception:
                    # Skip files that can't be read
                    continue

            if len(valid_files) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid test log files found in archive. Files must contain one of the required patterns: '<< START TESTING >>' (repeated 4 times), '=========[Start SFIS Test Result]======', or '*** Test flow ***'",
                )

            # If criteria provided, use enhanced parsing on first valid file
            # (archive mode with criteria currently processes first file only)
            # Build scoring config map from user-provided configs
            config_map = _build_scoring_config_map(scoring_configs)

            if criteria_rules:
                result = TestLogParser.parse_file_enhanced(valid_files[0], criteria_rules=criteria_rules, show_only_criteria=show_only_criteria)
                # Apply universal scoring post-processing
                _apply_universal_scoring_to_parse_result(result, config_map)
                response = TestLogParseResponseEnhanced(**result)
                return response
            else:
                # Parse archive (returns all valid files)
                # Note: We need to filter the archive parser to only process valid files
                result = TestLogParser.parse_archive(str(temp_file_path))
                # Filter results to only include valid files
                if isinstance(result, dict) and "files" in result:
                    valid_file_names = [Path(f).name for f in valid_files]
                    result["files"] = [f for f in result["files"] if f.get("filename") in valid_file_names]
                return JSONResponse(content=result)
        else:
            # Parse single .txt file - validate pattern first
            with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if not validate_test_log_pattern(content):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File does not contain valid test log format. Files must contain one of the required patterns: '<< START TESTING >>' (repeated 4 times), '=========[Start SFIS Test Result]======', or '*** Test flow ***'",
                    )

            # Build scoring config map from user-provided configs
            config_map = _build_scoring_config_map(scoring_configs)

            # Always use enhanced parsing (with or without criteria)
            result = TestLogParser.parse_file_enhanced(str(temp_file_path), criteria_rules=criteria_rules, show_only_criteria=show_only_criteria)
            # Apply universal scoring post-processing
            _apply_universal_scoring_to_parse_result(result, config_map)
            response = TestLogParseResponseEnhanced(**result)
            return response

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to parse file: {str(e)}") from e

    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        await file.close()
        if criteria_file:
            await criteria_file.close()


@router.post(
    "/compare",
    response_model=None,
    summary="Compare test logs (with optional criteria filtering and deviation analysis)",
    description="""
    Compare multiple test log files (.txt) or archives (.zip, .rar, .7z).

    **Enhanced with criteria support:**
    - Upload .json criteria file for filtering and scoring
    - JSON format: {"criteria": [{"test_item": "...", "ucl": 20, "lcl": 10, "target": 15}]}
    - Per-ISN deviation from median baseline (or criteria target)
    - Separated value-type items (numeric) from non-value items
    - Aggregate statistics: avg deviation, avg score, median score

    **Response includes:**
    - File summaries with ISNs
    - Per-ISN measurements, deviations, and scores
    - Baseline values (median or criteria target)
    - Value vs non-value item separation
    """,
)
async def compare_test_logs(
    files: Annotated[list[UploadFile], File(description="Test log files (.txt) or archives (.zip, .rar, .7z) to compare")],
    criteria_file: Annotated[UploadFile | None, File(description="Optional .json criteria file for filtering")] = None,
    show_only_criteria: Annotated[bool, Form(description="If true, only show items matching criteria")] = False,
    scoring_configs: Annotated[str | None, Form(description="Optional JSON string of scoring configurations")] = None,
) -> CompareResponse | CompareResponseEnhanced:
    """
    Compare test items across multiple test log files or archives.

    Args:
        files: List of uploaded files (minimum 1 archive or 2 .txt files)
        criteria_file: Optional .json criteria file for filtering
        show_only_criteria: If True, only return items matching criteria

    Returns:
        CompareResponse (basic) or CompareResponseEnhanced (with criteria)

    Raises:
        HTTPException 400: If insufficient .txt files or invalid file type
        HTTPException 500: If comparison fails
    """
    if len(files) < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least 1 file is required.")

    # Validate all files
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File with no name provided.")
        filename_lower = file.filename.lower()
        if not filename_lower.endswith((".txt", ".zip", ".rar", ".7z")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type for {file.filename}. Only .txt, .zip, .rar, and .7z files are accepted.")

    # Parse criteria file if provided (JSON format only)
    criteria_rules = None
    if criteria_file:
        if not criteria_file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Criteria file must have a filename.")

        criteria_filename_lower = criteria_file.filename.lower()
        if not criteria_filename_lower.endswith(".json"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Criteria file must be .json format.")

        try:
            criteria_content = await criteria_file.read()
            criteria_rules = parse_test_log_criteria_file(criteria_content, criteria_file.filename)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse criteria file: {str(e)}") from e

    temp_file_paths = []
    txt_file_paths = []

    try:
        # Save all uploaded files temporarily
        for file in files:
            temp_path = UPLOAD_DIR / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_path)

            # Check if archive
            if file.filename.lower().endswith((".zip", ".rar", ".7z")):
                # Extract archive and get .txt files
                extracted_files = TestLogParser.extract_archive(str(temp_path))

                # Validate patterns in extracted files
                for txt_file in extracted_files:
                    try:
                        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if validate_test_log_pattern(content):
                                txt_file_paths.append(txt_file)
                    except Exception:
                        # Skip files that can't be read
                        continue
            else:
                # Validate .txt file pattern before adding
                try:
                    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if validate_test_log_pattern(content):
                            txt_file_paths.append(str(temp_path))
                except Exception:
                    # Skip files that can't be read
                    pass

        if len(txt_file_paths) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No valid test log files found for comparison. Files must contain one of the required patterns: '<< START TESTING >>' (repeated 4 times), '=========[Start SFIS Test Result]======', or '*** Test flow ***'",
            )

        # Always use enhanced comparison for BY UPLOAD LOG feature
        result = TestLogParser.compare_files_enhanced(txt_file_paths, criteria_rules=criteria_rules, show_only_criteria=show_only_criteria)

        # Apply universal scoring post-processing
        config_map = _build_scoring_config_map(scoring_configs)
        _apply_universal_scoring_to_compare_result(result, config_map)

        response = CompareResponseEnhanced(**result)

        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to compare files: {str(e)}") from e

    finally:
        # Clean up temporary files
        for temp_path in temp_file_paths:
            path = Path(temp_path)
            if path.exists():
                path.unlink()

        for file in files:
            await file.close()

        if criteria_file:
            await criteria_file.close()


# ============================================================================
# Rescoring endpoint - uses universal scoring system like iPLAS API
# ============================================================================


class TestLogRescoreRequest(BaseModel):
    """Request model for rescoring test log items."""

    test_items: list[dict] = Field(..., description="List of test items in format: {test_item, value, usl, lsl, status}")
    scoring_configs: list[dict] = Field(default=[], description="List of scoring configurations per test item")
    include_binary_in_overall: bool = Field(default=True, description="Include binary (PASS/FAIL) items in overall score")


class TestLogRescoreItemResult(BaseModel):
    """Score result for a single test item."""

    test_item: str
    value: float | None = None
    usl: float | None = None
    lsl: float | None = None
    status: str
    scoring_type: str
    policy: str | None = None
    score: float = Field(description="Score on 0-10 scale")
    deviation: float | None = None
    weight: float = 1.0
    target: float | None = None


class TestLogRescoreResponse(BaseModel):
    """Response model for rescoring test log items."""

    test_item_scores: list[TestLogRescoreItemResult]
    overall_score: float = Field(description="Overall score on 0-10 scale")
    value_items_score: float | None = None
    total_items: int
    scored_items: int


def _convert_test_log_item_to_iplas_format(item: dict) -> dict:
    """Convert test log item format to iPLAS format for scoring."""
    return {
        "NAME": item.get("test_item", ""),
        "VALUE": item.get("value", ""),
        "UCL": str(item.get("usl", "")) if item.get("usl") is not None else "",
        "LCL": str(item.get("lsl", "")) if item.get("lsl") is not None else "",
        "STATUS": item.get("status", "PASS"),
    }


@router.post(
    "/rescore",
    response_model=TestLogRescoreResponse,
    summary="Rescore test log items using universal scoring system",
    description="""
    Rescore test log items using the same scoring algorithms as the iPLAS API.
    
    This endpoint accepts parsed test log items and scoring configurations,
    then applies the universal 0-10 scoring system with support for:
    - **symmetrical**: Linear scoring with centered target (UCL + LCL / 2)
    - **asymmetrical**: User-defined custom target with Policy (symmetrical/higher/lower)
    - **evm**: UCL-only scoring for EVM measurements (lower is better)
    - **per_mask**: UCL-only scoring for PER/MASK items (zero is best)
    - **binary**: Simple PASS/FAIL scoring
    - **throughput**: LCL-only scoring (higher is better)
    """,
)
async def rescore_test_log_items(request: TestLogRescoreRequest) -> TestLogRescoreResponse:
    """Rescore test log items using the universal scoring system."""

    # Build config map
    config_map: dict[str, ScoringConfig] = {}
    for cfg in request.scoring_configs:
        name = cfg.get("test_item_name", "")
        if name:
            config_map[name] = ScoringConfig(
                test_item_name=name,
                scoring_type=ScoringType(cfg.get("scoring_type", "symmetrical")),
                enabled=cfg.get("enabled", True),
                weight=cfg.get("weight", 1.0),
                target=cfg.get("target"),
                policy=ScoringPolicy(cfg.get("policy", "symmetrical")) if cfg.get("policy") else ScoringPolicy.SYMMETRICAL,
                limit_score=cfg.get("limit_score"),
                alpha=cfg.get("alpha"),
            )

    # Score each test item
    item_scores: list[TestLogRescoreItemResult] = []
    value_weighted_scores: list[tuple[float, float]] = []  # (score * weight^2, weight^2)
    all_weighted_scores: list[tuple[float, float]] = []

    for item in request.test_items:
        # Convert to iPLAS format
        iplas_item = _convert_test_log_item_to_iplas_format(item)
        name = iplas_item["NAME"]

        # Get config or use auto-detection
        config = config_map.get(name)

        # Score the item
        result = score_test_item(iplas_item, config)

        # Convert score from 0-1 to 0-10 for response
        score_0_10 = result.score * 10.0

        item_scores.append(
            TestLogRescoreItemResult(
                test_item=result.test_item_name,
                value=result.value,
                usl=result.ucl,
                lsl=result.lcl,
                status=result.status,
                scoring_type=result.scoring_type.value,
                policy=result.policy.value if result.policy else None,
                score=score_0_10,
                deviation=result.deviation,
                weight=result.weight,
                target=result.target,
            )
        )

        # Accumulate weighted scores
        weight = result.weight
        effective_weight = weight * weight
        weighted_score = result.score * effective_weight

        if result.scoring_type != ScoringType.BINARY:
            value_weighted_scores.append((weighted_score, effective_weight))

        if result.scoring_type != ScoringType.BINARY or request.include_binary_in_overall:
            all_weighted_scores.append((weighted_score, effective_weight))

    # Calculate aggregate scores
    def weighted_average(scores: list[tuple[float, float]]) -> float:
        if not scores:
            return 0.0
        total_weighted = sum(ws for ws, _ in scores)
        total_weight = sum(w for _, w in scores)
        return (total_weighted / total_weight) * 10.0 if total_weight > 0 else 0.0

    overall_score = weighted_average(all_weighted_scores)
    value_items_score = weighted_average(value_weighted_scores) if value_weighted_scores else None

    return TestLogRescoreResponse(
        test_item_scores=item_scores,
        overall_score=overall_score,
        value_items_score=value_items_score,
        total_items=len(request.test_items),
        scored_items=len(item_scores),
    )


@router.get("/health", summary="Health check for test log parser", description="Check if the test log parsing service is operational.", operation_id="health_check_test_log")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status message
    """
    return JSONResponse(content={"status": "healthy", "service": "Test Log Parser", "upload_dir": str(UPLOAD_DIR.absolute()), "upload_dir_exists": UPLOAD_DIR.exists()})
