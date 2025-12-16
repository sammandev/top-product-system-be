import base64
import logging
import os
import re
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path as FilePath
from statistics import StatisticsError, median
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi_cache.decorator import cache
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import Base, get_db
from app.dependencies.authz import get_current_user
from app.external_services.dut_api_client import DUTAPIClient
from app.models.test_log import ScoreBreakdown
from app.models.top_product import TopProduct, TopProductMeasurement
from app.schemas.dut_schemas import (
    AdjustedPowerRequestSchema,
    AdjustedPowerResponseSchema,
    BatchDevicesRequestSchema,
    BatchDevicesResponseSchema,
    BatchLatestTestItemsRequestSchema,
    BatchLatestTestItemsResponseSchema,
    BatchTestItemsRequestSchema,
    BatchTestItemsResponseSchema,
    CombinedPAAdjustedPowerSchema,
    CompleteDUTInfoSchema,
    DeviceInfoSchema,
    DevicePeriodEntrySchema,
    DeviceSummarySchema,
    DeviceTestResultSchema,
    DUTIdentifierListSchema,
    DUTIdentifierSchema,
    DUTISNVariantListSchema,
    DUTProgressSchema,
    DUTRunSummarySchema,
    DUTTestSummarySchema,
    LatestTestsResponseSchema,
    ModelSchema,
    ModelSummaryRequestSchema,
    ModelSummarySchema,
    PAActualAdjustedISNGroupSchema,
    PAActualAdjustedResponseSchema,
    PAActualAdjustedStationDataSchema,
    PADiffResponseSchema,
    PADiffStationDataSchema,
    PAISNGroupSchema,
    PAItemValueSchema,
    PAStationDataSchema,
    PATrendStationDataSchema,
    PATrendStationItemSchema,
    PAAdjustedPowerDataItemSchema,
    PAAdjustedPowerScoredItemSchema,
    PAAdjustedPowerScoringResponseSchema,
    PASROMDataItemSchema,
    PASROMEnhancedResponseSchema,
    GenericTestItemSchema,
    SimplePATrendResponseSchema,
    SiteSchema,
    StationDeviceListSchema,
    StationDevicePeriodListSchema,
    StationFilterConfigSchema,
    StationLatestRecordSchema,
    StationLatestTestItemsSchema,
    StationProgressSchema,
    StationRecordResponseSchema,
    StationRecordRowSchema,
    StationRunSummarySchema,
    StationSchema,
    StationTestItemListSchema,
    StationTestSummarySchema,
    StationTopProductDetailSchema,
    StationTopProductsResponseSchema,
    SVNInfoSchema,
    TestItemDefinitionSchema,
    TestItemSchema,
    TestLogDownloadRequest,
    TestResultQuerySchema,
    TestResultRecordSchema,
    TopProductBatchResponseSchema,
    TopProductErrorSchema,
    TopProductResponseSchema,
    TopProductStationSummarySchema,
)
from app.services import dut_metadata_cache, dut_token_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dut",
    tags=["DUT_Management"],
    responses={404: {"description": "Not found"}},
)

METADATA_CACHE_TTL = dut_metadata_cache.DEFAULT_TTL
RECORD_CACHE_TTL = dut_metadata_cache.DEFAULT_RECORD_TTL
_MAX_TOP_PRODUCT_WINDOW = timedelta(days=7)
_SCORE_CRITERIA_PATTERN = re.compile(r"^\s*(?P<min>-?\d+(?:\.\d+)?)\s*(?:-\s*(?P<max>-?\d+(?:\.\d+)?))?\s*$")
_REQUIRED_DEVICE_FIELDS_ERROR = "Please fill all required fields (site_id, model_id, device_id, start_time, end_time)."


@dataclass(slots=True)
class DeviceResultQueryParams:
    site_id: str
    model_id: str
    device_ids: list[str]
    start_time: str
    end_time: str
    test_result: str
    model: str | None
    station_identifier: str | None


def _client_base_url(client: DUTAPIClient) -> str:
    base = getattr(client, "base_url", None)
    if isinstance(base, str) and base:
        return base
    return "mock"


def _normalise_identifier(value: Any) -> str | None:
    if value is None:
        return None
    return _normalize_str(str(value))


def _matches_device_filters(filters: set[str], device_id: Any, device_name: Any) -> bool:
    candidates = set()
    norm_id = _normalise_identifier(device_id)
    if norm_id:
        candidates.add(norm_id)
    norm_name = _normalise_identifier(device_name)
    if norm_name:
        candidates.add(norm_name)
    return any(candidate in filters for candidate in candidates)


def _expand_device_identifiers(values: Iterable[str]) -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        if raw_value is None:
            continue
        tokens = [token.strip() for token in re.split(r"[,\s]+", str(raw_value)) if token.strip()]
        for token in tokens:
            lowered = token.lower()
            if lowered in seen:
                continue
            identifiers.append(token)
            seen.add(lowered)
    return identifiers


async def _parse_device_results_form(
    site_id: Annotated[str, Form(description="Site identifier or name used to narrow device lookup (e.g., 2 or PTB).")],
    model_id: Annotated[str, Form(description="Model identifier or name to narrow device lookup (e.g., 44 or HH5K).")],
    device_id: Annotated[list[str], Form(description="Device identifiers (IDs or names | e.g., 1351 or 614670). Repeat this field or provide comma/whitespace separated values.")],
    start_time: Annotated[str, Form(description="Start timestamp in ISO format (e.g., 2023-01-01T00:00:00Z).")],
    end_time: Annotated[str, Form(description="End timestamp in ISO format (e.g., 2023-01-02T00:00:00Z).")],
    test_result: Annotated[str | None, Form(description="Test result filter: 'ALL', '1' (PASS), or '0' (FAIL).")] = "ALL",
    model: Annotated[str | None, Form(description="Optional model name/identifier override (e.g., HH5K).")] = "",
    station_identifier: Annotated[str | None, Form(description="Optional station identifier (numeric ID or name) to narrow device lookup.")] = None,
) -> DeviceResultQueryParams:
    required_values = {
        "site_id": site_id,
        "model_id": model_id,
        "device_id": device_id,
        "start_time": start_time,
        "end_time": end_time,
    }
    for key, value in required_values.items():
        if value is None:
            raise HTTPException(status_code=400, detail=_REQUIRED_DEVICE_FIELDS_ERROR)
        if isinstance(value, str):
            if not value.strip():
                raise HTTPException(status_code=400, detail=_REQUIRED_DEVICE_FIELDS_ERROR)
        elif isinstance(value, list):
            if not any(str(item).strip() for item in value if item is not None):
                raise HTTPException(status_code=400, detail=_REQUIRED_DEVICE_FIELDS_ERROR)

    raw_devices = device_id or []
    if isinstance(raw_devices, str):
        raw_devices = [raw_devices]
    device_identifiers = _expand_device_identifiers(raw_devices)
    if not device_identifiers:
        raise HTTPException(status_code=400, detail=_REQUIRED_DEVICE_FIELDS_ERROR)

    sanitized_model = None
    if model is not None:
        normalized_model = model.strip()
        if normalized_model and normalized_model.lower() not in {"null", "none"}:
            sanitized_model = normalized_model

    sanitized_test_result = (test_result or "ALL").strip().upper() or "ALL"

    sanitized_station = None
    if station_identifier and station_identifier.strip():
        sanitized_station = station_identifier.strip()

    return DeviceResultQueryParams(
        site_id=site_id.strip(),
        model_id=model_id.strip(),
        device_ids=device_identifiers,
        start_time=start_time.strip(),
        end_time=end_time.strip(),
        test_result=sanitized_test_result,
        model=sanitized_model,
        station_identifier=sanitized_station,
    )


def _extract_measurement_overrides(row: Any, run_index: int | None) -> tuple[float | None, float | None, float | None]:
    if not isinstance(row, list) or len(row) <= 3:
        return None, None, None
    numeric_values = [_to_float(value) for value in row[3:]]
    actual_candidates: list[float | None] = [value for value in numeric_values if value is not None]
    target_override: float | None = None
    score_override: float | None = None

    if len(numeric_values) >= 3:
        score_candidate = numeric_values[-1]
        target_candidate = numeric_values[-2]
        has_metadata = score_candidate is not None and 0.0 <= score_candidate <= 10.0 and target_candidate is not None and abs(target_candidate) > 10
        if has_metadata:
            score_override = score_candidate
            target_override = target_candidate
            actual_candidates = [value for value in numeric_values[:-2] if value is not None]

    if not actual_candidates:
        actual_candidates = [value for value in numeric_values if value is not None]

    actual_override = None
    if actual_candidates:
        if run_index is not None and 0 <= run_index < len(actual_candidates):
            actual_override = actual_candidates[run_index]
        else:
            for candidate in reversed(actual_candidates):
                if candidate is not None:
                    actual_override = candidate
                    break

    return actual_override, target_override, score_override


def _tokenize_test_item(value: str) -> list[str]:
    if not value:
        return []
    raw_tokens = [segment for segment in re.split(r"[-_\s]+", value) if segment]
    normalized: list[str] = []
    idx = 0
    while idx < len(raw_tokens):
        token = raw_tokens[idx]
        next_token = raw_tokens[idx + 1] if idx + 1 < len(raw_tokens) else None
        if token.isalpha() and next_token and next_token.isdigit() and len(next_token) <= 2:
            normalized.append(f"{token.lower()}{next_token}")
            idx += 2
            continue
        normalized.append(token.lower())
        idx += 1
    return normalized


def _strip_test_item_digits(value: str) -> str:
    tokens = _tokenize_test_item(value)
    cleaned: list[str] = []
    for token in tokens:
        match = re.match(r"^(tx|rx|pa|ant|rssi)(\d{1,2})$", token)
        if match:
            cleaned.append(match.group(1))
            continue
        cleaned.append(token)
    return "_".join(cleaned)


@dataclass(slots=True)
class TestItemSignature:
    group: str
    subgroup: str
    subgroup_base: str
    antenna: str
    antenna_base: str
    categories: tuple[str, ...]


def _build_test_item_signature(value: str) -> TestItemSignature | None:
    parsed = _parse_measurement_components(value)
    if parsed is None:
        return None
    group_key, subgroup, antenna, category = parsed
    group = (group_key or "").upper()
    subgroup_up = subgroup.upper() if subgroup else ""
    antenna_up = antenna.upper() if antenna else ""
    subgroup_base = re.sub(r"\d+$", "", subgroup_up)
    antenna_base = re.sub(r"\d+$", "", antenna_up)
    categories = tuple(token for token in (category.upper().split("_") if category else []) if token)
    return TestItemSignature(
        group=group,
        subgroup=subgroup_up,
        subgroup_base=subgroup_base or subgroup_up,
        antenna=antenna_up,
        antenna_base=antenna_base or antenna_up,
        categories=categories,
    )


def _compile_test_item_patterns(
    patterns: list[str] | None,
) -> tuple[list[re.Pattern[str]], list[list[str]], list["TestItemSignature | None"], list[bool]]:
    """
    Compile test item patterns for matching.

    Returns:
        - compiled: List of compiled regex patterns
        - tokenized: List of tokenized patterns (for fuzzy matching)
        - signatures: List of pattern signatures (for structural matching)
        - is_exact: List of booleans indicating if pattern is exact match (True) or fuzzy (False)
    """
    compiled: list[re.Pattern[str]] = []
    tokenized: list[list[str]] = []
    signatures: list[TestItemSignature | None] = []
    is_exact: list[bool] = []  # Track which patterns are exact matches

    if not patterns:
        return compiled, tokenized, signatures, is_exact

    for raw in patterns:
        if not raw:
            continue
        text_value = raw.strip()
        if not text_value:
            continue

        # Determine if this is an exact match pattern (no wildcards) vs a fuzzy/regex pattern
        has_regex_chars = ".*" in text_value or any(char in text_value for char in ["^", "$", "[", "]", "(", ")", "|", "+", "?", "{", "}", "*"])

        # Check if this looks like a complete test item name vs a search term
        # Complete test item names typically have structure like: WiFi_TX1_POW_5300 or BT_RX2_SENSITIVITY
        # They usually contain multiple underscores and specific patterns
        # Search terms like "tx", "evm", "pow" are typically short and simple
        #
        # UPDATED: Check for variant patterns like WiFi_TX_POW (missing digit) which should match
        # WiFi_TX1_POW, WiFi_TX2_POW, etc. These are general patterns, not exact matches.
        has_variant_pattern = re.search(r"_(TX|RX|PA)_", text_value, re.IGNORECASE)  # e.g., WiFi_TX_POW (no digit)

        is_complete_name = (
            "_" in text_value  # Has at least one underscore
            and len(text_value) > 10  # Reasonably long (not just "TX_1")
            and not text_value.islower()  # Contains uppercase (test items are usually UPPERCASE_MIXED)
            and not has_regex_chars  # No regex characters
            and not has_variant_pattern  # Not a variant pattern (missing digit after TX/RX/PA)
        )

        try:
            if is_complete_name:
                # This looks like a complete test item name - use exact match
                compiled.append(re.compile(f"^{re.escape(text_value)}$", re.IGNORECASE))
                is_exact.append(True)
            elif has_regex_chars:
                # Has regex characters - use as regex pattern
                compiled.append(re.compile(text_value, re.IGNORECASE))
                is_exact.append(False)
            else:
                # Simple search term like "tx", "evm", "pow" - use partial match
                # This allows "tx" to match "WiFi_TX1_POW"
                compiled.append(re.compile(re.escape(text_value), re.IGNORECASE))
                is_exact.append(False)
        except re.error:
            # If regex compilation fails, treat as literal string with exact match
            compiled.append(re.compile(f"^{re.escape(text_value)}$", re.IGNORECASE))
            is_exact.append(True)

        tokenized.append(_tokenize_test_item(text_value))
        signatures.append(_build_test_item_signature(text_value))

    return compiled, tokenized, signatures, is_exact


def _tokens_match(filter_tokens: list[str], item_tokens: list[str]) -> bool:
    if not filter_tokens:
        return True
    filter_idx = 0

    def _tokens_equivalent(expected: str, candidate: str) -> bool:
        if expected == candidate:
            return True
        if candidate.startswith(expected):
            suffix = candidate[len(expected) :]
            if suffix.isdigit() and 1 <= len(suffix) <= 2:
                return True
        if expected.startswith(candidate):
            suffix = expected[len(candidate) :]
            if suffix.isdigit() and 1 <= len(suffix) <= 2:
                return True
        return False

    for token in item_tokens:
        if _tokens_equivalent(filter_tokens[filter_idx], token):
            filter_idx += 1
            if filter_idx == len(filter_tokens):
                return True
    return False


def _signature_matches(filter_sig: TestItemSignature, item_sig: TestItemSignature) -> bool:
    if filter_sig.group and filter_sig.group != item_sig.group:
        return False
    if filter_sig.subgroup:
        if not (filter_sig.subgroup == item_sig.subgroup or filter_sig.subgroup_base == item_sig.subgroup_base):
            return False
    if filter_sig.antenna:
        if not (filter_sig.antenna == item_sig.antenna or filter_sig.antenna_base == item_sig.antenna_base):
            return False
    if filter_sig.categories and filter_sig.categories != ("RESULT",):
        item_categories = set(item_sig.categories)
        for token in filter_sig.categories:
            if token not in item_categories:
                return False
    return True


def _pattern_matches(
    test_item: str,
    compiled_patterns: list[re.Pattern[str]],
    tokenized_patterns: list[list[str]],
    signatures: list[TestItemSignature | None],
    is_exact: list[bool] | None = None,
) -> bool:
    """
    Check if test item matches any of the patterns.

    For exact match patterns (is_exact=True), only regex matching is used.
    For fuzzy patterns (is_exact=False), falls back to token and signature matching.
    """
    if not compiled_patterns and not tokenized_patterns and not signatures:
        return True

    # If is_exact is not provided, assume all patterns are fuzzy (old behavior)
    if is_exact is None:
        is_exact = [False] * len(compiled_patterns)

    # Try regex matching first
    for idx, pattern in enumerate(compiled_patterns):
        if pattern.search(test_item):
            return True
        # For exact match patterns, if regex doesn't match, don't try fuzzy matching
        if idx < len(is_exact) and is_exact[idx]:
            # This was an exact match pattern that didn't match - continue to next pattern
            continue

    # Only do fuzzy matching for patterns that aren't exact matches
    item_tokens = _tokenize_test_item(test_item)
    for idx, tokens in enumerate(tokenized_patterns):
        # Skip fuzzy matching if this was an exact match pattern
        if idx < len(is_exact) and is_exact[idx]:
            continue
        if _tokens_match(tokens, item_tokens):
            return True

    # Only do signature matching for patterns that aren't exact matches
    filter_signatures = signatures
    item_signature = _build_test_item_signature(test_item)
    if item_signature is None:
        return False

    for idx, signature in enumerate(filter_signatures):
        # Skip signature matching if this was an exact match pattern
        if idx < len(is_exact) and is_exact[idx]:
            continue
        if signature and _signature_matches(signature, item_signature):
            return True

    return False


async def _get_cached_sites(client: DUTAPIClient) -> list[dict[str, Any]]:
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "sites")
    return await dut_metadata_cache.get_or_set(cache_key, client.get_sites, ttl=METADATA_CACHE_TTL)


async def _get_cached_models(client: DUTAPIClient, site_id: int) -> list[dict[str, Any]]:
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "models", str(site_id))

    async def loader() -> list[dict[str, Any]]:
        return await client.get_models_by_site(site_id)

    return await dut_metadata_cache.get_or_set(cache_key, loader, ttl=METADATA_CACHE_TTL)


async def _get_cached_stations(client: DUTAPIClient, model_id: int) -> list[dict[str, Any]]:
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "stations", str(model_id))

    async def loader() -> list[dict[str, Any]]:
        return await client.get_stations_by_model(model_id)

    return await dut_metadata_cache.get_or_set(cache_key, loader, ttl=METADATA_CACHE_TTL)


async def _get_cached_station_devices(client: DUTAPIClient, station_id: int) -> list[dict[str, Any]]:
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "station-devices", str(station_id))

    async def loader() -> list[dict[str, Any]]:
        try:
            return await client.get_devices_by_station(station_id)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Skipping device catalogue for station %s due to upstream error: %s",
                station_id,
                exc,
            )
            return []
        except httpx.HTTPError as exc:
            logger.warning(
                "Transient upstream error fetching devices for station %s: %s",
                station_id,
                exc,
            )
            return []

    return await dut_metadata_cache.get_or_set(cache_key, loader, ttl=METADATA_CACHE_TTL)


async def _get_device_catalog(client: DUTAPIClient) -> dict[str, Any]:
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "device-catalog")

    async def loader() -> dict[str, Any]:
        catalog: dict[str, dict[str, int]] = {"by_id": {}, "by_name": {}}
        sites = await _get_cached_sites(client)
        for site in sites:
            site_id = _coerce_optional_int(site.get("id"))
            if site_id is None:
                continue
            models = await _get_cached_models(client, site_id)
            for model in models:
                model_id = _coerce_optional_int(model.get("id"))
                if model_id is None:
                    continue
                stations = await _get_cached_stations(client, model_id)
                for station in stations:
                    station_id = _coerce_optional_int(station.get("id"))
                    if station_id is None:
                        continue
                    devices = await _get_cached_station_devices(client, station_id)
                    for device in devices:
                        device_id = _coerce_optional_int(device.get("id"))
                        if device_id is None:
                            continue
                        catalog["by_id"][str(device_id)] = device_id
                        name = str(device.get("name") or device.get("device") or "").strip()
                        if name:
                            catalog["by_name"][name.lower()] = device_id
        return catalog

    catalog = await dut_metadata_cache.get_or_set(cache_key, loader, ttl=METADATA_CACHE_TTL)
    if not isinstance(catalog, dict):
        return {"by_id": {}, "by_name": {}}
    return catalog


async def _resolve_device_identifier(
    client: DUTAPIClient,
    identifier: str,
    *,
    station_hint: str | None = None,
    model_hint: str | None = None,
    site_hint: str | None = None,
) -> str:
    ident = str(identifier).strip()
    if not ident:
        raise HTTPException(status_code=400, detail="device_id is required")
    normalized = _normalize_str(ident)

    def _match_devices(devices: list[dict[str, Any]]) -> str | None:
        for device in devices:
            device_id = _coerce_optional_int(device.get("id"))
            name = str(device.get("name") or device.get("device") or "").strip()
            if device_id is not None and (str(device_id) == ident or _normalize_str(str(device_id)) == normalized):
                return str(device_id)
            if name and _normalize_str(name) == normalized and device_id is not None:
                return str(device_id)
        return None

    visited_stations: set[int] = set()

    async def _search_station(station_identifier: str) -> str | None:
        try:
            station_id = await _resolve_station_id(
                client,
                station_identifier,
                model_hint=model_hint,
                site_hint=site_hint,
            )
        except HTTPException:
            return None
        if station_id in visited_stations:
            return None
        visited_stations.add(station_id)
        devices = await _get_cached_station_devices(client, station_id)
        return _match_devices(devices)

    if station_hint:
        match = await _search_station(station_hint)
        if match:
            return match

    if model_hint:
        try:
            model_id = await _resolve_model_id(client, model_hint, site_hint=site_hint)
        except HTTPException:
            model_id = None
        if model_id is not None:
            stations = await _get_cached_stations(client, model_id)
            for station in stations:
                station_id = _coerce_optional_int(station.get("id"))
                if station_id is None:
                    continue
                if station_id in visited_stations:
                    continue
                visited_stations.add(station_id)
                devices = await _get_cached_station_devices(client, station_id)
                match = _match_devices(devices)
                if match:
                    return match

    if site_hint:
        try:
            site_id = await _resolve_site_id(client, site_hint)
        except HTTPException:
            site_id = None
        if site_id is not None:
            models = await _get_cached_models(client, site_id)
            for model in models:
                model_id = _coerce_optional_int(model.get("id"))
                if model_id is None:
                    continue
                stations = await _get_cached_stations(client, model_id)
                for station in stations:
                    station_id = _coerce_optional_int(station.get("id"))
                    if station_id is None:
                        continue
                    if station_id in visited_stations:
                        continue
                    visited_stations.add(station_id)
                    devices = await _get_cached_station_devices(client, station_id)
                    match = _match_devices(devices)
                    if match:
                        return match

    catalog = await _get_device_catalog(client)
    by_id: dict[str, int] = catalog.get("by_id", {})
    by_name: dict[str, int] = catalog.get("by_name", {})

    if ident in by_id:
        return str(by_id[ident])
    if normalized in by_name:
        return str(by_name[normalized])

    # Fallback: if we cannot resolve the identifier via cached catalogues,
    # return the original identifier to avoid dropping it from upstream payloads.
    return ident


@dataclass(slots=True)
class CriteriaRule:
    pattern: re.Pattern[str]
    usl: float | None
    lsl: float | None
    target: float | None


@dataclass(slots=True)
class MeasurementRow:
    name: str
    usl: float | None
    lsl: float | None
    latest: float | None


@dataclass(slots=True)
class MeasurementScore:
    test_item: str
    usl: float | None
    lsl: float | None
    target: float | None
    actual: float | None
    deviation: float | None
    score_value: float
    score_breakdown: dict | None = None


@dataclass(slots=True)
class StationEvaluation:
    station_id: int | None
    station_name: str
    dut_isn: str
    dut_id: int | None
    device_id: int | None
    device_name: str | None
    test_date: datetime | None
    test_duration: float | None
    error_item: str | None
    pass_count: int
    fail_count: int
    retest_count: int
    test_count: int
    score: float
    status: int | None
    order: int | None
    data: list[list[str | int | float | None]]
    measurements: list[MeasurementScore]
    site_name: str | None
    project_name: str | None
    model_name: str | None
    metadata: dict[str, Any] | None = None


def _select_station_criteria(
    criteria_map: dict[str, list[CriteriaRule]] | None,
    station_name: str,
    model_name: str | None,
) -> list[CriteriaRule]:
    if not criteria_map:
        return []
    normalized_station = _normalize_str(station_name)
    normalized_model = _normalize_str(model_name) if model_name else ""
    selected: list[CriteriaRule] = []
    for raw_key, rules in criteria_map.items():
        if not rules:
            continue
        key = _normalize_str(raw_key)
        if "|" in key:
            model_part, station_part = key.split("|", 1)
            if station_part == normalized_station and (not model_part or model_part == normalized_model):
                selected.extend(rules)
        elif key == normalized_station:
            selected.extend(rules)
    return selected


def _match_station_rule(rules: list[CriteriaRule], test_item: str) -> CriteriaRule | None:
    if not rules:
        return None
    sanitized_name = _strip_test_item_digits(test_item)
    for rule in rules:
        if rule.pattern.search(test_item):
            return rule
        if rule.pattern.search(sanitized_name):
            return rule
        sanitized_pattern = _strip_test_item_digits(rule.pattern.pattern)
        if sanitized_pattern and sanitized_pattern in sanitized_name:
            return rule
    return None


def _row_matches_rule(row: MeasurementRow, rule: CriteriaRule) -> bool:
    name = row.name
    if rule.pattern.search(name):
        return True
    sanitized_name = _strip_test_item_digits(name)
    if rule.pattern.search(sanitized_name):
        return True
    sanitized_pattern = _strip_test_item_digits(rule.pattern.pattern)
    if sanitized_pattern and sanitized_pattern in sanitized_name:
        return True
    return False


def _determine_target_value(
    rule: CriteriaRule | None,
    usl: float | None,
    lsl: float | None,
    actual: float | None,
    test_item_name: str | None = None,
) -> float | None:
    if actual is None:
        return None
    if rule is not None:
        row = MeasurementRow(name="", usl=usl, lsl=lsl, latest=actual)
        target = _compute_target_value(rule, row, actual, test_item_name)
        if target is not None:
            return target
    # Check for PER pattern - return 0.0 as target
    if test_item_name and ("PER" in test_item_name.upper() or "_PER_" in test_item_name.upper()):
        return 0.0
    # Check for POW_DIF_ABS pattern - return 0.0 as target (ideal = no difference)
    if test_item_name and "POW_DIF_ABS" in test_item_name.upper():
        return 0.0
    if usl is not None and lsl is not None:
        if lsl == 0:
            return 0.0
        return (usl + lsl) / 2
    if usl is not None:
        return usl
    if lsl is not None:
        return lsl
    return actual


DEFAULT_CRITERIA_PATH = FilePath(os.getenv("DUT_CRITERIA_PATH", "reference/conf_dut_criteria.ini"))
_CRITERIA_CACHE: dict[Path, dict[str, list[CriteriaRule]]] = {}
_CRITERIA_LINE_PATTERN = re.compile(r'^\s*"(?P<test>.+?)"\s*<(?P<usl>[^,]*),(?P<lsl>[^>]*)>\s*===\>\s*"(?P<target>.*)"\s*$')


def _parse_criteria_line(line: str) -> CriteriaRule | None:
    match = _CRITERIA_LINE_PATTERN.match(line.strip())
    if not match:
        return None
    test_pattern = match.group("test")
    usl = _to_float(match.group("usl"))
    lsl = _to_float(match.group("lsl"))
    target = _to_float(match.group("target"))
    try:
        compiled = re.compile(test_pattern, re.IGNORECASE)
    except re.error as exc:
        logger.warning("Invalid regex '%s' in criteria line: %s", test_pattern, exc)
        return None
    return CriteriaRule(pattern=compiled, usl=usl, lsl=lsl, target=target)


# Module-level dependency object to avoid calling Depends() in function defaults
dut_get_current_user_dependency = Depends(get_current_user)


def get_user_dut_client(current_user=dut_get_current_user_dependency) -> DUTAPIClient:
    token = dut_token_service.ensure_valid_access(current_user.username)
    if not token:
        raise HTTPException(status_code=401, detail="Please re-login via /api/auth/external-login")

    token_bundle = dut_token_service.get_tokens(current_user.username)
    if not token_bundle or not token_bundle.get("access"):
        raise HTTPException(status_code=401, detail="Please re-login via /api/auth/external-login")

    client = DUTAPIClient(base_url=os.getenv("DUT_API_BASE_URL", "http://192.168.180.56:9001"))
    client.access_token = token_bundle.get("access")
    client.refresh_token = token_bundle.get("refresh")

    expiry_str = token_bundle.get("expiry")
    if expiry_str:
        try:
            expiry_dt = datetime.fromisoformat(expiry_str)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=UTC)
            else:
                expiry_dt = expiry_dt.astimezone(UTC)
            client.token_expiry = expiry_dt
        except ValueError:
            client.token_expiry = None
    return client


# Module-level dependency object to avoid calling Depends() in function defaults
dut_client_dependency = Depends(get_user_dut_client)
dut_db_dependency = Depends(get_db)


@router.get(
    "/sites",
    tags=["DUT_Management"],
    summary="Retrieve all sites",
    response_model=list[SiteSchema],
    responses={
        200: {
            "description": "List of sites",
            "content": {
                "application/json": {
                    "example": [
                        {"id": 2, "name": "TY", "iplas_url": "...", "ftp_url": "..."},
                    ]
                }
            },
        }
    },
)
@cache(expire=300)  # Cache for 5 minutes (very stable metadata)
async def get_sites(client: DUTAPIClient = dut_client_dependency):
    """
    Get all sites.

    Returns:
        List of sites with their information
    """
    try:
        sites = await _get_cached_sites(client)
        return sites
    except Exception as e:
        logger.error(f"Error fetching sites: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/sites/{site_id}/models",
    tags=["DUT_Management"],
    summary="Retrieve models for a site",
    response_model=list[ModelSchema],
    responses={200: {"description": "List of models for the site"}},
)
@cache(expire=300)  # Cache for 5 minutes (very stable metadata)
async def get_models_by_site(
    site_id: str = Path(..., description="Site identifier (numeric ID or name) (e.g., 2 or PTB)"),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get all models for a specific site.

    Args:
        site_id: Site identifier (ID or name)

    Returns:
        List of models for the specified site
    """
    try:
        resolved_site_id = await _resolve_site_id(client, site_id)
        models = await _get_cached_models(client, resolved_site_id)
        return models
    except Exception as e:
        logger.error(f"Error fetching models for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/models/{model_id}/stations",
    tags=["DUT_Management"],
    summary="Retrieve stations for a model",
    response_model=list[StationSchema],
    responses={200: {"description": "List of stations for the model"}},
)
@cache(expire=300)  # Cache for 5 minutes (very stable metadata)
async def get_stations_by_model(
    model_id: str = Path(..., description="Model identifier (numeric ID or name) (e.g., 44 or HH5K)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to scope the model lookup (e.g., 2 or PTB).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get all stations for a specific model.

    Args:
        model_id: Model identifier (ID or name)

    Returns:
        List of stations for the specified model
    """
    try:
        site_hint = _select_identifier(site_identifier, label="site")
        model_id = _select_identifier(model_id, label="model")
        resolved_model_id = await _resolve_model_id(client, model_id, site_hint=site_hint)
        stations = await _get_cached_stations(client, resolved_model_id)
        if not isinstance(stations, list):
            raise HTTPException(status_code=502, detail="Unexpected response format from external API")
        stations_sorted = sorted(stations, key=lambda entry: _order_sort_components(entry))
        return stations_sorted
    except Exception as e:
        logger.error(f"Error fetching stations for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/stations/{station_id}/devices",
    tags=["DUT_Management"],
    summary="Retrieve devices assigned to a station",
    response_model=StationDeviceListSchema,
    responses={200: {"description": "Devices grouped with station metadata."}},
)
@cache(expire=120)  # Cache for 2 minutes (relatively stable)
async def get_devices_by_station(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to scope the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to scope the station lookup (e.g., 44 or HH5K).",
    ),
    status: str | None = Query(  # noqa: B008
        None,
        description="Optional device status filter: ALL (default), Active/Online, Inactive/Offline.",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get devices associated with a given test station.
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id = await _resolve_station_id(
        client,
        station_id,
        model_hint=model_hint,
        site_hint=site_hint,
    )
    try:
        devices = await client.get_devices_by_station(resolved_station_id)
    except Exception as exc:
        logger.error("Error fetching devices for station %s (%s): %s", station_id, resolved_station_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not isinstance(devices, list):
        raise HTTPException(status_code=502, detail="Unexpected response format from external API")
    filtered = _filter_entries_by_status(devices, status, default_active_codes={1, 2})
    metadata = await _get_station_metadata(client, resolved_station_id)
    enriched = _enrich_devices_with_context(filtered)
    data = [DeviceInfoSchema.model_validate(item) for item in enriched]
    data.sort(key=_device_sort_key_schema)
    return StationDeviceListSchema(**metadata, data=data)


@router.get(
    "/stations/{station_id}/devices/period",
    tags=["DUT_Management"],
    summary="Retrieve devices by station and time window",
    response_model=StationDevicePeriodListSchema,
    responses={
        200: {"description": "Devices seen by the station within the supplied time window."},
        400: {"description": "Invalid or missing time window parameters."},
    },
)
@cache(expire=60)  # Cache for 1 minute (time-windowed data)
async def get_devices_by_station_period(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to scope the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to scope the station lookup (e.g., 44 or HH5K).",
    ),
    start_time: datetime = Query(..., description="Inclusive start timestamp in ISO format (e.g., 2023-01-01T00:00:00Z)"),  # noqa: B008
    end_time: datetime = Query(..., description="Exclusive end timestamp in ISO format (e.g., 2023-01-02T00:00:00Z)"),  # noqa: B008
    status: str | None = Query(
        None,
        description="Optional device status filter: ALL (default), Active/Online, Inactive/Offline.",
    ),
    test_result: str | None = Query(
        None,
        include_in_schema=False,
        description="Deprecated: retained for backward compatibility with earlier PASS/FAIL filters.",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    List devices processed by a station during a specific time interval.
    """
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    legacy_result_filter = (test_result or "ALL").upper()
    if legacy_result_filter not in {"ALL", "PASS", "FAIL"}:
        legacy_result_filter = "ALL"
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id = await _resolve_station_id(
        client,
        station_id,
        model_hint=model_hint,
        site_hint=site_hint,
    )
    try:
        devices = await client.get_devices_by_period(resolved_station_id, start_time, end_time, legacy_result_filter)
    except Exception as exc:
        logger.error(
            "Error fetching devices by period for station %s (%s): %s",
            station_id,
            resolved_station_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not isinstance(devices, list):
        raise HTTPException(status_code=502, detail="Unexpected response format from external API")
    filtered = _filter_entries_by_status(devices, status, default_active_codes={1, 2})
    metadata = await _get_station_metadata(client, resolved_station_id)
    enriched = _enrich_devices_with_context(filtered)
    data = [DevicePeriodEntrySchema.model_validate(item) for item in enriched]
    data.sort(key=_device_sort_key_schema)
    return StationDevicePeriodListSchema(**metadata, data=data)


@router.get(
    "/stations/{station_id}/test-items",
    tags=["DUT_Management"],
    summary="Retrieve test items configured for a station",
    response_model=StationTestItemListSchema,
    responses={200: {"description": "Test items grouped with station metadata."}},
)
@cache(expire=300)  # Cache for 5 minutes (very stable metadata)
async def get_test_items_by_station(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to scope the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to scope the station lookup (e.g., 44 or HH5K).",
    ),
    status: str | None = Query(
        None,
        description="Optional test item status filter: ALL (default) or Active/Online to show only enabled items.",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Fetch the upstream test item catalogue for a station.
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id = await _resolve_station_id(
        client,
        station_id,
        model_hint=model_hint,
        site_hint=site_hint,
    )
    try:
        items = await client.get_test_items_by_station(resolved_station_id)
    except Exception as exc:
        logger.error("Error fetching test items for station %s (%s): %s", station_id, resolved_station_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not isinstance(items, list):
        raise HTTPException(status_code=502, detail="Unexpected response format from external API")
    filtered = _filter_entries_by_status(items, status)
    metadata = await _get_station_metadata(client, resolved_station_id)
    enriched = _enrich_test_items_with_context(filtered)
    data = [TestItemSchema.model_validate(item) for item in enriched]
    return StationTestItemListSchema(**metadata, data=data)


@router.post(
    "/test-items/batch",
    tags=["DUT_Management"],
    summary="Fetch test items for multiple stations at once",
    response_model=BatchTestItemsResponseSchema,
    responses={200: {"description": "Test items for all requested stations grouped with metadata."}},
)
async def get_test_items_batch(
    request: BatchTestItemsRequestSchema,
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Efficiently fetch test item catalogs for multiple stations in a single request.
    Useful for frontend applications that need to populate test item filters for selected stations.
    """
    results: list[StationTestItemListSchema] = []

    for station_identifier in request.station_identifiers:
        try:
            # Resolve station ID
            resolved_station_id = await _resolve_station_id(
                client,
                station_identifier,
                model_hint=request.model_identifier,
                site_hint=request.site_identifier,
            )

            # Fetch test items
            items = await client.get_test_items_by_station(resolved_station_id)
            if not isinstance(items, list):
                logger.warning("Unexpected response format for station %s", station_identifier)
                continue

            # Filter by status if requested
            filtered = _filter_entries_by_status(items, request.status)

            # Get metadata and enrich items
            metadata = await _get_station_metadata(client, resolved_station_id)
            enriched = _enrich_test_items_with_context(filtered)
            data = [TestItemSchema.model_validate(item) for item in enriched]

            results.append(StationTestItemListSchema(**metadata, data=data))
        except Exception as exc:
            logger.warning("Failed to fetch test items for station %s: %s", station_identifier, exc)
            # Continue with other stations even if one fails
            continue

    return BatchTestItemsResponseSchema(stations=results)


@router.post(
    "/test-items/batch/filtered",
    tags=["DUT_Management"],
    summary="Fetch filtered test items for multiple stations (excludes items with both limits as 0)",
    response_model=BatchTestItemsResponseSchema,
    responses={200: {"description": "Filtered test items for all requested stations grouped with metadata."}},
)
async def get_test_items_batch_filtered(
    request: BatchTestItemsRequestSchema,
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Efficiently fetch test item catalogs for multiple stations in a single request.
    Excludes test items where both upperlimit=0 AND lowerlimit=0.

    These filtered items are useful for top product analysis where test items without
    meaningful limits are not relevant. Items with only one limit as 0 are still included.
    """
    results: list[StationTestItemListSchema] = []

    for station_identifier in request.station_identifiers:
        try:
            # Resolve station ID
            resolved_station_id = await _resolve_station_id(
                client,
                station_identifier,
                model_hint=request.model_identifier,
                site_hint=request.site_identifier,
            )

            # Fetch test items
            items = await client.get_test_items_by_station(resolved_station_id)
            if not isinstance(items, list):
                logger.warning("Unexpected response format for station %s", station_identifier)
                continue

            # Filter by status if requested
            filtered = _filter_entries_by_status(items, request.status)

            # Get metadata and enrich items
            metadata = await _get_station_metadata(client, resolved_station_id)
            enriched = _enrich_test_items_with_context(filtered)

            # Filter out test items where both upperlimit=0 and lowerlimit=0
            # These items are not useful for finding top products
            # Keep items where only one limit is 0 or both are null/non-zero
            valid_items = []
            for item in enriched:
                upper = item.get("upperlimit")
                lower = item.get("lowerlimit")
                # Exclude only if BOTH limits are exactly 0
                if upper == 0 and lower == 0:
                    continue
                valid_items.append(item)

            data = [TestItemSchema.model_validate(item) for item in valid_items]

            results.append(StationTestItemListSchema(**metadata, data=data))
        except Exception as exc:
            logger.warning("Failed to fetch filtered test items for station %s: %s", station_identifier, exc)
            # Continue with other stations even if one fails
            continue

    return BatchTestItemsResponseSchema(stations=results)


@router.post(
    "/devices/batch",
    tags=["DUT_Management"],
    summary="Fetch devices for multiple stations at once",
    response_model=BatchDevicesResponseSchema,
    responses={200: {"description": "Devices for all requested stations grouped with metadata."}},
)
async def get_devices_batch(
    request: BatchDevicesRequestSchema,
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Efficiently fetch device catalogs for multiple stations in a single request.
    Useful for frontend applications that need to populate device filters for selected stations.
    """
    results: list[StationDeviceListSchema] = []

    for station_identifier in request.station_identifiers:
        try:
            # Resolve station ID
            resolved_station_id = await _resolve_station_id(
                client,
                station_identifier,
                model_hint=request.model_identifier,
                site_hint=request.site_identifier,
            )

            # Fetch devices
            devices = await client.get_devices_by_station(resolved_station_id)
            if not isinstance(devices, list):
                logger.warning("Unexpected response format for station %s", station_identifier)
                continue

            # Filter by status if requested
            filtered = _filter_entries_by_status(devices, request.status)

            # Get metadata
            metadata = await _get_station_metadata(client, resolved_station_id)
            data = [DeviceInfoSchema.model_validate(device) for device in filtered]

            results.append(StationDeviceListSchema(**metadata, data=data))
        except Exception as exc:
            logger.warning("Failed to fetch devices for station %s: %s", station_identifier, exc)
            # Continue with other stations even if one fails
            continue

    return BatchDevicesResponseSchema(stations=results)


@router.post(
    "/test-items/latest/batch",
    tags=["DUT_Management"],
    summary="Fetch latest test items for multiple stations based on DUT ISN",
    response_model=BatchLatestTestItemsResponseSchema,
    responses={200: {"description": "Latest test items for all requested stations based on DUT ISN."}},
)
async def get_latest_test_items_batch(
    request: BatchLatestTestItemsRequestSchema,
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Efficiently fetch latest test item names for multiple stations based on a DUT ISN.
    This endpoint retrieves test items from the latest test run for each station,
    combining value-based, non-value, and non-value BIN test items.
    """

    # Treat blank/placeholder identifiers as absent to avoid over-filtering
    def _clean_hint(value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed or trimmed.lower() in {"none", "null", "string"}:
            return None
        return trimmed

    request.site_identifier = _clean_hint(request.site_identifier)  # type: ignore[assignment]
    request.model_identifier = _clean_hint(request.model_identifier)  # type: ignore[assignment]

    return await _build_latest_test_items_response(client, request)


@router.post(
    "/stations/{station_id}/top-products",
    tags=["DUT_Management"],
    summary="Identify the top DUTs for a specific station within a time window",
    description=(
        "Retrieve and rank the top-performing DUT products for a test station within a specified time window. "
        "Products are scored based on test criteria and sorted by overall data score (highest first). "
        "Results can be limited to return only the best N products (default: 5, max: 100). "
        "\n\n**Time Window Constraints:**\n"
        "- Maximum allowed time window: 7 days\n"
        "- Exceeding 7 days returns 400 Bad Request\n"
        "- Exactly 7 days is accepted\n\n"
        "**Sorting & Limiting:**\n"
        "- Results sorted by overall_data_score (descending)\n"
        "- Then by test_date (newest first) for equal scores\n"
        "- `limit` parameter controls maximum results returned\n\n"
        "**Example Usage:**\n"
        "- `limit=1`: Get single best product\n"
        "- `limit=10`: Get top 10 products\n"
        "- Default `limit=5`: Get top 5 products\n"
    ),
    response_model=StationTopProductsResponseSchema,
)
async def get_station_top_products(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    site_id: str = Query(..., description="Site identifier or name to scope the station lookup (e.g., 2 or PTB)."),
    model_id: str = Query(..., description="Model identifier or name to scope the station lookup (e.g., 44 or HH5K)."),
    start_time: str = Query(..., description="Inclusive start of the evaluation window (ISO format, e.g., 2023-01-01T00:00:00Z)."),
    end_time: str = Query(..., description="Inclusive end of the evaluation window (ISO format, max 7 days from start_time, e.g., 2023-01-07T23:59:59Z)."),
    criteria_score: str = Query(..., description="Overall score threshold or range (e.g., '8.5' or '8-10')."),
    limit: int = Query(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of top products to return based on highest overall_data_score (default: 5, max: 100).",
    ),
    device_identifiers: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of device identifiers (IDs or names) to include (e.g., 1351 or 614670).",
    ),
    test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional regex patterns to include specific measurement rows.",
    ),
    exclude_test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional regex patterns to exclude measurement rows.",
    ),
    criteria_file: Annotated[UploadFile | None, File(description="Optional uploaded criteria configuration file.")] = None,
    client: DUTAPIClient = dut_client_dependency,
):
    try:
        start_dt = _parse_input_datetime(start_time)
        end_dt = _parse_input_datetime(end_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _validate_time_window(start_dt, end_dt)

    # Validate that the time window is not longer than 7 days
    time_delta = end_dt - start_dt
    if time_delta.days > 7:
        raise HTTPException(
            status_code=400,
            detail=f"Time window cannot exceed 7 days. Requested window: {time_delta.days} days.",
        )

    min_score, max_score = _parse_score_criteria(criteria_score)

    resolved_site_id = await _resolve_site_id(client, site_id)
    resolved_model_id = await _resolve_model_id(client, model_id, site_hint=site_id)
    resolved_station_id = await _resolve_station_id(client, station_id, model_hint=model_id, site_hint=site_id)

    metadata = await _get_station_metadata(client, resolved_station_id)
    meta_site_id = metadata.get("site_id")
    if meta_site_id is not None and meta_site_id != resolved_site_id:
        raise HTTPException(status_code=404, detail="Station not associated with the requested site")
    meta_model_id = metadata.get("model_id")
    if meta_model_id is not None and meta_model_id != resolved_model_id:
        raise HTTPException(status_code=404, detail="Station not associated with the requested model")

    station_name = metadata.get("station_name") or str(resolved_station_id)
    site_name = metadata.get("site_name") or str(resolved_site_id)
    model_name = metadata.get("model_name") or str(resolved_model_id)
    station_status = metadata.get("station_status")
    station_order = metadata.get("station_order")
    fallback_model_hint = None
    if isinstance(metadata.get("model_name"), str) and metadata.get("model_name").strip():
        fallback_model_hint = _normalize_optional_string(metadata.get("model_name"))
    elif isinstance(model_id, str) and model_id.strip():
        fallback_model_hint = _normalize_optional_string(model_id)

    device_filters: set[str] | None = None
    if device_identifiers:
        normalized_devices = {_normalize_str(value) for value in device_identifiers if value}
        if normalized_devices:
            device_filters = normalized_devices

    criteria_rules: dict[str, list[CriteriaRule]] | None = None
    if criteria_file is not None:
        content = await criteria_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded criteria file is empty")
        criteria_rules = _load_station_criteria_from_bytes(content)

    normalized_station_key = _normalize_str(station_name)
    default_rules = [CriteriaRule(pattern=re.compile(".*"), usl=None, lsl=None, target=None)]
    if criteria_rules is None:
        criteria_rules = {normalized_station_key: default_rules}
    else:
        station_specific_rules = _select_station_criteria(criteria_rules, station_name, model_name)
        if not station_specific_rules:
            criteria_rules.setdefault(normalized_station_key, default_rules)

    include_patterns, include_tokens, include_signatures, include_is_exact = _compile_test_item_patterns(test_item_filters)
    exclude_patterns, exclude_tokens, exclude_signatures, exclude_is_exact = _compile_test_item_patterns(exclude_test_item_filters)

    request_start = start_dt.astimezone(UTC)
    request_end = end_dt.astimezone(UTC)

    device_entries: list[dict[str, Any]] = []
    try:
        device_entries = await client.get_devices_by_period(resolved_station_id, request_start, request_end, "ALL")
        if not isinstance(device_entries, list):
            device_entries = []
    except httpx.HTTPStatusError as exc:
        detail = _extract_upstream_error_detail(exc.response)
        status_code = exc.response.status_code if exc.response is not None else 502
        logger.warning(
            "Primary device-period lookup failed for station %s (%s): status=%s detail=%s, will try fallback",
            station_id,
            resolved_station_id,
            status_code,
            detail,
        )
        # Don't raise immediately, let fallback handle it
        device_entries = []
    except Exception as exc:
        logger.warning(
            "Error fetching devices by period for station %s (%s): %s, will try fallback",
            station_id,
            resolved_station_id,
            exc,
        )
        device_entries = []

    enriched_entries = _enrich_devices_with_context(device_entries)
    if not enriched_entries:
        logger.info(
            "Primary lookup returned no results for station %s (%s), attempting fallback device search",
            station_name,
            resolved_station_id,
        )
        fallback_entries = await _fallback_station_device_results(
            client,
            resolved_station_id,
            station_name,
            start_dt,
            end_dt,
            device_filters,
            fallback_model_hint,
        )
        enriched_entries = _enrich_devices_with_context(fallback_entries)
        if enriched_entries:
            logger.info(
                "Fallback search found %d entries for station %s (%s)",
                len(enriched_entries),
                station_name,
                resolved_station_id,
            )
    grouped_runs: dict[tuple[int | None, str], list[dict]] = defaultdict(list)
    start_dt_naive = start_dt.astimezone(UTC).replace(tzinfo=None)
    end_dt_naive = end_dt.astimezone(UTC).replace(tzinfo=None)

    for entry in enriched_entries:
        test_date = _parse_test_date(entry.get("test_date"))
        if test_date == datetime.min:
            continue
        if test_date < start_dt_naive or test_date > end_dt_naive:
            continue
        device_id = entry.get("device_id") or entry.get("id")
        device_name = entry.get("device_name") or entry.get("device_id__name") or entry.get("device")
        if device_filters and not _matches_device_filters(device_filters, device_id, device_name):
            continue
        key_isn = _normalize_str(entry.get("dut_id__isn") or entry.get("dut_isn") or entry.get("isn"))
        key = (_coerce_optional_int(entry.get("dut_id")), key_isn)
        grouped_runs[key].append(entry)

    if not grouped_runs:
        raise HTTPException(status_code=404, detail="No DUT runs matched the requested filter criteria")

    station_rules = _select_station_criteria(criteria_rules, station_name, model_name)
    if not station_rules:
        station_rules = default_rules

    station_payload_cache: dict[tuple[int, int], StationRecordResponseSchema] = {}
    detail_rows: list[StationTopProductDetailSchema] = []

    for (raw_dut_id, _), runs in grouped_runs.items():
        runs.sort(key=lambda entry: _parse_test_date(entry.get("test_date")) or datetime.min)
        latest_entry = runs[-1]
        resolved_dut_id = _coerce_optional_int(latest_entry.get("dut_id") or raw_dut_id)
        if resolved_dut_id is None:
            dut_identifier = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn")
            if dut_identifier:
                try:
                    _, resolved_candidate = await _resolve_station_and_dut_ids(
                        client,
                        str(resolved_station_id),
                        str(dut_identifier),
                        site_hint=site_id,
                        model_hint=model_id,
                    )
                except HTTPException:
                    continue
                resolved_dut_id = resolved_candidate
            if resolved_dut_id is None:
                continue

        cache_key = (resolved_station_id, resolved_dut_id)
        station_payload = station_payload_cache.get(cache_key)
        if station_payload is None:
            try:
                payload = await client.get_station_records(resolved_station_id, resolved_dut_id)
                station_payload = StationRecordResponseSchema.model_validate(payload)
                station_payload_cache[cache_key] = station_payload
            except Exception as exc:
                logger.warning(
                    "Unable to retrieve station records for station %s (%s) and dut %s: %s",
                    station_name,
                    resolved_station_id,
                    resolved_dut_id,
                    exc,
                )
                continue

        run_index = _find_run_index(station_payload, latest_entry)
        measurement_matrix = _compress_station_data(station_payload, run_index)

        # FEATURE: Fetch PA trend measurements and merge into measurement matrix
        # Compares PA trend (historical average) with PA nonvalue (current state)
        # Scoring: closer values = higher scores, using formula |min/max| * 10
        try:
            entry_test_date = _parse_test_date(latest_entry.get("test_date"))
            entry_dut_isn = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn") or latest_entry.get("isn") or ""
            pa_measurements = await _fetch_pa_trend_measurements(
                client=client,
                station_id=resolved_station_id,
                dut_id=resolved_dut_id,
                dut_isn=entry_dut_isn,
                test_date=entry_test_date if entry_test_date != datetime.min else None,
                site_hint=site_id,
                model_hint=model_id,
            )
            if pa_measurements:
                logger.debug(
                    "Adding %d PA trend measurements to station %s (%s) for top products",
                    len(pa_measurements),
                    station_name,
                    resolved_station_id,
                )
                # Extend measurement matrix with PA items
                # Format: [test_item, null, null, nonvalue_actual, trend_target, score]
                measurement_matrix.extend(pa_measurements)
        except Exception as exc:
            # Non-critical: If PA measurements fail, continue with regular measurements
            logger.warning(
                "Failed to fetch PA trend measurements for station %s (%s): %s",
                station_name,
                resolved_station_id,
                exc,
            )

        measurement_rows: list[list[str | float | None]] = []
        score_values: list[float] = []

        for idx, row in enumerate(measurement_matrix):
            if not isinstance(row, list) or len(row) < 4:
                continue
            test_item = str(row[0])
            usl = _to_float(row[1]) if len(row) > 1 else None
            lsl = _to_float(row[2]) if len(row) > 2 else None
            original_row = station_payload.data[idx] if idx < len(station_payload.data) else row
            actual_override, target_override, score_override = _extract_measurement_overrides(original_row, run_index)
            actual = actual_override if actual_override is not None else (_to_float(row[3]) if len(row) > 3 else None)
            existing_score = score_override if score_override is not None else (_to_float(row[5]) if len(row) > 5 else None)
            if actual is None:
                continue
            if (include_patterns or include_tokens or include_signatures) and not _pattern_matches(test_item, include_patterns, include_tokens, include_signatures, include_is_exact):
                continue
            if (exclude_patterns or exclude_tokens or exclude_signatures) and _pattern_matches(test_item, exclude_patterns, exclude_tokens, exclude_signatures, exclude_is_exact):
                continue

            rule = _match_station_rule(station_rules, test_item)

            # FEATURE: Exempt PA adjusted power and POW_DIF_ABS measurements from criteria rule requirement
            # PA items are virtual test items that won't have criteria rules, but should always be included
            measurement_category = _detect_measurement_category(test_item)
            is_pa_adjusted_power = measurement_category == "PA_ADJUSTED_POWER"
            is_pow_dif_abs = measurement_category == "PA_POW_DIF_ABS"

            if rule is None and criteria_rules and not (is_pa_adjusted_power or is_pow_dif_abs):
                continue
            if rule is None:
                rule = station_rules[0]

            target = target_override if target_override is not None else _determine_target_value(rule, usl, lsl, actual, test_item)
            display_usl = rule.usl if rule and rule.usl is not None else usl
            display_lsl = rule.lsl if rule.lsl is not None else lsl
            # UPDATED: Pass test_item for category-specific scoring (EVM, PER, POW_DIF_ABS)
            deviation, computed_score, breakdown = _calculate_measurement_metrics(display_usl, display_lsl, target, actual, test_item)
            # FEATURE: Always compute score for POW_DIF_ABS items (don't use existing_score from external API)
            # POW_DIF_ABS items from external API have score=0, but we need to calculate the actual score
            score_value = computed_score if is_pow_dif_abs else (existing_score if existing_score is not None else computed_score)

            # Ensure target_used is in breakdown
            if breakdown and "target_used" not in breakdown:
                breakdown["target_used"] = target

            measurement_rows.append({"test_item": test_item, "usl": display_usl, "lsl": display_lsl, "actual": actual, "score_breakdown": breakdown})
            score_values.append(score_value)

        if not score_values:
            continue

        overall_score = round(sum(score_values) / len(score_values), 2)
        if overall_score < min_score:
            continue
        if max_score is not None and overall_score > max_score:
            continue

        pass_count = sum(1 for run in runs if _coerce_optional_int(run.get("test_result")) == 1)
        fail_count = sum(1 for run in runs if _coerce_optional_int(run.get("test_result")) == 0)
        test_count = len(runs)

        test_date_value = _parse_test_date(latest_entry.get("test_date"))
        test_date_aware = None if test_date_value == datetime.min else test_date_value.replace(tzinfo=UTC)

        device_id_val = _coerce_optional_int(latest_entry.get("device_id") or latest_entry.get("id"))
        device_name_val = latest_entry.get("device_name") or latest_entry.get("device_id__name") or latest_entry.get("device")
        device_status_val = _coerce_optional_int(latest_entry.get("status"))
        if device_status_val is None:
            device_status_val = _coerce_optional_int(latest_entry.get("test_result"))

        detail_rows.append(
            StationTopProductDetailSchema(
                test_date=test_date_aware,
                station_id=resolved_station_id,
                station_name=station_name,
                station_status=_coerce_optional_int(station_status),
                station_order=_coerce_optional_int(station_order),
                device_id=device_id_val,
                device=device_name_val,
                device_status=device_status_val,
                dut_id=resolved_dut_id,
                isn=latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn"),
                test_count=test_count,
                pass_count=pass_count,
                fail_count=fail_count,
                test_duration=_to_float(latest_entry.get("test_duration")),
                error_item=latest_entry.get("error_item"),
                latest_data=measurement_rows,
                overall_data_score=overall_score,
                metadata={"measurement_count": len(measurement_rows)},
            )
        )

    if not detail_rows:
        raise HTTPException(status_code=404, detail="No DUTs satisfied the supplied criteria")

    # Sort by overall_data_score (descending) and test_date (newest first)
    detail_rows.sort(
        key=lambda item: (
            -item.overall_data_score,
            datetime.min.replace(tzinfo=UTC) if item.test_date is None else item.test_date,
        )
    )

    # Apply limit to return only top N products
    limited_results = detail_rows[:limit]

    return StationTopProductsResponseSchema(
        site_name=site_name,
        model_name=model_name,
        start_time=start_dt,
        end_time=end_dt,
        criteria_score=criteria_score,
        requested_data=limited_results,
    )


@router.post(
    "/top-product",
    tags=["DUT_Management"],
    summary="Identify the top DUT across selected stations based on criteria targets",
    response_model=TopProductBatchResponseSchema,
    response_model_exclude_none=True,
)
@cache(expire=120)  # Cache for 2 minutes
async def get_top_product(
    dut_isns: list[str] = Query(
        ...,
        alias="dut_isn",
        description="One or more DUT ISN identifiers used to seed the evaluation (e.g., 260884980003907).",
    ),
    stations: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of station identifiers (IDs or names) to evaluate (e.g., 144 or Wireless_Test_6G).",
    ),
    site_identifier: str | None = Query(  # noqa: B008
        default=None,
        description="Optional site identifier (ID or name) to narrow DUT lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(  # noqa: B008
        default=None,
        description="Optional model identifier (ID or name) to narrow DUT lookup (e.g., 44 or HH5K).",
    ),
    device_identifiers: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of device identifiers (IDs or names) to focus evaluation (e.g., 1351 or 614670).",
    ),
    test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of regex patterns to restrict measurement rows (e.g., WiFi_TX_POW_6185_11AX_MCS11_B160).",
    ),
    exclude_test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of regex patterns to exclude from scoring (e.g., WiFi_PA_POW_OLD_6985_11AX_MCS9_B160).",
    ),
    criteria_file: Annotated[UploadFile | None, File(description="Optional uploaded criteria configuration file.")] = None,
    station_filters_json: str | None = Query(  # noqa: B008
        default=None,
        alias="station_filters",
        description='Optional JSON string mapping station identifiers to filter configs. Format: {"station_id": {"device_identifiers": [...], "test_item_filters": [...], "exclude_test_item_filters": [...]}}',
    ),
    client: DUTAPIClient = dut_client_dependency,
    db: Session = dut_db_dependency,
):
    # UPDATED: Parse station_filters from JSON string
    station_filters_map: dict[str, StationFilterConfigSchema] | None = None
    if station_filters_json:
        try:
            import json

            parsed = json.loads(station_filters_json)
            station_filters_map = {k: StationFilterConfigSchema(**v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid station_filters JSON: {exc}") from exc

    criteria_rules, criteria_label = await _load_criteria_rules(criteria_file)
    return await _generate_top_product_batch(
        dut_isns,
        stations=stations,
        site_identifier=site_identifier,
        model_identifier=model_identifier,
        device_identifiers=device_identifiers,
        test_item_filters=test_item_filters,
        exclude_test_item_filters=exclude_test_item_filters,
        station_filters_map=station_filters_map,
        criteria_rules=criteria_rules,
        criteria_label=criteria_label,
        client=client,
        db=db,
    )


@router.post(
    "/top-product/with-pa-trends",
    tags=["DUT_Management"],
    summary="Identify the top DUT across selected stations WITH PA trend data",
    response_model=TopProductBatchResponseSchema,
    response_model_exclude_none=True,
)
@cache(expire=300)  # Cache for 5 minutes (parallel processing + caching improves performance)
async def get_top_product_with_pa_trends(
    dut_isns: list[str] = Query(
        ...,
        alias="dut_isn",
        description="One or more DUT ISN identifiers used to seed the evaluation (e.g., 260884980003907).",
    ),
    stations: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of station identifiers (IDs or names) to evaluate (e.g., 144 or Wireless_Test_6G).",
    ),
    site_identifier: str | None = Query(  # noqa: B008
        default=None,
        description="Optional site identifier (ID or name) to narrow DUT lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(  # noqa: B008
        default=None,
        description="Optional model identifier (ID or name) to narrow DUT lookup (e.g., 44 or HH5K).",
    ),
    device_identifiers: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of device identifiers (IDs or names) to focus evaluation (e.g., 1351 or 614670).",
    ),
    test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of regex patterns to restrict measurement rows (e.g., WiFi_TX_POW_6185_11AX_MCS11_B160).",
    ),
    exclude_test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of regex patterns to exclude from scoring (e.g., WiFi_PA_POW_OLD_6985_11AX_MCS9_B160).",
    ),
    criteria_file: Annotated[UploadFile | None, File(description="Optional uploaded criteria configuration file.")] = None,
    station_filters_json: str | None = Query(  # noqa: B008
        default=None,
        alias="station_filters",
        description='Optional JSON string mapping station identifiers to filter configs. Format: {"station_id": {"device_identifiers": [...], "test_item_filters": [...], "exclude_test_item_filters": [...]}}',
    ),
    client: DUTAPIClient = dut_client_dependency,
    db: Session = dut_db_dependency,
):
    """
    Extended version of /top-product that **includes PA trend measurements**.

    ⚠️ **WARNING**: Including PA trends adds 200-500ms per station with PA test items.
    Use `/top-product` (without PA trends) for faster responses.

    This endpoint fetches and includes virtual PA TREND test items that compare:
    - Historical PA trend data (average over time window)
    - Current PA nonvalue measurements
    - Scores based on proximity: closer values = higher scores
    """
    station_filters_map: dict[str, StationFilterConfigSchema] | None = None
    if station_filters_json:
        try:
            import json

            parsed = json.loads(station_filters_json)
            station_filters_map = {k: StationFilterConfigSchema(**v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid station_filters JSON: {exc}") from exc

    criteria_rules, criteria_label = await _load_criteria_rules(criteria_file)
    return await _generate_top_product_batch(
        dut_isns,
        stations=stations,
        site_identifier=site_identifier,
        model_identifier=model_identifier,
        device_identifiers=device_identifiers,
        test_item_filters=test_item_filters,
        exclude_test_item_filters=exclude_test_item_filters,
        station_filters_map=station_filters_map,
        criteria_rules=criteria_rules,
        criteria_label=criteria_label,
        client=client,
        db=db,
        include_pa_trends=True,
    )


@router.post(
    "/top-product/hierarchical",
    tags=["DUT_Management"],
    summary="Identify the top DUT across selected stations with hierarchical scoring",
    response_model=TopProductBatchResponseSchema,
)
@cache(expire=180)  # Cache for 3 minutes (more expensive computation)
async def get_top_product_hierarchical(
    dut_isns: list[str] = Query(
        ...,
        alias="dut_isn",
        description="One or more DUT ISN identifiers used to seed the evaluation (e.g., 260884980003907).",
    ),
    stations: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of station identifiers (IDs or names) to evaluate (e.g., 144 or Wireless_Test_6G).",
    ),
    site_identifier: str | None = Query(  # noqa: B008
        default=None,
        description="Optional site identifier (ID or name) to narrow DUT lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(  # noqa: B008
        default=None,
        description="Optional model identifier (ID or name) to narrow DUT lookup (e.g., 44 or HH5K).",
    ),
    device_identifiers: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of device identifiers (IDs or names) to focus evaluation (e.g., 1351 or 614670).",
    ),
    test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of regex patterns to restrict measurement rows (e.g., WiFi_TX_POW_6185_11AX_MCS11_B160).",
    ),
    exclude_test_item_filters: list[str] | None = Query(  # noqa: B008
        default=None,
        description="Optional list of regex patterns to exclude from scoring (e.g., WiFi_PA_POW_OLD_6985_11AX_MCS9_B160).",
    ),
    criteria_file: Annotated[UploadFile | None, File(description="Optional uploaded criteria configuration file.")] = None,
    station_filters_json: str | None = Query(  # noqa: B008
        default=None,
        alias="station_filters",
        description='Optional JSON string mapping station identifiers to filter configs. Format: {"station_id": {"device_identifiers": [...], "test_item_filters": [...], "exclude_test_item_filters": [...]}}',
    ),
    client: DUTAPIClient = dut_client_dependency,
    db: Session = dut_db_dependency,
):
    # UPDATED: Parse station_filters from JSON string
    station_filters_map: dict[str, StationFilterConfigSchema] | None = None
    if station_filters_json:
        try:
            import json

            parsed = json.loads(station_filters_json)
            station_filters_map = {k: StationFilterConfigSchema(**v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid station_filters JSON: {exc}") from exc

    criteria_rules, criteria_label = await _load_criteria_rules(criteria_file)
    return await _generate_top_product_batch(
        dut_isns,
        stations=stations,
        site_identifier=site_identifier,
        model_identifier=model_identifier,
        device_identifiers=device_identifiers,
        test_item_filters=test_item_filters,
        exclude_test_item_filters=exclude_test_item_filters,
        station_filters_map=station_filters_map,
        criteria_rules=criteria_rules,
        criteria_label=criteria_label,
        client=client,
        db=db,
        postprocessor=_apply_hierarchical_scoring,
    )


@router.get(
    "/records/{dut_id}",
    tags=["DUT_Management"],
    summary="Get DUT test records",
    response_model=dict,
    responses={200: {"description": "DUT record data"}},
)
@cache(expire=60)
async def get_dut_records(
    dut_id: str = Path(..., description="DUT ID or ISN identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the returned records (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the returned records (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get DUT records by DUT ID.

    Args:
        dut_id: DUT identifier

    Returns:
        DUT record data
    """
    try:
        site_hint = _select_identifier(site_identifier, label="site")
        model_hint = _select_identifier(model_identifier, label="model")
        records = await client.get_dut_records(dut_id)
        filtered_records = _filter_record_payload(records, site_hint, model_hint)
        return filtered_records
    except Exception as e:
        logger.error(f"Error fetching DUT records for {dut_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/records/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get station-level DUT records",
    response_model=StationRecordResponseSchema,
)
@cache(expire=60)
async def get_station_records(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_station_records(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error("Error fetching station records for station %s and dut %s: %s", station_id, dut_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc


@router.get(
    "/records/latest/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get latest station-level DUT record",
    response_model=StationRecordResponseSchema,
)
@cache(expire=30)
async def get_latest_station_record(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_station_records(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching station records for latest result station %s dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        validated = StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc

    latest_record = []
    trimmed_data = []
    if validated.record:
        latest_entry = None
        latest_date = datetime.min
        for row in validated.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]
        for row in validated.data:
            if len(row) >= 4:
                trimmed_data.append(row[:3] + [row[-1]])
            else:
                trimmed_data.append(row)
    else:
        trimmed_data = validated.data
    return StationRecordResponseSchema(record=latest_record, data=trimmed_data)


@router.get(
    "/records/nonvalue/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get station non-value test records",
    response_model=StationRecordResponseSchema,
)
@cache(expire=60)
async def get_station_nonvalue_records(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_station_nonvalue_records(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching nonvalue station records for station %s and dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for nonvalue station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc


@router.get(
    "/records/nonvalue/latest/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get latest station non-value test record",
    response_model=StationRecordResponseSchema,
)
@cache(expire=30)
async def get_latest_station_nonvalue_record(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching latest nonvalue station records for station %s and dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        validated = StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for latest nonvalue station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc

    # Extract only the latest record and trim data to last value per test item
    latest_record = []
    trimmed_data = []
    if validated.record:
        latest_entry = None
        latest_date = datetime.min
        for row in validated.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]
        # Trim data array to show only last value per test item (index 0: name, index -1: last value)
        for row in validated.data:
            if len(row) >= 2:
                trimmed_data.append([row[0], row[-1]])
            else:
                trimmed_data.append(row)
    else:
        trimmed_data = validated.data
    return StationRecordResponseSchema(record=latest_record, data=trimmed_data)


@router.get(
    "/records/nonvalue2/latest/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get latest PA SROM test records with hex, decimal, and adjusted power data",
    response_model=PASROMEnhancedResponseSchema,
)
@cache(expire=30)
async def get_latest_pa_srom_enhanced_record(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get latest station non-value test record for PA SROM items (PA SROM_OLD and PA SROM_NEW).

    Returns structured data with:
    - Record metadata (test_date, device, station, etc.)
    - PA SROM_OLD and SROM_NEW items with hex measurements and decimal values
    - PA ADJUSTED_POW calculated from paired SROM items: (SROM_NEW - SROM_OLD) / 256

    Data is sorted like Test_Log_Processing endpoints:
    1. By frequency (ascending: 2412, 5985, 6015, etc.)
    2. By PA antenna number (PA1, PA2, PA3, PA4)
    3. By type (SROM_OLD, SROM_NEW, ADJUSTED_POW)
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching latest nonvalue station records for station %s and dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        validated = StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for latest nonvalue station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc

    # Extract only the latest record
    latest_record = []
    if validated.record:
        latest_entry = None
        latest_date = datetime.min
        for row in validated.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]

    # Process PA SROM data with pairing and adjusted power calculation
    processed_data, has_pa_srom = _process_pa_srom_enhanced(validated)

    return PASROMEnhancedResponseSchema(record=latest_record, data=processed_data, has_pa_srom=has_pa_srom)


@router.get(
    "/records/nonvalue2/scored/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get scored PA SROM adjusted power by comparing current values against trend (24h mean/median)",
    response_model=PAAdjustedPowerScoringResponseSchema,
)
@cache(expire=30)
async def get_scored_pa_adjusted_power(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    start_time: datetime | None = Query(
        None,
        description="Optional start time for trend calculation (ISO format). If not provided, uses 24h before latest test.",
    ),
    end_time: datetime | None = Query(
        None,
        description="Optional end time for trend calculation (ISO format). If not provided, uses latest test time.",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get scored PA adjusted power values by comparing current test against historical trend.

    This endpoint:
    1. Fetches current PA SROM data and calculates adjusted power: (SROM_NEW - SROM_OLD) / 256
    2. Fetches PA trend data (24h mean and median) for the same test items
    3. Calculates deviation of current values from trend mean
    4. Scores each item (0-10 scale) based on deviation from trend mean
    5. Returns scored results with detailed breakdown

    **Scoring Logic:**
    - Compares current adjusted power against 24-hour trend mean
    - Score = 10.0 * (1 - |deviation| / threshold)
    - Default threshold: 5.0 (configurable)
    - Lower deviation = higher score (10 is perfect match with trend)

    **Use Cases:**
    - Identify DUTs with abnormal PA calibration
    - Quality control for production testing
    - Detect calibration drift over time
    """
    # Step 1: Get current adjusted power values
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
    )

    try:
        payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching latest nonvalue station records for station %s and dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        validated = StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for latest nonvalue station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc

    # Extract latest record metadata
    latest_record = []
    test_date_for_trend = None
    dut_isn_for_trend = None

    if validated.record:
        latest_entry = None
        latest_date = datetime.min
        for row in validated.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]
            test_date_for_trend = latest_entry.test_date
            dut_isn_for_trend = latest_entry.isn or dut_id

    # Process current PA SROM data
    current_data, has_pa_srom = _process_pa_srom_enhanced(validated)

    if not has_pa_srom:
        raise HTTPException(status_code=404, detail="No PA SROM test items found for this DUT. Scoring requires PA SROM data.")

    # Extract current adjusted power values
    current_adjusted_power = {}  # {test_item: value}
    for item in current_data:
        if isinstance(item, PAAdjustedPowerDataItemSchema):
            current_adjusted_power[item.test_item] = item.adjusted_value

    if not current_adjusted_power:
        raise HTTPException(status_code=404, detail="No PA adjusted power values calculated. Need matching SROM_OLD and SROM_NEW pairs.")

    # Step 2: Determine time window for trend calculation (following pa/trend/decimal pattern)
    parsed_test_date = _parse_test_date(test_date_for_trend) if test_date_for_trend else None

    if start_time is None or end_time is None:
        # Auto-calculate time window based on test_date
        if parsed_test_date:
            if parsed_test_date.tzinfo is None:
                window_end = parsed_test_date.replace(tzinfo=UTC)
            else:
                window_end = parsed_test_date.astimezone(UTC)
        else:
            window_end = datetime.now(UTC)

        window_start = window_end - timedelta(hours=24)
        actual_start = start_time or window_start
        actual_end = end_time or window_end
    else:
        # Use provided time window
        actual_start = start_time.replace(tzinfo=UTC) if start_time.tzinfo is None else start_time
        actual_end = end_time.replace(tzinfo=UTC) if end_time.tzinfo is None else end_time

        time_diff = actual_end - actual_start
        if time_diff > timedelta(days=7):
            raise HTTPException(status_code=400, detail="Time window exceeds 7 days")
        if time_diff.total_seconds() < 0:
            raise HTTPException(status_code=400, detail="end_time must be after start_time")

    time_window = {"start_time": actual_start.isoformat(), "end_time": actual_end.isoformat(), "duration_hours": (actual_end - actual_start).total_seconds() / 3600}

    logger.info(f"[PA Scoring] Time window: {time_window}")

    # Step 3: Fetch PA trend data (following pa/trend/decimal pattern)
    try:
        # Extract PA SROM test items for trend API from the already-fetched nonvalue data
        pa_srom_test_items = []
        for row in validated.data:
            if not isinstance(row, list) or len(row) < 1:
                continue
            test_item_name = str(row[0])
            if _is_pa_srom_test_item(test_item_name, "all"):
                pa_srom_test_items.append(test_item_name)

        if not pa_srom_test_items:
            raise HTTPException(status_code=404, detail="No PA SROM test items found for trend calculation")

        logger.info(f"[PA Scoring] Found {len(pa_srom_test_items)} PA SROM test items")

        # Build PA trend API payload (matching pa/trend/decimal format exactly)
        start_time_str = actual_start.isoformat().replace("+00:00", "Z")
        end_time_str = actual_end.isoformat().replace("+00:00", "Z")

        trend_payload = {
            "start_time": start_time_str,
            "end_time": end_time_str,
            "station_id": resolved_station_id,
            "test_items": pa_srom_test_items,
            "model": "",  # Empty string as per PA trend API requirements
        }

        logger.info(f"[PA Scoring] Calling PA trend API with payload: {trend_payload}")

        # Call PA trend API
        trend_data = await client.get_pa_test_items_trend(trend_payload)

        logger.info(f"[PA Scoring] PA trend API returned {len(trend_data)} items")

    except Exception as exc:
        logger.error("Error fetching PA trend data: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch PA trend data: {str(exc)}") from exc

    # Step 4: Parse trend data and calculate adjusted power from trend
    trend_adjusted_power = {}  # {adjusted_pow_test_item: {"mean": float, "mid": float}}

    # Pair SROM_OLD and SROM_NEW from trend data
    old_trend = {}
    new_trend = {}

    for test_item_name, trend_values in trend_data.items():
        if _is_pa_srom_test_item(test_item_name, "old"):
            base_key = re.sub(r"_SROM_OLD_", "_SROM_", test_item_name, flags=re.IGNORECASE)
            old_trend[base_key] = trend_values
        elif _is_pa_srom_test_item(test_item_name, "new"):
            base_key = re.sub(r"_SROM_NEW_", "_SROM_", test_item_name, flags=re.IGNORECASE)
            new_trend[base_key] = trend_values

    # Calculate adjusted power from trend pairs
    for base_key in set(old_trend.keys()) & set(new_trend.keys()):
        old_vals = old_trend[base_key]
        new_vals = new_trend[base_key]

        adjusted_item_name = re.sub(r"_SROM_", "_ADJUSTED_POW_", base_key, flags=re.IGNORECASE)

        mean_old = old_vals.get("mean")
        mean_new = new_vals.get("mean")
        mid_old = old_vals.get("mid")
        mid_new = new_vals.get("mid")

        trend_adjusted_power[adjusted_item_name] = {
            "mean": round((mean_new - mean_old) / 256, 2) if mean_old is not None and mean_new is not None else None,
            "mid": round((mid_new - mid_old) / 256, 2) if mid_old is not None and mid_new is not None else None,
        }

    # Step 5: Calculate scores by comparing current vs trend
    scored_items = []
    scores_for_avg = []

    for test_item, current_value in current_adjusted_power.items():
        trend_vals = trend_adjusted_power.get(test_item, {})
        trend_mean = trend_vals.get("mean")
        trend_mid = trend_vals.get("mid")

        # Calculate deviations
        deviation_from_mean = (current_value - trend_mean) if trend_mean is not None else None
        deviation_from_mid = (current_value - trend_mid) if trend_mid is not None else None

        # Calculate score based on how close current value is to trend mean
        # Closer to trend = higher score (10 is perfect match)
        score = None
        score_breakdown = {}

        if deviation_from_mean is not None:
            # Calculate score: closer to trend mean = higher score
            # Using linear scoring with threshold of 5.0 dB
            # Score = 10 * (1 - |deviation| / threshold)
            threshold = 5.0
            abs_deviation = abs(deviation_from_mean)

            if abs_deviation >= threshold:
                score = 0.0
            else:
                score = 10.0 * (1.0 - abs_deviation / threshold)
                score = max(0.0, min(10.0, score))

            scores_for_avg.append(score)

            # Create detailed breakdown
            score_breakdown = {
                "category": "PA Adjusted Power Trend Comparison",
                "method": "Linear scoring based on deviation from trend mean",
                "comparison": "current vs trend_mean",
                "threshold": threshold,
                "target_used": trend_mean,
                "current_value": round(current_value, 2),
                "trend_mean": round(trend_mean, 2) if trend_mean is not None else None,
                "deviation_from_mean": round(deviation_from_mean, 2),
                "abs_deviation": round(abs_deviation, 2),
                "raw_score": round(score, 2),
                "final_score": round(score, 2),
                "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{current} - \text{trend}|}{\text{threshold}}\right)$",
                "interpretation": (
                    "Perfect match (0.0 dev)"
                    if abs_deviation < 0.1
                    else "Excellent match (<1.0 dev)"
                    if abs_deviation < 1.0
                    else "Good match (<2.5 dev)"
                    if abs_deviation < 2.5
                    else "Acceptable (<5.0 dev)"
                    if abs_deviation < 5.0
                    else "Out of range (≥5.0 dev)"
                ),
            }

            score = round(score, 2)

        scored_items.append(
            PAAdjustedPowerScoredItemSchema(
                test_item=test_item, current_value=current_value, trend_mean=trend_mean, trend_mid=trend_mid, deviation_from_mean=deviation_from_mean, deviation_from_mid=deviation_from_mid, score=score, score_breakdown=score_breakdown
            )
        )

    # Calculate aggregate scores
    avg_score = round(sum(scores_for_avg) / len(scores_for_avg), 2) if scores_for_avg else None
    median_score = round(float(median(scores_for_avg)), 2) if scores_for_avg else None

    logger.info(f"[PA Scoring] Calculated {len(scored_items)} scored items, avg_score={avg_score}, median_score={median_score}")

    return PAAdjustedPowerScoringResponseSchema(record=latest_record, data=scored_items, trend_time_window=time_window, avg_score=avg_score, median_score=median_score)


@router.get(
    "/records/nonvalueBin/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get station non-value BIN test records",
    response_model=StationRecordResponseSchema,
)
@cache(expire=60)
async def get_station_nonvalue_bin_records(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_station_nonvalue_bin_records(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching nonvalueBin station records for station %s and dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        return StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for nonvalueBin station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc


@router.get(
    "/records/nonvalueBin/latest/{station_id}/{dut_id}",
    tags=["DUT_Management"],
    summary="Get latest station non-value BIN test record",
    response_model=StationRecordResponseSchema,
)
@cache(expire=30)
async def get_latest_station_nonvalue_bin_record(
    station_id: str = Path(..., description="Station identifier (numeric ID or name) (e.g., 144 or Wireless_Test_6G)"),
    dut_id: str = Path(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
        client,
        station_id,
        dut_id,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    try:
        payload = await client.get_latest_nonvalue_bin_record(resolved_station_id, resolved_dut_id)
    except Exception as exc:
        logger.error(
            "Error fetching latest nonvalueBin station records for station %s and dut %s: %s",
            station_id,
            dut_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        validated = StationRecordResponseSchema.model_validate(payload)
    except ValidationError as exc:
        logger.error("Unexpected payload shape for latest nonvalueBin station records: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc

    # Extract only the latest record and trim data to last value per test item
    latest_record = []
    trimmed_data = []
    if validated.record:
        latest_entry = None
        latest_date = datetime.min
        for row in validated.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]
        # Trim data array to show only last value per test item (index 0: name, index -1: last value)
        for row in validated.data:
            if len(row) >= 2:
                trimmed_data.append([row[0], row[-1]])
            else:
                trimmed_data.append(row)
    else:
        trimmed_data = validated.data
    return StationRecordResponseSchema(record=latest_record, data=trimmed_data)


def _convert_hex_to_decimal(hex_value: str) -> int | None:
    """
    Convert hexadecimal string to decimal integer.

    Args:
        hex_value: Hexadecimal string (e.g., "0x235e" or "235e")

    Returns:
        Decimal integer or None if conversion fails
    """
    if not hex_value:
        return None

    try:
        # Remove whitespace and convert to string
        hex_str = str(hex_value).strip()

        # Handle empty string
        if not hex_str:
            return None

        # Convert hex to decimal (handles both "0x235e" and "235e" formats)
        if hex_str.lower().startswith("0x"):
            return int(hex_str, 16)
        else:
            # Assume it's hex without 0x prefix
            return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def _is_pa_srom_test_item(test_item_name: str, pattern_type: str = "all") -> bool:
    """
    Check if a test item name matches PA SROM patterns.

    Args:
        test_item_name: Test item name to check
        pattern_type: "old", "new", or "all" (default)

    Returns:
        True if test item matches the specified pattern type
    """
    if not test_item_name:
        return False

    name_upper = test_item_name.upper()

    # Check for PA{1-4}_SROM_OLD or PA{1-4}_SROM_NEW patterns
    has_old = bool(re.search(r"PA[1-4]_SROM_OLD", name_upper))
    has_new = bool(re.search(r"PA[1-4]_SROM_NEW", name_upper))

    if pattern_type == "old":
        return has_old
    elif pattern_type == "new":
        return has_new
    else:  # "all"
        return has_old or has_new


def _get_pa_srom_sort_key(test_item_name: str) -> tuple:
    """
    Extract sort key from PA SROM test item name for ordering.

    Sorts by: 1) Frequency (ascending), 2) Antenna number (PA1-4), 3) SROM type (OLD before NEW), 4) Name

    Args:
        test_item_name: Test item name (e.g., "WiFi_PA2_SROM_NEW_5985_11AX_MCS9_B80")

    Returns:
        Tuple of (frequency, antenna_num, srom_priority, test_item_name)
    """
    name_upper = test_item_name.upper()

    # Extract frequency (e.g., "2412", "5985") - default to 0 if not found
    frequency = 0
    freq_match = re.search(r"_(\d{4})_", test_item_name)
    if freq_match:
        frequency = int(freq_match.group(1))

    # Extract antenna number (PA1, PA2, PA3, PA4) - default to 99 if not found
    antenna_num = 99
    pa_match = re.search(r"PA([1-4])_SROM", name_upper)
    if pa_match:
        antenna_num = int(pa_match.group(1))

    # SROM type priority: OLD (0) before NEW (1)
    srom_priority = 0 if "_SROM_OLD" in name_upper else 1

    return (frequency, antenna_num, srom_priority, test_item_name)


def _process_pa_srom_data(validated: StationRecordResponseSchema, pattern_type: str = "all") -> StationRecordResponseSchema:
    """
    Process PA SROM data by filtering and converting hex to decimal.

    Args:
        validated: Validated station record response
        pattern_type: "old", "new", or "all" to filter PA SROM items

    Returns:
        Processed station record with filtered PA SROM items and converted decimal values
    """
    latest_record = []
    processed_data = []

    if validated.record:
        # Extract only the latest record
        latest_entry = None
        latest_date = datetime.min
        for row in validated.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]

    # Process data array - filter PA SROM items and convert hex to decimal
    for row in validated.data:
        if not isinstance(row, list) or len(row) < 2:
            continue

        test_item_name = str(row[0]) if row else ""

        # Filter by PA SROM pattern type
        if not _is_pa_srom_test_item(test_item_name, pattern_type):
            continue

        # Get the latest value (last column)
        latest_value = row[-1] if len(row) > 1 else None

        # Convert hex to decimal
        decimal_value = _convert_hex_to_decimal(latest_value)

        # Only include if conversion was successful
        if decimal_value is not None:
            processed_data.append([test_item_name, decimal_value])

    return StationRecordResponseSchema(record=latest_record, data=processed_data)


def _process_pa_srom_enhanced(validated: StationRecordResponseSchema) -> tuple[list[PASROMDataItemSchema | PAAdjustedPowerDataItemSchema | GenericTestItemSchema], bool]:
    """
    Process PA SROM data with enhanced structure showing hex, decimal, and adjusted power.

    If DUT has no PA SROM items, falls back to showing all test items like original nonvalue endpoint.

    This function:
    1. Filters for PA SROM_OLD and PA SROM_NEW test items only
    2. Converts hex measurements to decimal
    3. Pairs SROM_OLD and SROM_NEW items
    4. Calculates adjusted power: (SROM_NEW - SROM_OLD) / 256
    5. Sorts by frequency, PA number, and type (OLD, NEW, ADJUSTED_POW)
    6. If no PA SROM items found, returns all test items in generic format

    Args:
        validated: Validated station record response with raw data

    Returns:
        Tuple of (data_items, has_pa_srom)
        - data_items: List of structured data items
        - has_pa_srom: Boolean indicating if PA SROM items were found
    """
    from ..schemas.dut_schemas import GenericTestItemSchema, PAAdjustedPowerDataItemSchema, PASROMDataItemSchema

    # Step 1: Extract PA SROM items with hex-to-decimal conversion
    pa_srom_items = {}  # {test_item_name: {"hex": "0x2c89", "decimal": 11190}}

    for row in validated.data:
        if not isinstance(row, list) or len(row) < 2:
            continue

        test_item_name = str(row[0]) if row else ""

        # Only process PA SROM items
        if not _is_pa_srom_test_item(test_item_name, "all"):
            continue

        # Get the latest value (last column)
        latest_value = row[-1] if len(row) > 1 else None

        # Convert hex to decimal
        decimal_value = _convert_hex_to_decimal(latest_value)

        if decimal_value is not None:
            # Ensure hex format has 0x prefix
            hex_str = str(latest_value).strip()
            if not hex_str.lower().startswith("0x"):
                hex_str = f"0x{hex_str}"

            pa_srom_items[test_item_name] = {"hex": hex_str, "decimal": decimal_value}

    # Step 2: Pair SROM_OLD and SROM_NEW items for adjusted power calculation
    old_items = {}  # {base_key: {"name": ..., "hex": ..., "decimal": ...}}
    new_items = {}  # {base_key: {"name": ..., "hex": ..., "decimal": ...}}

    for test_item_name, values in pa_srom_items.items():
        if _is_pa_srom_test_item(test_item_name, "old"):
            # Extract base key: WiFi_PA1_SROM_OLD_5985... -> WiFi_PA1_SROM_5985...
            base_key = re.sub(r"_SROM_OLD_", "_SROM_", test_item_name, flags=re.IGNORECASE)
            old_items[base_key] = {"name": test_item_name, "hex": values["hex"], "decimal": values["decimal"]}
        elif _is_pa_srom_test_item(test_item_name, "new"):
            # Extract base key: WiFi_PA1_SROM_NEW_5985... -> WiFi_PA1_SROM_5985...
            base_key = re.sub(r"_SROM_NEW_", "_SROM_", test_item_name, flags=re.IGNORECASE)
            new_items[base_key] = {"name": test_item_name, "hex": values["hex"], "decimal": values["decimal"]}

    # Step 3: Create structured data items and calculate adjusted power
    result_items = []

    # Collect all base keys that have matching pairs
    paired_base_keys = set(old_items.keys()) & set(new_items.keys())

    for base_key in paired_base_keys:
        old_item = old_items[base_key]
        new_item = new_items[base_key]

        # Add OLD item
        result_items.append(PASROMDataItemSchema(test_item=old_item["name"], measurement=old_item["hex"], decimal_value=old_item["decimal"]))

        # Add NEW item
        result_items.append(PASROMDataItemSchema(test_item=new_item["name"], measurement=new_item["hex"], decimal_value=new_item["decimal"]))

        # Calculate and add ADJUSTED_POW
        adjusted_value = (new_item["decimal"] - old_item["decimal"]) / 256
        adjusted_value_rounded = round(adjusted_value, 2)

        # Create adjusted item name: WiFi_PA1_ADJUSTED_POW_5985_11AX_MCS9_B80
        adjusted_item_name = re.sub(r"_SROM_", "_ADJUSTED_POW_", base_key, flags=re.IGNORECASE)

        result_items.append(PAAdjustedPowerDataItemSchema(test_item=adjusted_item_name, adjusted_value=adjusted_value_rounded))

    # Step 4: Sort items like Test_Log_Processing endpoints
    # Sort by: 1) Frequency, 2) PA number, 3) Type order (OLD=0, NEW=1, ADJUSTED=2)
    def get_sort_key(item):
        test_item_name = item.test_item
        name_upper = test_item_name.upper()

        # Extract frequency (e.g., "2412", "5985", "6015") - default to 0 if not found
        frequency = 0
        freq_match = re.search(r"_(\d{4})_", test_item_name)
        if freq_match:
            frequency = int(freq_match.group(1))

        # Extract PA antenna number (PA1, PA2, PA3, PA4) - default to 99 if not found
        antenna_num = 99
        pa_match = re.search(r"PA([1-4])_", name_upper)
        if pa_match:
            antenna_num = int(pa_match.group(1))

        # Type priority: SROM_OLD (0), SROM_NEW (1), ADJUSTED_POW (2)
        if "_SROM_OLD" in name_upper:
            type_priority = 0
        elif "_SROM_NEW" in name_upper:
            type_priority = 1
        else:  # ADJUSTED_POW
            type_priority = 2

        return (frequency, antenna_num, type_priority, test_item_name)

    result_items.sort(key=get_sort_key)

    # If no PA SROM items found, fallback to showing all test items (like original nonvalue endpoint)
    if not result_items:
        logger.info("[nonvalue2] No PA SROM items found, returning all test items as fallback")
        generic_items = []
        for row in validated.data:
            if not isinstance(row, list) or len(row) < 2:
                continue
            test_item_name = str(row[0]) if row else ""
            latest_value = str(row[-1]) if len(row) > 1 else ""
            generic_items.append(GenericTestItemSchema(test_item=test_item_name, value=latest_value))
        return (generic_items, False)

    return (result_items, True)


@router.get(
    "/complete/{dut_id}",
    tags=["DUT_Management"],
    summary="Get complete DUT information",
    response_model=CompleteDUTInfoSchema,
    responses={200: {"description": "Combined DUT info (sites, models, stations, records)"}},
)
@cache(expire=60)
async def get_complete_dut_info(
    dut_id: str = Path(..., description="DUT ID or ISN/SSN/MAC identifier (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the result set (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the result set (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get complete DUT information including sites, models, stations, and records.

    This endpoint chains multiple API calls to gather comprehensive information.

    Args:
        dut_id: DUT identifier
    Returns:
        Complete DUT information with enriched data
    """
    try:
        site_hint = _select_identifier(site_identifier, label="site")
        model_hint = _select_identifier(model_identifier, label="model")

        site_filter_name: str | None = None
        if site_hint:
            if site_hint.isdigit():
                resolved_site_id = await _resolve_site_id(client, site_hint)
                try:
                    sites_catalog = await _get_cached_sites(client)
                except Exception as exc:  # pragma: no cover - defensive, rare
                    logger.warning("Unable to load site catalogue while resolving site filter: %s", exc)
                    sites_catalog = []
                site_filter_name = next(
                    (site.get("name") for site in sites_catalog if _identifier_matches(site.get("id"), site_hint) or site.get("id") == resolved_site_id),
                    None,
                )
            else:
                site_filter_name = site_hint

        info = await client.get_complete_dut_info(dut_id, site_filter_name)
        info_data = dict(info)

        if site_hint or model_hint:
            if info_data.get("dut_records"):
                info_data["dut_records"] = _filter_record_payload(info_data["dut_records"], site_hint, model_hint)

            if site_hint:
                info_data["sites"] = [site for site in info_data.get("sites", []) if _identifier_matches(site.get("id"), site_hint) or _identifier_matches(site.get("name"), site_hint)]

            if model_hint:
                info_data["models"] = [model for model in info_data.get("models", []) if _identifier_matches(model.get("id"), model_hint) or _identifier_matches(model.get("name"), model_hint)]

            info_data["stations"] = [station for station in info_data.get("stations", []) if _station_matches_hints(station, site_hint, model_hint)]

        return info_data
    except Exception as e:
        logger.error(f"Error fetching complete DUT info for {dut_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/svn/info",
    tags=["DUT_Management"],
    summary="Retrieve SVN information",
    response_model=list[SVNInfoSchema],
    responses={200: {"description": "List of SVN entries"}},
)
@cache(expire=120)
async def get_svn_info(client: DUTAPIClient = dut_client_dependency):
    """
    Get SVN information.

    Returns:
        List of SVN information
    """
    try:
        svn_info = await client.get_svn_info()
        return svn_info
    except Exception as e:
        logger.error(f"Error fetching SVN info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _status_from_result(result: int | None) -> str:
    if result == 1:
        return "PASS"
    if result == 0:
        return "FAIL"
    return "UNKNOWN"


async def _fetch_dut_records(
    client: DUTAPIClient,
    dut_isn: str,
    *,
    site_hint: str | None = None,
    model_hint: str | None = None,
):
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "dut-records", str(dut_isn))

    async def loader() -> Any:
        return await client.get_dut_records(dut_isn)

    try:
        records = await dut_metadata_cache.get_or_set(cache_key, loader, ttl=RECORD_CACHE_TTL)
    except HTTPException as http_exc:
        # Log and re-raise HTTP exceptions from upstream API
        logger.error(
            "HTTP error fetching DUT records for %s: status=%s, detail=%s",
            dut_isn,
            http_exc.status_code,
            http_exc.detail,
        )
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error fetching DUT records for %s: %s (type: %s, traceback: %s)",
            dut_isn,
            exc,
            type(exc).__name__,
            traceback.format_exc(),
        )
        # Provide more detailed error message including DUT ISN
        error_detail = f"Failed to retrieve DUT records for {dut_isn}: {type(exc).__name__}"
        if hasattr(exc, "response"):
            status_code = getattr(exc.response, "status_code", "unknown")
            error_detail += f" (HTTP {status_code})"
        raise HTTPException(status_code=502, detail=error_detail) from exc

    site_name = None
    model_name = None
    record_data = []

    if isinstance(records, dict):
        site_name = records.get("site_name")
        model_name = records.get("model_name")
        record_data = records.get("record_data") or records.get("record") or []
    elif isinstance(records, list):
        record_data = records

    if site_hint or model_hint:
        filtered_records = [station for station in record_data if _station_matches_hints(station, site_hint, model_hint)]
        if filtered_records:
            record_data = filtered_records
        else:
            record_data = []

    if record_data:
        if site_hint:
            derived_site = _extract_station_attribute(
                record_data[0],
                [
                    "site_name",
                    "device_id__station_id__model_id__site_id__name",
                ],
            )
            if derived_site:
                site_name = derived_site
        if model_hint:
            derived_model = _extract_station_attribute(
                record_data[0],
                [
                    "model_name",
                    "device_id__station_id__model_id__name",
                ],
            )
            if derived_model:
                model_name = derived_model

    return site_name, model_name, record_data


def _normalize_str(value) -> str:
    return str(value).strip().lower() if value is not None else ""


def _identifiers_equal(a: str, b: str) -> bool:
    if str(a).strip() == str(b).strip():
        return True
    a_norm = _normalize_str(a)
    b_norm = _normalize_str(b)
    if a_norm and b_norm and a_norm == b_norm:
        return True
    a_trimmed = str(a).strip()
    b_trimmed = str(b).strip()
    if a_trimmed.isdigit() and b_trimmed.isdigit():
        return int(a_trimmed) == int(b_trimmed)
    return False


def _select_identifier(primary: str | None, secondary: str | None = None, *, label: str) -> str | None:
    if primary and secondary and not _identifiers_equal(primary, secondary):
        raise HTTPException(status_code=400, detail=f"Conflicting {label} parameters provided")
    return primary or secondary


def _identifier_matches(candidate: Any, identifier: str | None) -> bool:
    if identifier is None or candidate is None:
        return False
    candidate_str = str(candidate).strip()
    identifier_str = str(identifier).strip()
    if not candidate_str or not identifier_str:
        return False
    if candidate_str == identifier_str:
        return True
    if candidate_str.isdigit() and identifier_str.isdigit():
        return int(candidate_str) == int(identifier_str)
    return _normalize_str(candidate_str) == _normalize_str(identifier_str)


def _station_matches_hints(station: dict[str, Any] | Any, site_hint: str | None, model_hint: str | None) -> bool:
    station_dict = station if isinstance(station, dict) else {}
    if site_hint and not _station_matches_site(station_dict, site_hint):
        return False
    if model_hint and not _station_matches_model(station_dict, model_hint):
        return False
    return True


def _station_matches_site(station: dict[str, Any], site_hint: str) -> bool:
    candidate_fields = [
        "site_id",
        "site_name",
        "device_id__station_id__model_id__site_id",
        "device_id__station_id__model_id__site_id__name",
    ]
    values = list(_iter_station_values(station, candidate_fields))
    return _matches_hint_with_fallback(values, site_hint)


def _station_matches_model(station: dict[str, Any], model_hint: str) -> bool:
    candidate_fields = [
        "model_id",
        "model_name",
        "device_id__station_id__model_id",
        "device_id__station_id__model_id__name",
    ]
    values = list(_iter_station_values(station, candidate_fields))
    return _matches_hint_with_fallback(values, model_hint)


def _matches_hint_with_fallback(values: list[Any], hint: str) -> bool:
    if not values:
        return True
    hint_str = str(hint).strip()
    hint_is_digit = hint_str.isdigit()
    comparable_seen = False
    for value in values:
        if value in (None, ""):
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        value_is_digit = value_str.isdigit()
        if value_is_digit != hint_is_digit:
            # Skip incomparable types (e.g., numeric id vs textual hint)
            continue
        comparable_seen = True
        if _identifier_matches(value, hint):
            return True
    return not comparable_seen


@dataclass(slots=True)
class _StationMatchResult:
    station_id: int
    station_name: str | None
    station_dut_id: int
    station_dut_isn: str | None
    raw_entry: dict[str, Any]


def _match_station_entry(
    record_data: list[dict[str, Any]],
    station_identifier: str,
    dut_identifier: str,
) -> _StationMatchResult | None:
    station_id_hint = int(station_identifier) if str(station_identifier).strip().isdigit() else None
    dut_id_hint = int(dut_identifier) if str(dut_identifier).strip().isdigit() else None
    normalized_station = _normalize_str(station_identifier)
    normalized_dut = _normalize_str(dut_identifier)

    for station in record_data:
        candidate_station_id = _coerce_optional_int(station.get("id"))
        candidate_station_name = station.get("name")

        station_matches = False
        if station_id_hint is not None and candidate_station_id == station_id_hint:
            station_matches = True
        elif station_id_hint is None and candidate_station_name and _normalize_str(candidate_station_name) == normalized_station:
            station_matches = True
        if not station_matches:
            continue

        candidate_dut_id = _coerce_optional_int(station.get("dut_id"))
        candidate_dut_isn = station.get("dut_isn")
        match_by_isn = bool(candidate_dut_isn and _normalize_str(candidate_dut_isn) == normalized_dut)

        if candidate_dut_id is None:
            for item in station.get("data") or []:
                if candidate_dut_id is None and item.get("dut_id"):
                    candidate_dut_id = _coerce_optional_int(item.get("dut_id"))
                item_isn = item.get("dut_id__isn") or item.get("dut_isn")
                if item_isn and _normalize_str(item_isn) == normalized_dut:
                    match_by_isn = True

        if candidate_dut_id is None:
            continue

        if match_by_isn or dut_id_hint is None or candidate_dut_id == dut_id_hint:
            resolved_station_id = candidate_station_id or station_id_hint
            if resolved_station_id is None:
                continue
            return _StationMatchResult(
                station_id=resolved_station_id,
                station_name=candidate_station_name,
                station_dut_id=candidate_dut_id,
                station_dut_isn=candidate_dut_isn,
                raw_entry=station,
            )

    return None


def _match_station_entry_loose(
    record_data: list[dict[str, Any]],
    station_identifier: str,
) -> _StationMatchResult | None:
    station_id_hint = int(station_identifier) if str(station_identifier).strip().isdigit() else None
    normalized_station = _normalize_str(station_identifier)

    for station in record_data:
        candidate_station_id = _coerce_optional_int(station.get("id"))
        candidate_station_name = station.get("name")

        station_matches = False
        if station_id_hint is not None and candidate_station_id == station_id_hint:
            station_matches = True
        elif station_id_hint is None and candidate_station_name and _normalize_str(candidate_station_name) == normalized_station:
            station_matches = True
        if not station_matches:
            continue

        candidate_dut_id = _coerce_optional_int(station.get("dut_id"))
        candidate_dut_isn = station.get("dut_isn")

        if candidate_dut_id is None or not candidate_dut_isn:
            for item in station.get("data") or []:
                if candidate_dut_id is None and item.get("dut_id"):
                    candidate_dut_id = _coerce_optional_int(item.get("dut_id"))
                if not candidate_dut_isn:
                    candidate_dut_isn = item.get("dut_id__isn") or item.get("dut_isn")
                if candidate_dut_id is not None and candidate_dut_isn:
                    break

        if candidate_dut_id is None:
            continue

        resolved_station_id = candidate_station_id or station_id_hint
        if resolved_station_id is None:
            continue

        return _StationMatchResult(
            station_id=resolved_station_id,
            station_name=candidate_station_name,
            station_dut_id=candidate_dut_id,
            station_dut_isn=candidate_dut_isn,
            raw_entry=station,
        )

    return None


async def _build_latest_test_items_response(
    client: DUTAPIClient,
    request: BatchLatestTestItemsRequestSchema,
) -> BatchLatestTestItemsResponseSchema:
    site_hint = _select_identifier(request.site_identifier, label="site")
    model_hint = _select_identifier(request.model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        request.dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    # If filters removed everything, retry without hints so stations still resolve
    if not record_data and (site_hint or model_hint):
        site_name, model_name, record_data = await _fetch_dut_records(
            client,
            request.dut_isn,
            site_hint=None,
            model_hint=None,
        )

    async def _build_placeholder_entry(identifier: str) -> StationLatestTestItemsSchema | None:
        try:
            resolved_station_id = await _resolve_station_id(
                client,
                identifier,
                model_hint=model_hint,
                site_hint=site_hint,
            )
        except Exception:
            resolved_station_id = _coerce_optional_int(identifier)
            if resolved_station_id is None:
                return None
            station_name = str(identifier)
        else:
            try:
                metadata = await _get_station_metadata(client, resolved_station_id)
            except Exception:
                station_name = str(identifier)
            else:
                station_name = metadata.get("station_name") or str(identifier)

        return StationLatestTestItemsSchema(
            station_id=resolved_station_id,
            station_name=station_name,
            station_dut_id=None,
            station_dut_isn=None,
            value_test_items=[],
            nonvalue_bin_test_items=[],
            nonvalue_test_items=[],
            error="station not associated with the requested DUT",
        )

    stations: list[StationLatestTestItemsSchema] = []
    for station_identifier in request.station_identifiers:
        match = _match_station_entry(record_data, station_identifier, request.dut_isn)
        if not match:
            match = _match_station_entry_loose(record_data, station_identifier)
        if not match:
            placeholder = await _build_placeholder_entry(station_identifier)
            if placeholder is not None:
                stations.append(placeholder)
            continue

        station_id = match.station_id
        station_dut_id = match.station_dut_id
        station_name = match.station_name or str(station_identifier)
        station_dut_isn = match.station_dut_isn or request.dut_isn

        if station_dut_id is None:
            placeholder = await _build_placeholder_entry(station_identifier)
            if placeholder is not None:
                stations.append(placeholder)
            continue

        value_items: list[TestItemDefinitionSchema] = []
        nonvalue_bin_items: list[TestItemDefinitionSchema] = []
        nonvalue_items: list[TestItemDefinitionSchema] = []
        error_parts: list[str] = []

        try:
            raw_value_payload = await client.get_latest_station_records(station_id, station_dut_id)
            value_items = _extract_value_items_from_payload(raw_value_payload)
            if not value_items:
                try:
                    fallback_value_payload = await client.get_station_records(station_id, station_dut_id)
                except Exception as fallback_exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Fallback value record lookup failed for station %s (DUT %s): %s",
                        station_identifier,
                        request.dut_isn,
                        fallback_exc,
                    )
                else:
                    value_items = _extract_value_items_from_payload(fallback_value_payload)
            if not value_items:
                error_parts.append("value items unavailable")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to collect value test items for station %s (DUT %s): %s",
                station_identifier,
                request.dut_isn,
                exc,
            )
            error_parts.append("value items unavailable")

        try:
            raw_nonvalue_bin_payload = await client.get_station_nonvalue_bin_records(station_id, station_dut_id)
            latest_nonvalue_bin = _trim_station_record_to_latest(
                StationRecordResponseSchema.model_validate(raw_nonvalue_bin_payload),
                preserve_limits=False,
            )
            nonvalue_bin_items = _extract_latest_test_items(latest_nonvalue_bin.data, allow_numeric_status=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to collect nonvalueBin test items for station %s (DUT %s): %s",
                station_identifier,
                request.dut_isn,
                exc,
            )
            error_parts.append("nonvalueBin items unavailable")

        try:
            raw_nonvalue_payload = await client.get_station_nonvalue_records(station_id, station_dut_id)
            latest_nonvalue = _trim_station_record_to_latest(
                StationRecordResponseSchema.model_validate(raw_nonvalue_payload),
                preserve_limits=False,
            )
            nonvalue_items = _extract_latest_test_items(latest_nonvalue.data, allow_numeric_status=True)
            if not nonvalue_items:
                error_parts.append("nonvalue items unavailable")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to collect nonvalue test items for station %s (DUT %s): %s",
                station_identifier,
                request.dut_isn,
                exc,
            )
            error_parts.append("nonvalue items unavailable")

        station_entry = StationLatestTestItemsSchema(
            station_id=station_id,
            station_name=station_name,
            station_dut_id=station_dut_id,
            station_dut_isn=station_dut_isn,
            value_test_items=value_items,
            nonvalue_bin_test_items=nonvalue_bin_items,
            nonvalue_test_items=nonvalue_items,
            error="; ".join(dict.fromkeys(error_parts)) if error_parts else None,
        )
        stations.append(station_entry)

    return BatchLatestTestItemsResponseSchema(
        dut_isn=request.dut_isn,
        site_name=site_name,
        model_name=model_name,
        stations=stations,
    )


def _trim_station_record_to_latest(
    payload: StationRecordResponseSchema,
    *,
    metadata: dict[str, Any] | None = None,
    default_isn: str | None = None,
    preserve_limits: bool = True,
) -> StationRecordResponseSchema:
    latest_record: list[StationRecordRowSchema] = []
    trimmed_data: list[list[str | int | float | None]] = []

    if payload.record:
        latest_entry = None
        latest_date = datetime.min
        for row in payload.record:
            parsed_date = _parse_test_date(row.test_date)
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = row
        if latest_entry is not None:
            latest_record = [latest_entry]

    if payload.data:
        for row in payload.data:
            values = list(row)
            trimmed_data.append(_trim_measurement_row(values, preserve_limits))

    trimmed = StationRecordResponseSchema(record=latest_record, data=trimmed_data)
    if metadata or default_isn:
        trimmed = _attach_record_metadata(trimmed, metadata, default_isn)
    return trimmed


def _trim_measurement_row(values: list[str | int | float | None], preserve_limits: bool) -> list[str | int | float | None]:
    if not values:
        return []
    name = values[0]
    if preserve_limits:
        usl = values[1] if len(values) > 1 else None
        lsl = values[2] if len(values) > 2 else None
        measurements = values[3:] if len(values) > 3 else []
        latest = measurements[-1] if measurements else (values[3] if len(values) > 3 else None)
        result: list[str | int | float | None] = [name]
        result.append(usl)
        result.append(lsl)
        result.append(latest)
        return result
    measurements = values[1:]
    latest = measurements[-1] if measurements else None
    return [name, latest]


def _attach_record_metadata(
    payload: StationRecordResponseSchema,
    metadata: dict[str, Any] | None,
    default_isn: str | None,
) -> StationRecordResponseSchema:
    station_name = (metadata or {}).get("station_name")
    site_name = (metadata or {}).get("site_name")
    model_name = (metadata or {}).get("model_name")
    normalized_isn = _normalize_optional_string(default_isn)
    enriched_records: list[StationRecordRowSchema] = []
    for row in payload.record:
        row_data = row.model_dump()
        if station_name and not row_data.get("station"):
            row_data["station"] = station_name
        if site_name and not row_data.get("site_name"):
            row_data["site_name"] = site_name
        if model_name and model_name.strip():
            row_data["model_name"] = model_name
        elif not row_data.get("model_name"):
            row_data.pop("model_name", None)
        if normalized_isn and not row_data.get("isn"):
            row_data["isn"] = normalized_isn
        enriched_records.append(StationRecordRowSchema(**row_data))
    return StationRecordResponseSchema(
        record=enriched_records,
        data=[list(row) if isinstance(row, list) else row for row in payload.data],
    )


def _iter_station_values(station: dict[str, Any], field_names: list[str]) -> Any:
    if not isinstance(station, dict):
        return []
    for field in field_names:
        value = station.get(field)
        if value is not None:
            yield value
    data_entries = station.get("data") if isinstance(station.get("data"), list) else []
    for item in data_entries:
        if not isinstance(item, dict):
            continue
        for field in field_names:
            value = item.get(field)
            if value is not None:
                yield value


def _parse_value_test_items(payload: Any) -> list[TestItemDefinitionSchema]:
    rows = _normalize_record_rows(payload)
    items: list[TestItemDefinitionSchema] = []
    for row in rows:
        if not row:
            continue
        name = _safe_item_name(row[0])
        if not name:
            continue
        usl = _coerce_optional_float(row[1] if len(row) > 1 else None)
        lsl = _coerce_optional_float(row[2] if len(row) > 2 else None)
        items.append(TestItemDefinitionSchema(name=name, usl=usl, lsl=lsl, status=None))
    items.sort(key=lambda item: item.name.lower())
    return items


def _parse_nonvalue_bin_test_items(payload: Any) -> list[TestItemDefinitionSchema]:
    rows = _normalize_record_rows(payload)
    items: list[TestItemDefinitionSchema] = []
    for row in rows:
        if not row:
            continue
        name = _safe_item_name(row[0])
        if not name:
            continue
        items.append(TestItemDefinitionSchema(name=name, usl=None, lsl=None))
    items.sort(key=lambda item: item.name.lower())
    return items


def _parse_nonvalue_test_items(payload: Any) -> list[TestItemDefinitionSchema]:
    rows = _normalize_record_rows(payload)
    items: list[TestItemDefinitionSchema] = []
    for row in rows:
        if not row:
            continue
        name = _safe_item_name(row[0])
        if not name:
            continue
        items.append(TestItemDefinitionSchema(name=name, usl=None, lsl=None))
    items.sort(key=lambda item: item.name.lower())
    return items


def _normalize_record_rows(payload: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if isinstance(payload, StationRecordResponseSchema):
        rows = list(payload.data or [])
    else:
        try:
            validated = StationRecordResponseSchema.model_validate(payload)
        except ValidationError:
            data = payload.get("data") if isinstance(payload, dict) else []
            rows = list(data or [])
        else:
            rows = list(validated.data or [])

    normalized: list[list[Any]] = []
    for entry in rows:
        if not isinstance(entry, list):
            continue
        trimmed = list(entry)
        if len(trimmed) >= 4:
            trimmed = trimmed[:3] + [trimmed[-1]]
        normalized.append(trimmed)
    return normalized


def _safe_item_name(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_status_value(values: Iterable[Any]) -> str | None:
    reversed_values = list(values)
    for raw in reversed(reversed_values):
        if raw in (None, ""):
            continue
        text = str(raw).strip()
        if not text:
            continue
        return text
    return None


def _extract_station_attribute(station: dict[str, Any], field_names: list[str]) -> Any:
    for value in _iter_station_values(station, field_names):
        if value not in (None, ""):
            return value
    return None


def _filter_record_payload(payload: Any, site_hint: str | None, model_hint: str | None) -> Any:
    if not site_hint and not model_hint:
        return payload
    if not isinstance(payload, dict):
        return payload

    data_key = "record_data" if "record_data" in payload else "record"
    record_data = payload.get(data_key)
    if not isinstance(record_data, list):
        return payload

    filtered = [station for station in record_data if _station_matches_hints(station, site_hint, model_hint)]

    if not filtered:
        new_payload = dict(payload)
        new_payload[data_key] = []
        return new_payload

    new_payload = dict(payload)
    new_payload[data_key] = filtered

    if site_hint and not new_payload.get("site_name"):
        candidate_site = _extract_station_attribute(
            filtered[0],
            [
                "site_name",
                "device_id__station_id__model_id__site_id__name",
            ],
        )
        if candidate_site:
            new_payload["site_name"] = candidate_site

    if model_hint and not new_payload.get("model_name"):
        candidate_model = _extract_station_attribute(
            filtered[0],
            [
                "model_name",
                "device_id__station_id__model_id__name",
            ],
        )
        if candidate_model:
            new_payload["model_name"] = candidate_model

    return new_payload


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if trimmed.lower() in {"null", "none"}:
        return None
    return trimmed


def _filter_entries_by_status(
    entries: list[dict] | None,
    status_filter: str | None,
    *,
    default_active_codes: set[int] | None = None,
) -> list[dict]:
    """
    Apply a status-based filter to upstream collections while tolerating legacy casing and synonyms.

    Args:
        entries: Collection returned by the upstream API (list of dict-like objects).
        status_filter: User supplied string (e.g. "ALL", "Active", "Inactive").
        default_active_codes: Override for which numeric codes represent "Active".

    Returns:
        Filtered list preserving original ordering.
    """
    if not entries:
        return entries or []

    allowed_codes = _resolve_status_codes(status_filter, default_active_codes=default_active_codes)
    if allowed_codes is None:
        return entries

    filtered: list[dict] = []
    for entry in entries:
        status_value = None
        if isinstance(entry, dict):
            status_value = entry.get("status")
        elif hasattr(entry, "get"):
            try:
                status_value = entry.get("status")
            except Exception:  # pragma: no cover - defensive for custom mappings
                status_value = None
        code = _coerce_status_code(status_value)
        if code is not None and code in allowed_codes:
            filtered.append(entry)
    return filtered


def _resolve_status_codes(
    status_filter: str | None,
    *,
    default_active_codes: set[int] | None = None,
) -> set[int] | None:
    default_active = default_active_codes or {1, 2}
    if status_filter is None:
        return None

    normalized = status_filter.strip()
    if not normalized:
        return None

    normalized_upper = normalized.upper()
    if normalized_upper == "ALL":
        return None

    cleaned = normalized_upper.replace(" ", "").replace("-", "").replace("/", "")

    if cleaned in {"ACTIVE", "ONLINE", "ENABLED"}:
        return default_active
    if cleaned in {"INACTIVE", "DEACTIVATED", "DEACTIVE", "OFFLINE"}:
        return {0}

    if cleaned.isdigit():
        return {int(cleaned)}

    return None


def _coerce_status_code(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)

    string_value = str(value).strip()
    return int(string_value) if string_value.isdigit() else None


def _coerce_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    string_value = str(value).strip()
    if not string_value:
        return None
    if string_value.isdigit() or (string_value.startswith("-") and string_value[1:].isdigit()):
        try:
            return int(string_value)
        except ValueError:
            return None
    return None


def _extract_attr(value: Any, attr: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(attr, default)
    return getattr(value, attr, default)


def _order_sort_components(
    value: Any,
    *,
    name_getter: Callable[[Any], str | None] | None = None,
) -> tuple[int, float, str]:
    raw_order = _extract_attr(value, "order")
    order_num = _coerce_optional_int(raw_order)
    missing_flag = 1 if order_num is None else 0
    order_value = float("inf") if order_num is None else float(order_num)
    if name_getter:
        fallback_name = name_getter(value) or ""
    else:
        fallback_name = _extract_attr(value, "station_name") or _extract_attr(value, "name") or ""
    if isinstance(fallback_name, str):
        fallback_key = fallback_name.lower()
    else:
        fallback_key = ""
    return missing_flag, order_value, fallback_key


def _device_sort_key_schema(device: DeviceInfoSchema | DevicePeriodEntrySchema) -> tuple[int, float, str]:
    device_id = _coerce_optional_int(getattr(device, "id", None))
    missing_flag = 1 if device_id is None else 0
    device_value = float("inf") if device_id is None else float(device_id)
    fallback_name = getattr(device, "device_name", None) or ""
    if isinstance(fallback_name, str):
        fallback_key = fallback_name.lower()
    else:
        fallback_key = ""
    return missing_flag, device_value, fallback_key
    return None


async def _get_station_metadata(client: DUTAPIClient, station_id: int) -> dict[str, Any]:
    cache_key = dut_metadata_cache.build_metadata_key(_client_base_url(client), "station-meta", str(station_id))

    async def loader() -> dict[str, Any]:
        return await _fetch_station_metadata(client, station_id)

    return await dut_metadata_cache.get_or_set(cache_key, loader, ttl=METADATA_CACHE_TTL)


async def _fetch_station_metadata(client: DUTAPIClient, station_id: int) -> dict[str, Any]:
    try:
        sites = await _get_cached_sites(client)
    except Exception as exc:
        logger.error("Error fetching sites while resolving metadata for station %s: %s", station_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve station metadata from external API") from exc

    for site in sites:
        site_id = _coerce_optional_int(site.get("id"))
        site_name = site.get("name")
        if site_id is None:
            continue
        try:
            models = await _get_cached_models(client, site_id)
        except Exception as exc:
            logger.warning(
                "Error fetching models for site %s while resolving metadata for station %s: %s",
                site_id,
                station_id,
                exc,
            )
            continue
        for model in models:
            model_id = model.get("id")
            model_name = model.get("name")
            if model_id is None:
                continue
            try:
                stations = await _get_cached_stations(client, model_id)
            except Exception as exc:
                logger.warning(
                    "Error fetching stations for model %s while resolving metadata for station %s: %s",
                    model_id,
                    station_id,
                    exc,
                )
                continue
            for station in stations:
                candidate_id = _coerce_optional_int(station.get("id"))
                if candidate_id == station_id:
                    return {
                        "station_id": candidate_id if candidate_id is not None else station_id,
                        "station_name": station.get("name") or "",
                        "model_id": _coerce_optional_int(station.get("model_id")) or _coerce_optional_int(model_id),
                        "model_name": station.get("model_name") or model_name or "",
                        "site_id": _coerce_optional_int(station.get("site_id")) or _coerce_optional_int(site_id),
                        "site_name": station.get("site_name") or site_name or "",
                        "station_status": _coerce_optional_int(station.get("status")),
                        "station_order": _coerce_optional_int(station.get("order")),
                    }
    raise HTTPException(status_code=404, detail="Station metadata not found")


def _enrich_devices_with_context(devices: list[dict]) -> list[dict]:
    if not devices:
        return []

    enriched: list[dict] = []
    for entry in devices:
        if isinstance(entry, dict):
            item = dict(entry)
        else:
            item = {}
        if item.get("device_name") in (None, "", 0):
            for candidate in ("device_id__name", "device", "name"):
                candidate_value = item.get(candidate)
                if candidate_value:
                    item["device_name"] = candidate_value
                    break
        if item.get("id") is None and item.get("device_id") is not None:
            item["id"] = item["device_id"]
        enriched.append(item)
    return enriched


def _enrich_test_items_with_context(items: list[dict]) -> list[dict]:
    if not items:
        return []

    enriched: list[dict] = []
    for entry in items:
        item = dict(entry) if isinstance(entry, dict) else {"value": entry}
        enriched.append(item)
    return enriched


def _prepare_station_filters(stations: list[str] | None):
    if not stations:
        return None
    raw_entries = [s.strip() for s in stations if isinstance(s, str) and s.strip()]
    if not raw_entries:
        return None
    raw_set = set(raw_entries)
    numeric = {entry for entry in raw_set if entry.isdigit()}
    normalized = {_normalize_str(entry) for entry in raw_set}
    return {"raw": raw_set, "numeric": numeric, "normalized": normalized}


def _clone_criteria_rules(rules: dict[str, list[CriteriaRule]] | None) -> dict[str, list[CriteriaRule]] | None:
    if rules is None:
        return None
    return {key: list(value) for key, value in rules.items()}


async def _load_criteria_rules(
    criteria_file: UploadFile | None,
) -> tuple[dict[str, list[CriteriaRule]] | None, str]:
    if criteria_file is None:
        return None, "latest-tests"
    content = await criteria_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded criteria file is empty")
    rules = _load_station_criteria_from_bytes(content)
    return rules, criteria_file.filename or "uploaded"


async def _generate_top_product_batch(
    dut_isns: list[str],
    *,
    stations: list[str] | None,
    site_identifier: str | None,
    model_identifier: str | None,
    device_identifiers: list[str] | None,
    test_item_filters: list[str] | None,
    exclude_test_item_filters: list[str] | None,
    station_filters_map: dict[str, StationFilterConfigSchema] | None,  # UPDATED: Added per-station filters
    criteria_rules: dict[str, list[CriteriaRule]] | None,
    criteria_label: str,
    client: DUTAPIClient,
    db: Session,
    postprocessor: Callable[[TopProductResponseSchema], None] | None = None,
    include_pa_trends: bool = False,
) -> TopProductBatchResponseSchema:
    normalized_isns = [value.strip() for value in dut_isns if value and value.strip()]
    if not normalized_isns:
        raise HTTPException(status_code=400, detail="dut_isn is required")

    results: list[TopProductResponseSchema] = []
    errors: list[TopProductErrorSchema] = []

    # Process ISNs in parallel for better performance
    # Especially important for PA trends which adds 200-500ms per station
    import asyncio

    async def process_isn(current_isn: str) -> tuple[TopProductResponseSchema | None, TopProductErrorSchema | None]:
        try:
            response = await _compute_top_product_response(
                client,
                db,
                current_isn,
                stations=stations,
                site_identifier=site_identifier,
                model_identifier=model_identifier,
                device_identifiers=device_identifiers,
                test_item_filters=test_item_filters,
                exclude_test_item_filters=exclude_test_item_filters,
                station_filters_map=station_filters_map,  # UPDATED: Pass station filters
                criteria_rules=_clone_criteria_rules(criteria_rules),
                criteria_label=criteria_label,
                include_pa_trends=include_pa_trends,
            )
            if postprocessor is not None:
                postprocessor(response)
            return response, None
        except HTTPException as exc:
            if len(normalized_isns) == 1:
                raise
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            return None, TopProductErrorSchema(dut_isn=current_isn, detail=detail)
        except Exception as exc:
            if len(normalized_isns) == 1:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return None, TopProductErrorSchema(dut_isn=current_isn, detail=str(exc))

    # Execute all ISN processing tasks concurrently
    tasks = [process_isn(isn) for isn in normalized_isns]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results and errors
    for item in completed:
        if isinstance(item, Exception):
            # If a task raised an exception (shouldn't happen with our error handling)
            if len(normalized_isns) == 1:
                raise item
            errors.append(TopProductErrorSchema(dut_isn="unknown", detail=str(item)))
        else:
            result, error = item
            if result:
                results.append(result)
            if error:
                errors.append(error)

    if not results:
        if errors:
            raise HTTPException(status_code=404, detail=errors[0].detail)
        raise HTTPException(status_code=404, detail="No evaluation data available for the provided DUT identifiers")

    return TopProductBatchResponseSchema(results=results, errors=errors)


def _parse_measurement_components(test_item: str) -> tuple[str, str, str, str] | None:
    tokens = [token for token in test_item.split("_") if token]
    if not tokens:
        return None

    pattern_tokens: list[str] = []
    idx = len(tokens) - 1
    while idx >= 0:
        token = tokens[idx]
        if any(char.isdigit() for char in token):
            pattern_tokens.append(token)
            idx -= 1
            continue
        if pattern_tokens:
            break
        break

    if pattern_tokens:
        group_key = "_".join(reversed(pattern_tokens))
        core_tokens = tokens[: len(tokens) - len(pattern_tokens)]
    else:
        group_key = "general"
        core_tokens = tokens

    if core_tokens and core_tokens[0].lower() == "wifi":
        core_tokens = core_tokens[1:]

    subgroup = None
    antenna = None
    category_tokens: list[str] = []
    antenna_pattern = re.compile(r"^(tx|rx|pa|ant|rssi)(\d+)$", re.IGNORECASE)

    for idx, token in enumerate(core_tokens):
        match = antenna_pattern.match(token)
        if match:
            subgroup = match.group(1).upper()
            antenna = token.upper()
            category_tokens = [tok.upper() for tok in core_tokens[idx + 1 :]]
            break

    if subgroup is None:
        for idx, token in enumerate(core_tokens):
            upper_token = token.upper()
            if upper_token in {"TX", "RX", "PA", "ANT", "RSSI"}:
                subgroup = upper_token
                antenna = upper_token
                category_tokens = [tok.upper() for tok in core_tokens[idx + 1 :]]
                break

    if subgroup is None:
        subgroup = "OTHER"
        antenna = "OTHER"
        category_tokens = [tok.upper() for tok in core_tokens]

    if not category_tokens:
        category_tokens = ["RESULT"]

    category = "_".join(category_tokens) or "RESULT"
    return group_key or "general", subgroup, antenna, category


def _build_hierarchical_scores(
    measurements: list[list[str | float | None]],
) -> tuple[dict[str, Any], dict[str, float]]:
    def _round_score(value: float) -> float:
        return round(value + 1e-9, 2)

    group_map: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

    for row in measurements:
        # Handle both old array format and new dict format
        if isinstance(row, dict):
            name = str(row.get("test_item", ""))
            # Extract score from score_breakdown
            breakdown = row.get("score_breakdown")
            if breakdown and isinstance(breakdown, dict):
                score = _to_float(breakdown.get("final_score"))
            else:
                score = None
        elif isinstance(row, list) and len(row) >= 6:
            # Legacy array format support
            name = str(row[0])
            score = _to_float(row[5])
        else:
            continue

        if score is None:
            continue
        parsed = _parse_measurement_components(name)
        if parsed is None:
            continue
        group_key, subgroup, antenna, category = parsed
        group_entry = group_map.setdefault(group_key, {})
        subgroup_entry = group_entry.setdefault(subgroup, {})
        antenna_entry = subgroup_entry.setdefault(antenna, {})
        antenna_entry.setdefault(category, []).append(score)

    final_groups: dict[str, Any] = {}
    subgroup_totals: dict[str, list[float]] = defaultdict(list)

    for group_key in sorted(group_map.keys()):
        subgroup_map = group_map[group_key]
        group_result: dict[str, Any] = {}
        subgroup_scores: list[float] = []

        for subgroup in sorted(subgroup_map.keys()):
            antenna_map = subgroup_map[subgroup]
            subgroup_result: dict[str, Any] = {}
            antenna_scores: list[float] = []

            for antenna in sorted(antenna_map.keys()):
                categories = antenna_map[antenna]
                category_result: dict[str, float] = {}
                category_scores: list[float] = []
                for category in sorted(categories.keys()):
                    values = categories[category]
                    avg = sum(values) / len(values)
                    category_result[category] = _round_score(avg)
                    category_scores.append(avg)
                antenna_avg = sum(category_scores) / len(category_scores) if category_scores else 0.0
                category_result[f"{antenna.lower()}_score"] = _round_score(antenna_avg)
                subgroup_result[antenna] = category_result
                if category_scores:
                    antenna_scores.append(antenna_avg)

            subgroup_avg = sum(antenna_scores) / len(antenna_scores) if antenna_scores else 0.0
            subgroup_result[f"{subgroup.lower()}_group_score"] = _round_score(subgroup_avg)
            group_result[subgroup] = subgroup_result
            if antenna_scores:
                subgroup_scores.append(subgroup_avg)
                subgroup_totals[subgroup].append(subgroup_avg)

        group_avg = sum(subgroup_scores) / len(subgroup_scores) if subgroup_scores else 0.0
        group_result["group_score"] = _round_score(group_avg)
        final_groups[group_key] = group_result

    overall_group_scores = {subgroup: _round_score(sum(values) / len(values)) for subgroup, values in subgroup_totals.items() if values}

    return final_groups, overall_group_scores


def _apply_hierarchical_scoring(response: TopProductResponseSchema) -> None:
    for station in response.test_result:
        group_scores, overall_group_scores = _build_hierarchical_scores(station.data)
        station.group_scores = group_scores
        station.overall_group_scores = overall_group_scores or None


def _parse_criteria_content(lines: Iterable[str]) -> dict[str, list[CriteriaRule]]:
    station_rules: dict[str, list[CriteriaRule]] = {}
    current_station: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_station = line[1:-1].strip()
            continue
        if current_station is None:
            continue
        rule = _parse_criteria_line(line)
        if rule is None:
            continue
        key = _normalize_str(current_station)
        station_rules.setdefault(key, []).append(rule)
    return station_rules


def _load_station_criteria_from_path(path: FilePath) -> dict[str, list[CriteriaRule]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Criteria file not found: {path}") from exc

    cached = _CRITERIA_CACHE.get(resolved)
    if cached is not None:
        return cached

    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read criteria file: {exc}") from exc

    station_rules = _parse_criteria_content(lines)
    _CRITERIA_CACHE[resolved] = station_rules
    return station_rules


def _load_station_criteria_from_bytes(data: bytes) -> dict[str, list[CriteriaRule]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
    return _parse_criteria_content(text.splitlines())


def _build_default_wireless_criteria(record_data: list[dict]) -> dict[str, list[CriteriaRule]]:
    keywords = ["PA_POW_OLD", "TX_POW", "TX_FIXTURE_OR_DUT_PROBLEM"]
    compiled_keywords = [re.compile(keyword, re.IGNORECASE) for keyword in keywords]
    station_rules: dict[str, list[CriteriaRule]] = {}

    for entry in record_data:
        name = entry.get("name") or ""
        if not name:
            continue
        if not re.search(r"(wireless|wifi)", name, re.IGNORECASE):
            continue
        normalized = _normalize_str(name)
        station_rules[normalized] = [CriteriaRule(pattern=pattern, usl=None, lsl=None, target=None) for pattern in compiled_keywords]
    return station_rules


def _select_station_rules(criteria_map: dict[str, list[CriteriaRule]], station_name: str) -> list[CriteriaRule]:
    return criteria_map.get(_normalize_str(station_name), [])


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_latest_test_items(rows: list[list[Any]] | None, *, allow_numeric_status: bool = False) -> list[TestItemDefinitionSchema]:
    """Extract test item definitions (name, USL, LSL, status) from latest station record rows."""
    if not rows:
        return []
    definitions: list[TestItemDefinitionSchema] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        name = str(row[0]).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        usl = _to_float(row[1]) if len(row) > 1 else None
        lsl = _to_float(row[2]) if len(row) > 2 else None

        status: str | None = None
        if len(row) > 3:
            for candidate in reversed(row[3:]):
                if candidate is None:
                    continue
                if isinstance(candidate, str):
                    candidate_str = candidate.strip()
                    if not candidate_str:
                        continue
                    try:
                        float(candidate_str)
                        continue
                    except ValueError:
                        status = candidate_str
                        break
                else:
                    if allow_numeric_status:
                        status = str(candidate).strip()
                        break
                    try:
                        float(candidate)
                        continue
                    except (TypeError, ValueError):
                        candidate_str = str(candidate).strip()
                        if candidate_str:
                            status = candidate_str
                            break

        definitions.append(TestItemDefinitionSchema(name=name, usl=usl, lsl=lsl))
        seen.add(key)

    definitions.sort(key=lambda item: item.name.lower())
    return definitions


def _extract_value_items_from_payload(payload: Any) -> list[TestItemDefinitionSchema]:
    try:
        value_payload = StationRecordResponseSchema.model_validate(payload)
    except ValidationError:
        logger.debug("Unable to validate value payload while extracting test items")
        return []
    return _extract_latest_test_items(value_payload.data)


def _extract_latest_measurements(payload: StationRecordResponseSchema, run_index: int | None) -> list[MeasurementRow]:
    measurements: list[MeasurementRow] = []
    for row in payload.data:
        if not isinstance(row, list) or len(row) < 4:
            continue
        name = str(row[0])
        usl = _to_float(row[1]) if len(row) > 1 else None
        lsl = _to_float(row[2]) if len(row) > 2 else None
        value_index = len(row) - 1
        if run_index is not None:
            candidate_index = 3 + run_index
            if candidate_index < len(row):
                value_index = candidate_index
        latest = _to_float(row[value_index])
        measurements.append(MeasurementRow(name=name, usl=usl, lsl=lsl, latest=latest))
    return measurements


def _compute_target_value(rule: CriteriaRule, row: MeasurementRow, actual: float | None, test_item_name: str | None = None) -> float | None:
    # Check if rule has explicit target - this takes highest priority
    if rule.target is not None:
        return rule.target
    usl = rule.usl if rule.usl is not None else row.usl
    lsl = rule.lsl if rule.lsl is not None else row.lsl
    # For PER test items, default to 0.0 instead of midpoint (but can be overridden by rule.target above)
    if test_item_name and ("PER" in test_item_name.upper() or "_PER_" in test_item_name.upper()):
        return 0.0
    # For PA POW_DIF_ABS test items, target is always 0.0 (ideal = no difference)
    if test_item_name and "POW_DIF_ABS" in test_item_name.upper():
        return 0.0
    if usl is not None and lsl is not None:
        return (usl + lsl) / 2
    if usl is not None:
        return usl
    if lsl is not None:
        return lsl
    return actual


def _detect_measurement_category(test_item: str) -> str | None:
    """
    Detect the measurement category from the test item name.
    Returns the category type (e.g., 'EVM', 'PER', 'POW', 'FREQ', 'PA_ADJUSTED_POWER', 'PA_POW_DIF_ABS') or None.
    """
    # Check for common patterns first
    test_item_upper = test_item.upper()

    # PA adjusted power pattern check (special virtual test item)
    # Format: WiFi_PA{1-4}_{frequency}_{protocol}_{mode}_ADJUSTED_POW_{MID|MEAN}
    # Note: Uses "ADJUSTED_POW" (abbreviated form)
    if "PA" in test_item_upper and ("ADJUSTED_POW" in test_item_upper or "ADJUSTED_POWER" in test_item_upper):
        return "PA_ADJUSTED_POWER"

    # PA POW_DIF_ABS pattern check (power difference absolute value after calibration)
    # Format: WiFi_PA{1-4}_POW_DIF_ABS_{frequency}_{protocol}_{mode}
    # This represents the delta between POW_OLD measurement and target (always has target=0)
    if "PA" in test_item_upper and "POW_DIF_ABS" in test_item_upper:
        return "PA_POW_DIF_ABS"

    # FREQ pattern check (e.g., WiFi_TX1_FREQ_2462)
    if "FREQ" in test_item_upper or "_FREQ_" in test_item_upper:
        return "FREQ"

    parsed = _parse_measurement_components(test_item)
    if parsed is None:
        return None
    _, _, _, category = parsed
    # Extract the first token from category which typically indicates the measurement type
    tokens = category.split("_")
    if tokens:
        return tokens[0]  # Returns 'EVM', 'PER', 'POW', etc.
    return None


def _calculate_evm_score(usl: float | None, actual: float) -> tuple[float | None, float, dict]:
    """
    Calculate score for EVM (Error Vector Magnitude) measurements.

    EVM scoring logic:
    - Lower EVM values are better (closer to theoretical minimum)
    - USL determines the acceptable range
    - Theoretical minimum is around -60 dB (difficult to achieve)
    - Score range scales based on USL value

    Args:
        usl: Upper Spec Limit (e.g., -5, -3)
        actual: Measured EVM value (e.g., -17)

    Returns:
        Tuple of (deviation, score, breakdown)
        - deviation: Distance from USL (0 if within spec)
        - score: 0-10 scale, where 10 is best
        - breakdown: Dict with detailed score calculation and LaTeX formula
    """
    if usl is None:
        # If no USL provided, use default scoring
        # Assume -3 dB as default acceptable limit
        usl = -3.0

    # Theoretical EVM minimum (excellent performance)
    theoretical_min = -60.0

    # If actual exceeds USL (worse than spec), fail with low score
    if actual > usl:
        deviation = actual - usl
        # Heavy penalty for exceeding USL
        penalty_factor = min(deviation / max(abs(usl), 1.0), 1.0)
        score = max(0.0, 5.0 - penalty_factor * 5.0)
        formula = r"$score = \max\left(0, 5 - \frac{actual - USL}{|USL|} \times 5\right)$ (exceeds USL)"
        breakdown = {
            "category": "EVM",
            "method": "EVM Advanced (Exceeds USL)",
            "usl": usl,
            "lsl": None,
            "target_used": theoretical_min,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": round(score, 2),
            "final_score": round(score, 2),
            "formula_latex": formula,
        }
        return deviation, round(score, 2), breakdown

    # Calculate how close we are to theoretical minimum
    # The range is from USL (acceptable) to theoretical_min (excellent)
    spec_range = abs(usl - theoretical_min)

    # Calculate position within the range (0 = at USL, 1 = at theoretical min)
    if spec_range > 0:
        position = (usl - actual) / spec_range
        position = max(0.0, min(1.0, position))  # Clamp to [0, 1]
    else:
        position = 1.0

    # Map position to score (6-10 range for within spec)
    # At USL: score = 6
    # At theoretical_min: score = 10
    base_score = 6.0
    score_range = 4.0
    score = base_score + (position * score_range)

    # For very good results (< USL - 10), give bonus
    if actual < (usl - 10):
        bonus = min((usl - 10 - actual) / 10.0, 1.0) * 0.5
        score = min(10.0, score + bonus)

    deviation = 0.0  # Within spec, no deviation

    formula = r"$score = 6 + 4 \times \frac{USL - actual}{|USL - (-60)|}$ (within spec, bonus if $< USL - 10$)"
    breakdown = {
        "category": "EVM",
        "method": "EVM Advanced",
        "usl": usl,
        "lsl": None,
        "target_used": theoretical_min,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": formula,
    }
    return deviation, round(score, 2), breakdown


def _calculate_freq_score(usl: float | None, lsl: float | None, target: float, actual: float) -> tuple[float | None, float, dict]:
    """
    Calculate score for FREQ (Frequency) measurements.

    FREQ scoring logic:
    - Target is usually 0 (center frequency)
    - Score based on deviation from target relative to spec limits
    - Linear scoring within the allowed range

    Args:
        usl: Upper Spec Limit
        lsl: Lower Spec Limit
        target: Target value (usually 0)
        actual: Measured frequency offset

    Returns:
        Tuple of (deviation, score, breakdown)
        - deviation: Distance from target
        - score: 0-10 scale based on position within limits
        - breakdown: Dict with scoring details and LaTeX formula
    """
    deviation = abs(actual - target)

    # Check if out of spec
    if (usl is not None and actual > usl) or (lsl is not None and actual < lsl):
        breakdown = {
            "category": "Frequency",
            "method": "Out of spec - Linear scoring based on deviation from target",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": 0.0,
            "final_score": 0.0,
            "formula_latex": r"$\text{Score} = 0 \text{ (out of spec)}$",
        }
        return deviation, 0.0, breakdown

    # Calculate span for normalization
    if usl is not None and lsl is not None:
        span = (usl - lsl) / 2.0
    elif usl is not None:
        span = abs(usl - target)
    elif lsl is not None:
        span = abs(target - lsl)
    else:
        span = max(abs(target) * 0.1, 1.0)

    # Ensure span is not too small
    span = max(span, 1e-6)

    # Normalize deviation (0 = at target, 1 = at limit)
    normalized = deviation / span

    # Score decreases linearly from 10 (at target) to 0 (at limit)
    score = max(0.0, min(10.0, 10.0 - (normalized * 10.0)))

    breakdown = {
        "category": "Frequency",
        "method": "Linear scoring based on deviation from target",
        "usl": usl,
        "lsl": lsl,
        "target_used": target,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{actual} - \text{target}|}{\text{span}}\right)$",
    }
    return deviation, round(score, 2), breakdown


def _calculate_per_score(usl: float | None, actual: float) -> tuple[float | None, float, dict]:
    """
    Calculate score for PER (Packet Error Rate) measurements.

    PER scoring logic:
    - Lower PER values are better (closer to 0)
    - PER of 0 is perfect (no packet errors)
    - Score decreases relative to the provided USL

    Args:
        usl: Upper Spec Limit used for normalization
        actual: Measured PER value (0 = perfect, higher = worse)

    Returns:
        Tuple of (deviation, score, breakdown)
        - deviation: Distance from the zero baseline
        - score: 0-10 scale, where 10 is best (PER = 0)
        - breakdown: Dict with scoring details and LaTeX formula
    """
    # Ensure we have a positive normalization range
    if usl is None or abs(usl) < 1e-9:
        usl = 1.0

    # Perfect measurement
    if actual <= 0:
        breakdown = {
            "category": "PER",
            "method": "Perfect PER (0 errors)",
            "usl": usl,
            "lsl": None,
            "target_used": 0.0,
            "actual": actual,
            "deviation": 0.0,
            "raw_score": 10.0,
            "final_score": 10.0,
            "formula_latex": r"$\text{Score} = 10 \text{ (PER} = 0)$",
        }
        return 0.0, 10.0, breakdown

    deviation = actual

    # Score is proportional to distance from USL, capped between 0 and 10
    remaining_margin = max(usl - actual, 0.0)
    score = max(0.0, min(10.0, (remaining_margin / usl) * 10.0))

    breakdown = {
        "category": "PER",
        "method": "Linear scoring based on remaining margin to USL",
        "usl": usl,
        "lsl": None,
        "target_used": 0.0,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": r"$\text{Score} = 10 \times \frac{\text{USL} - \text{actual}}{\text{USL}}$",
    }
    return deviation, round(score, 2), breakdown


def _calculate_pa_adjusted_power_score(actual: float, threshold: float = 5.0) -> tuple[float | None, float, dict]:
    """
    Calculate score for PA (Power Amplifier) adjusted power measurements.

    PA adjusted power scoring logic:
    - Target value is 0 (ideal calibration, no power adjustment needed)
    - Lower absolute deviation from 0 is better
    - Uses linear interpolation based on deviation from 0
    - Score uses 0-10 scale with linear decay based on threshold

    Linear scoring formula:
    - score = 10.0 * (1 - deviation / threshold)
    - threshold defaults to 5.0 dB (configurable)
    - Example scores (threshold=5.0):
      * deviation 0.0 → score 10.0 (perfect)
      * deviation 1.0 → score 8.0
      * deviation 2.5 → score 5.0
      * deviation 5.0 → score 0.0 (at threshold)
      * deviation >5.0 → score 0.0 (exceeds threshold)

    Args:
        actual: Calculated PA adjusted power value (NEW - OLD) / 256
        threshold: Maximum acceptable deviation (default: 5.0 dB)

    Returns:
        Tuple of (deviation, score, breakdown)
        - deviation: Absolute distance from 0
        - score: 0-10 scale, where 10 is best (value = 0)
        - breakdown: Dict with scoring details and LaTeX formula
    """
    # Deviation is absolute distance from ideal value (0)
    deviation = abs(actual)

    # Linear scoring: score = 10.0 * (1 - deviation / threshold)
    # Clamp to [0, 10] range
    if deviation >= threshold:
        score = 0.0
    else:
        score = 10.0 * (1.0 - deviation / threshold)
        score = max(0.0, min(10.0, score))

    breakdown = {
        "category": "PA Power",
        "method": "Linear scoring based on deviation from 0 (ideal calibration)",
        "usl": threshold,
        "lsl": -threshold,
        "target_used": 0.0,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{actual}|}{\text{threshold}}\right)$",
    }
    return deviation, round(score, 2), breakdown


def _calculate_pa_pow_dif_abs_score(actual: float, usl: float | None = None) -> tuple[float | None, float, dict]:
    """
    Calculate linear score for PA POW_DIF_ABS measurements.

    PA POW_DIF_ABS represents the absolute power difference between the POW_OLD
    measurement and its target value (before vs after calibration delta).
    The target is always 0 (ideal: no difference).

    Scoring logic:
    - Uses linear interpolation based on deviation from 0
    - If USL is provided: score linearly from 10 (at 0) to 0 (at USL)
    - If no USL: use default threshold of 5.0 dB
    - Closer to 0 = better score
    - Score range: 0-10

    Args:
        actual: Measured POW_DIF_ABS value (absolute difference, e.g., 0.5, 1.2)
        usl: Optional upper spec limit (if not provided, default is 5.0)

    Returns:
        Tuple of (deviation, score, breakdown)
        - deviation: Absolute value (same as actual since target is 0)
        - score: 0-10 scale, where 10 is perfect (0 difference)
        - breakdown: Dict with scoring details and LaTeX formula

    Examples:
        actual = 0.0, usl = 5.0 -> score = 10.0 (perfect match)
        actual = 1.0, usl = 5.0 -> score = 8.0
        actual = 2.5, usl = 5.0 -> score = 5.0
        actual = 4.0, usl = 5.0 -> score = 2.0
        actual = 5.0, usl = 5.0 -> score = 0.0 (at limit)
        actual = 6.0, usl = 5.0 -> score = 0.0 (exceeds limit)
    """
    # Target is always 0 for POW_DIF_ABS
    deviation = abs(actual)

    # Use USL if provided and > 0, otherwise default threshold of 5.0
    # Note: POW_DIF_ABS items typically have usl=0 from external API, which is not a valid threshold
    threshold = usl if (usl is not None and usl > 0) else 5.0

    # Linear interpolation: score = 10 * (1 - deviation / threshold)
    # Clamp to [0, 10] range
    if deviation >= threshold:
        score = 0.0
    else:
        score = 10.0 * (1.0 - deviation / threshold)
        score = max(0.0, min(10.0, score))

    breakdown = {
        "category": "PA Power",
        "method": "Linear scoring for POW_DIF_ABS (calibration delta)",
        "usl": threshold,
        "lsl": None,
        "target_used": 0.0,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{actual}|}{\text{threshold}}\right)$",
    }
    return deviation, round(score, 2), breakdown


def _calculate_fixture_power_score(target: float, actual: float) -> tuple[float | None, float, dict]:
    """
    Score FIXTURE_OR_DUT_PROBLEM_POW measurements based on deviation from the target power.
    """
    if target == 0:
        breakdown = {
            "category": "Fixture Power",
            "method": "Target is 0 - cannot calculate ratio",
            "usl": None,
            "lsl": None,
            "target_used": target,
            "actual": actual,
            "deviation": abs(actual - target),
            "raw_score": 0.0,
            "final_score": 0.0,
            "formula_latex": r"$\text{Score} = 0 \text{ (target is } 0)$",
        }
        return abs(actual - target), 0.0, breakdown
    deviation = abs(actual - target)
    ratio = max(0.0, 1.0 - (deviation / abs(target)))
    score = max(0.0, min(10.0, ratio * 10.0))
    breakdown = {
        "category": "Fixture Power",
        "method": "Ratio-based scoring relative to target",
        "usl": None,
        "lsl": None,
        "target_used": target,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{actual} - \text{target}|}{|\text{target}|}\right)$",
    }
    return deviation, round(score, 2), breakdown


def _calculate_bounded_measurement_score(
    usl: float,
    lsl: float,
    target: float,
    actual: float,
) -> tuple[float | None, float, dict]:
    """
    Score measurements with explicit USL/LSL/Target values (non EVM/PER/FIXTURE).

    Special handling:
    - When target is 0, uses deviation-based scoring (not headroom logic)
    - Otherwise uses ratio-based scoring
    """
    deviation = abs(actual - target)
    if actual < lsl or actual > usl:
        breakdown = {
            "category": "General",
            "method": "Out of spec - bounded measurement",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": 0.0,
            "final_score": 0.0,
            "formula_latex": r"$\text{Score} = 0 \text{ (out of spec)}$",
        }
        return deviation, 0.0, breakdown

    # Special case: When target is 0, use deviation-based scoring
    # This prevents the "headroom" logic that gives low scores for values near 0
    EPS = 1e-9
    if abs(target) <= EPS:
        # Calculate span for normalization
        span = (usl - lsl) / 2.0
        span = max(span, EPS)

        # Normalize deviation (0 = at target, 1 = at limit)
        normalized = deviation / span

        # Score decreases linearly from 10 (at target) to 0 (at limit)
        score = max(0.0, min(10.0, 10.0 - (normalized * 10.0)))
        breakdown = {
            "category": "General",
            "method": "Deviation-based scoring (target is 0)",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": round(score, 2),
            "final_score": round(score, 2),
            "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{actual} - \text{target}|}{\text{span}}\right)$",
        }
        return deviation, round(score, 2), breakdown

    # Original ratio-based logic for non-zero targets
    measured_value = abs(actual)
    target_value = abs(target)
    max_value = max(measured_value, target_value)
    if max_value <= EPS:
        score = 10.0
        formula = r"$\text{Score} = 10 \text{ (both values near 0)}$"
    else:
        min_value = min(measured_value, target_value)
        if min_value <= EPS:
            span = max(usl - lsl, EPS)
            remaining = max(usl - measured_value, 0.0)
            raw_score = max(0.0, min((remaining / span) * 10.0, 10.0))
            formula = r"$\text{Score} = 10 \times \frac{\text{USL} - |\text{actual}|}{\text{USL} - \text{LSL}}$"
        else:
            ratio = min_value / max_value
            raw_score = max(0.0, min(ratio * 10.0, 10.0))
            formula = r"$\text{Score} = 10 \times \frac{\min(|\text{actual}|, |\text{target}|)}{\max(|\text{actual}|, |\text{target}|)}$"
        if 0.0 < raw_score < 0.01:
            score = 0.01
        else:
            score = raw_score
    breakdown = {
        "category": "General",
        "method": "Ratio-based scoring for non-zero target",
        "usl": usl,
        "lsl": lsl,
        "target_used": target,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": round(score, 2),
        "final_score": round(score, 2),
        "formula_latex": formula,
    }
    return deviation, round(score, 2), breakdown


def _calculate_measurement_metrics(
    usl: float | None,
    lsl: float | None,
    target: float | None,
    actual: float | None,
    test_item: str | None = None,
) -> tuple[float | None, float, dict]:
    if actual is None or target is None:
        breakdown = {
            "category": "General",
            "method": "Missing actual or target value",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": None,
            "raw_score": 0.0,
            "final_score": 0.0,
            "formula_latex": r"$\text{Score} = 0 \text{ (missing data)}$",
        }
        return None, 0.0, breakdown

    normalized_test_item = test_item.upper() if isinstance(test_item, str) else ""
    if normalized_test_item and "FIXTURE_OR_DUT_PROBLEM_POW" in normalized_test_item:
        return _calculate_fixture_power_score(target, actual)

    # UPDATED: Check for specialized scoring categories (EVM, PER, FREQ, PA_ADJUSTED_POWER, PA_POW_DIF_ABS)
    if test_item is not None:
        category = _detect_measurement_category(test_item)
        if category == "EVM":
            # Use specialized EVM scoring
            return _calculate_evm_score(usl, actual)
        elif category == "PER":
            # Use specialized PER scoring
            return _calculate_per_score(usl, actual)
        elif category == "FREQ":
            # Use specialized FREQ scoring
            return _calculate_freq_score(usl, lsl, target, actual)
        elif category == "PA_ADJUSTED_POWER":
            # Use specialized PA adjusted power scoring (target = 0, linear with threshold)
            # Note: threshold would need to be passed from endpoint if customizable
            return _calculate_pa_adjusted_power_score(actual, threshold=5.0)
        elif category == "PA_POW_DIF_ABS":
            # Use specialized PA POW_DIF_ABS scoring (linear, target = 0)
            return _calculate_pa_pow_dif_abs_score(actual, usl)

    if usl is not None and lsl is not None and target is not None:
        return _calculate_bounded_measurement_score(usl, lsl, target, actual)

    # Default scoring logic for other categories
    if usl is None and lsl is not None:
        deviation = max(0.0, lsl - actual)
        span = max(abs(lsl) * 0.1, 1.0)
        normalized = deviation / span if span else 0.0
        score = max(0.0, 10.0 - min(normalized, 1.0) * 10.0)
        breakdown = {
            "category": "General",
            "method": "LSL-only scoring (lower limit only)",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": score,
            "final_score": score,
            "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{\max(0, \text{LSL} - \text{actual})}{\text{span}}\right)$",
        }
        return deviation, score, breakdown

    if usl is not None and lsl is None:
        deviation = max(0.0, actual - usl)
        span = max(abs(usl) * 0.1, 1.0)
        normalized = deviation / span if span else 0.0
        score = max(0.0, 10.0 - min(normalized, 1.0) * 10.0)
        breakdown = {
            "category": "General",
            "method": "USL-only scoring (upper limit only)",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": score,
            "final_score": score,
            "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{\max(0, \text{actual} - \text{USL})}{\text{span}}\right)$",
        }
        return deviation, score, breakdown

    deviation = abs(actual - target)
    span: float | None = None
    if usl is not None and lsl is not None and usl > lsl:
        span = max((usl - lsl) / 2.0, 1e-6)
    elif usl is not None:
        span = max(abs(usl - target), 1.0)
    elif lsl is not None:
        span = max(abs(target - lsl), 1.0)
    else:
        span = max(abs(target) * 0.1, 1.0)

    normalized = deviation / span if span else 0.0
    score = max(0.0, 10.0 - min(normalized, 1.0) * 10.0)
    breakdown = {
        "category": "General",
        "method": "Default deviation-based scoring",
        "usl": usl,
        "lsl": lsl,
        "target_used": target,
        "actual": actual,
        "deviation": round(deviation, 2) if deviation is not None else None,
        "raw_score": score,
        "final_score": score,
        "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{actual} - \text{target}|}{\text{span}}\right)$",
    }
    return deviation, score, breakdown


def _score_measurement(rule: CriteriaRule, row: MeasurementRow, *, target_override: float | None = None) -> MeasurementScore | None:
    actual = row.latest
    if actual is None:
        return None
    usl = rule.usl if rule.usl is not None else row.usl
    lsl = rule.lsl if rule.lsl is not None else row.lsl
    target = target_override
    if target is None:
        target = _compute_target_value(rule, row, actual, row.name)
    if target is None:
        target = actual

    out_of_spec = False
    if usl is not None and actual > usl:
        out_of_spec = True
    if lsl is not None and actual < lsl:
        out_of_spec = True

    if out_of_spec:
        deviation = abs(actual - target) if target is not None else None
        # Create breakdown for out-of-spec case
        out_of_spec_breakdown = {
            "category": "General",
            "method": "Out of specification",
            "usl": usl,
            "lsl": lsl,
            "target_used": target,
            "actual": actual,
            "deviation": round(deviation, 2) if deviation is not None else None,
            "raw_score": 0.0,
            "final_score": 0.0,
            "formula_latex": r"$\text{Score} = 0 \text{ (out of spec)}$",
        }
        return MeasurementScore(
            test_item=row.name,
            usl=usl,
            lsl=lsl,
            target=target,
            actual=actual,
            deviation=deviation,
            score_value=0.0,
            score_breakdown=out_of_spec_breakdown,
        )

    # UPDATED: Pass test_item (row.name) for category-specific scoring (EVM, PER)
    deviation, score_value, breakdown = _calculate_measurement_metrics(usl, lsl, target, actual, row.name)
    return MeasurementScore(
        test_item=row.name,
        usl=usl,
        lsl=lsl,
        target=target,
        actual=actual,
        deviation=deviation,
        score_value=score_value,
        score_breakdown=breakdown,
    )


def _extract_candidate_runs(
    station_entry: dict,
    start_time: datetime | None,
    end_time: datetime | None,
) -> dict[tuple[int | None, str], list[dict]]:
    groups: dict[tuple[int | None, str], list[dict]] = {}
    for item in station_entry.get("data") or []:
        test_date = _parse_test_date(item.get("test_date"))
        if start_time and (test_date is None or test_date.replace(tzinfo=UTC) < start_time):
            continue
        if end_time and (test_date is None or test_date.replace(tzinfo=UTC) > end_time):
            continue
        key_isn = _normalize_str(item.get("dut_id__isn") or item.get("dut_isn"))
        key = (item.get("dut_id"), key_isn)
        groups.setdefault(key, []).append(item)
    return groups


def _resolve_selected_station_names(
    criteria_map: dict[str, list[CriteriaRule]],
    requested: list[str] | None,
    record_data: list[dict],
) -> set[str]:
    if not requested:
        return set(criteria_map.keys())

    id_to_name: dict[str, str] = {str(entry.get("id")): _normalize_str(entry.get("name")) for entry in record_data if entry.get("id") is not None}
    selected: set[str] = set()
    for raw in requested:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        if candidate.isdigit() and candidate in id_to_name:
            selected.add(id_to_name[candidate])
        else:
            selected.add(_normalize_str(candidate))
    if not selected:
        return set(criteria_map.keys())
    return selected


def _evaluate_station_candidate(
    station_entry: dict,
    station_payload: StationRecordResponseSchema,
    latest_entry: dict,
    candidate_runs: list[dict],
    rules: list[CriteriaRule],
    site_name: str | None,
    model_name: str | None,
) -> StationEvaluation | None:
    if not rules:
        return None

    run_index = _find_run_index(station_payload, latest_entry)
    measurement_rows = _extract_latest_measurements(station_payload, run_index)
    measurement_scores: list[MeasurementScore] = []

    matched_any_rule = False
    for rule in rules:
        matching_rows = [row for row in measurement_rows if _row_matches_rule(row, rule)]
        if not matching_rows:
            continue
        matched_any_rule = True
        target_override: float | None = None
        if rule.target is None and rule.usl is None and rule.lsl is None and all(row.usl is None and row.lsl is None for row in matching_rows):
            actual_values = [row.latest for row in matching_rows if row.latest is not None]
            if actual_values:
                try:
                    target_override = float(median(actual_values))
                except StatisticsError:
                    target_override = float(actual_values[-1])
        for row in matching_rows:
            score = _score_measurement(rule, row, target_override=target_override)
            if score is None:
                return None
            measurement_scores.append(score)

    if not matched_any_rule or not measurement_scores:
        return None

    pass_count = sum(1 for item in candidate_runs if item.get("test_result") == 1)
    fail_count = sum(1 for item in candidate_runs if item.get("test_result") == 0)
    retest_count = max(len(candidate_runs) - 1, 0)

    test_date_str = latest_entry.get("test_date")
    test_duration = _to_float(latest_entry.get("test_duration"))
    device_id = latest_entry.get("device_id")
    device_name = latest_entry.get("device_id__name") or latest_entry.get("device")
    dut_isn_val = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn")
    dut_id_val = latest_entry.get("dut_id")

    score_value = sum(m.score_value for m in measurement_scores) / max(len(measurement_scores), 1)
    test_count = len(candidate_runs)
    trimmed_data = _compress_station_data(station_payload, run_index)

    return StationEvaluation(
        station_id=station_entry.get("id"),
        station_name=station_entry.get("name") or "Unknown Station",
        dut_isn=dut_isn_val or "",
        dut_id=dut_id_val,
        device_id=device_id,
        device_name=device_name,
        test_date=_parse_test_date(test_date_str),
        test_duration=test_duration,
        error_item=latest_entry.get("error_item"),
        pass_count=pass_count,
        fail_count=fail_count,
        retest_count=retest_count,
        test_count=test_count,
        score=score_value,
        status=latest_entry.get("test_result"),
        order=station_entry.get("order"),
        data=trimmed_data,
        measurements=measurement_scores,
        site_name=site_name,
        project_name=station_entry.get("model_name") or model_name,
        model_name=model_name,
        metadata=None,
    )


def _store_top_products(db: Session | None, evaluations: Iterable[StationEvaluation]) -> None:
    if db is None:
        return
    bind = db.get_bind()
    if bind is None:
        return
    try:
        Base.metadata.create_all(bind=bind)
    except Exception as exc:  # pragma: no cover - metadata creation errors are logged
        logger.warning("Failed to ensure top product tables exist: %s", exc)
    try:
        for evaluation in evaluations:
            test_date = evaluation.test_date
            if isinstance(test_date, datetime) and test_date.tzinfo is None:
                test_date = test_date.replace(tzinfo=UTC)
            record = TopProduct(
                dut_isn=evaluation.dut_isn,
                dut_id=evaluation.dut_id,
                site_name=evaluation.site_name,
                project_name=evaluation.project_name,
                model_name=evaluation.model_name,
                station_name=evaluation.station_name,
                device_name=evaluation.device_name,
                test_date=test_date,
                test_duration=evaluation.test_duration,
                pass_count=evaluation.pass_count,
                fail_count=evaluation.fail_count,
                retest_count=evaluation.retest_count,
                score=evaluation.score,
            )
            db.add(record)
            db.flush()
            for measurement in evaluation.measurements:
                db.add(
                    TopProductMeasurement(
                        top_product_id=record.id,
                        test_item=measurement.test_item,
                        usl=measurement.usl,
                        lsl=measurement.lsl,
                        target_value=measurement.target,
                        actual_value=measurement.actual,
                        deviation=measurement.deviation,
                    )
                )
        db.commit()
    except SQLAlchemyError as exc:  # pragma: no cover - database persistence errors are logged
        db.rollback()
        logger.warning("Failed to persist top product results: %s", exc)


def _build_station_result_schema(evaluation: StationEvaluation) -> TopProductStationSummarySchema:
    measurement_rows: list[dict[str, str | float | None | dict]] = []
    for measurement in evaluation.measurements:
        # UPDATED: New dict format with named fields
        score_breakdown = measurement.score_breakdown or {}
        if measurement.target is not None and score_breakdown:
            # Ensure target is in breakdown (scoring functions use "target_used" key)
            if "target_used" not in score_breakdown:
                score_breakdown["target_used"] = measurement.target

        measurement_rows.append({"test_item": measurement.test_item, "usl": measurement.usl, "lsl": measurement.lsl, "actual": measurement.actual, "score_breakdown": score_breakdown if score_breakdown else None})

    return TopProductStationSummarySchema(
        station_id=evaluation.station_id,
        station_name=evaluation.station_name,
        dut_id=evaluation.dut_id,
        isn=evaluation.dut_isn,
        device_id=evaluation.device_id,
        device=evaluation.device_name,
        test_date=evaluation.test_date,
        status=evaluation.status,
        order=evaluation.order,
        test_duration=evaluation.test_duration,
        test_count=evaluation.test_count,
        pass_count=evaluation.pass_count,
        fail_count=evaluation.fail_count,
        error_item=evaluation.error_item,
        data=measurement_rows or evaluation.data,
        overall_data_score=evaluation.score,
        metadata=evaluation.metadata or {},
    )


def _build_top_product_candidates(
    evaluations: Iterable[StationEvaluation],
    required_station_names: set[str],
) -> list[tuple[float, list[StationEvaluation]]]:
    grouped: dict[str, list[StationEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        if not evaluation.dut_isn:
            continue
        grouped[_normalize_str(evaluation.dut_isn)].append(evaluation)

    candidates: list[tuple[float, list[StationEvaluation]]] = []
    for evals in grouped.values():
        coverage = {_normalize_str(entry.station_name) for entry in evals}
        if required_station_names and coverage != required_station_names:
            continue
        sorted_evals = sorted(
            evals,
            key=lambda entry: _order_sort_components(entry, name_getter=lambda value: value.station_name),
        )
        total_score = sum(entry.score for entry in sorted_evals)
        average_score = total_score / max(len(sorted_evals), 1)
        candidates.append((average_score, sorted_evals))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _build_station_top_products(evaluations: Iterable[StationEvaluation]) -> list[TopProductStationSummarySchema]:
    sorted_evals = sorted(
        evaluations,
        key=lambda entry: _order_sort_components(entry, name_getter=lambda value: value.station_name) + (-entry.score,),
    )
    return [_build_station_result_schema(entry) for entry in sorted_evals]


async def _enrich_evaluations_with_pa_adjusted_power(
    client: DUTAPIClient,
    evaluations: list[StationEvaluation],
    start_time: datetime,
    end_time: datetime,
    model_hint: str | None = None,
) -> None:
    """
    Enrich station evaluations with PA adjusted power measurements.

    This function:
    1. Detects PA SROM test items in each evaluation's measurements
    2. Fetches PA trend data for the time window
    3. Calculates adjusted power values: (NEW - OLD) / 256
    4. Creates virtual "PA_ADJUSTED_POWER" measurement scores
    5. Appends them to the evaluation's measurements list
    6. Recalculates the overall score including PA measurements

    PA adjusted power is scored based on deviation from 0 (ideal value),
    with no USL/LSL constraints.

    Args:
        client: DUT API client for fetching PA trend data
        evaluations: List of station evaluations to enrich
        start_time: Start of the evaluation time window
        end_time: End of the evaluation time window
        model_hint: Optional model identifier for PA trend queries
    """
    for evaluation in evaluations:
        # Extract PA test items from current measurements
        measurement_rows = [
            MeasurementRow(
                name=m.test_item,
                usl=m.usl,
                lsl=m.lsl,
                latest=m.actual,
            )
            for m in evaluation.measurements
        ]

        pa_test_items, base_name_mapping = _extract_pa_test_items_from_measurements(measurement_rows)

        if not pa_test_items:
            # No PA test items in this station, skip
            continue

        # Fetch and calculate PA adjusted power
        if evaluation.station_id is None:
            logger.debug("Skipping PA enrichment for %s: missing station_id", evaluation.station_name)
            continue

        adjusted_power_values = await _fetch_and_calculate_pa_adjusted_power(
            client=client,
            station_id=evaluation.station_id,
            pa_test_items=pa_test_items,
            start_time=start_time,
            end_time=end_time,
            model_hint=model_hint,
        )

        if not adjusted_power_values:
            logger.debug("No PA adjusted power values calculated for station %s", evaluation.station_name)
            continue

        # Create virtual PA adjusted power measurements and score them
        pa_measurements: list[MeasurementScore] = []
        for base_name, value_dict in adjusted_power_values.items():
            # Create virtual test items for both MID and MEAN if available
            # Format: WiFi_PA{n}_ADJUSTED_POWER_MID_<frequency>_<protocol>_<mode>
            #         WiFi_PA{n}_ADJUSTED_POWER_MEAN_<frequency>_<protocol>_<mode>

            if "mid" in value_dict:
                # Replace PA{n}_ with PA{n}_ADJUSTED_POWER_MID_
                virtual_test_item_mid = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POWER_MID_", base_name, count=1, flags=re.IGNORECASE)
                deviation_mid, score_mid, breakdown_mid = _calculate_pa_adjusted_power_score(value_dict["mid"])

                pa_measurement_mid = MeasurementScore(
                    test_item=virtual_test_item_mid,
                    usl=None,  # No USL for PA adjusted power
                    lsl=None,  # No LSL for PA adjusted power
                    target=0.0,  # Ideal PA adjusted power is 0
                    actual=value_dict["mid"],
                    deviation=deviation_mid,
                    score_value=score_mid,
                )
                pa_measurements.append(pa_measurement_mid)
                logger.debug(
                    "Added PA adjusted power (MID) measurement for %s: value=%.2f, score=%.2f",
                    virtual_test_item_mid,
                    value_dict["mid"],
                    score_mid,
                )

            if "mean" in value_dict:
                # Replace PA{n}_ with PA{n}_ADJUSTED_POWER_MEAN_
                virtual_test_item_mean = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POWER_MEAN_", base_name, count=1, flags=re.IGNORECASE)
                deviation_mean, score_mean, breakdown_mean = _calculate_pa_adjusted_power_score(value_dict["mean"])

                pa_measurement_mean = MeasurementScore(
                    test_item=virtual_test_item_mean,
                    usl=None,  # No USL for PA adjusted power
                    lsl=None,  # No LSL for PA adjusted power
                    target=0.0,  # Ideal PA adjusted power is 0
                    actual=value_dict["mean"],
                    deviation=deviation_mean,
                    score_value=score_mean,
                )
                pa_measurements.append(pa_measurement_mean)
                logger.debug(
                    "Added PA adjusted power (MEAN) measurement for %s: value=%.2f, score=%.2f",
                    virtual_test_item_mean,
                    value_dict["mean"],
                    score_mean,
                )

        # Append PA measurements to evaluation
        evaluation.measurements.extend(pa_measurements)

        # Recalculate overall score including PA measurements
        if evaluation.measurements:
            total_score = sum(m.score_value for m in evaluation.measurements)
            evaluation.score = total_score / len(evaluation.measurements)
            logger.debug(
                "Updated score for %s after PA enrichment: %.2f (from %d measurements, %d PA)",
                evaluation.station_name,
                evaluation.score,
                len(evaluation.measurements),
                len(pa_measurements),
            )


async def _evaluate_top_products(
    client: DUTAPIClient,
    dut_isn: str,
    criteria_map: dict[str, list[CriteriaRule]],
    selected_station_names: set[str],
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    device_filters: set[str] | None = None,
    model_hint: str | None = None,
    site_hint: str | None = None,
) -> tuple[list[StationEvaluation], str | None, str | None, set[str]]:
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    if not record_data:
        return [], site_name, model_name, set()

    if site_hint and site_name and _normalize_str(site_name) != site_hint:
        raise HTTPException(status_code=404, detail="Requested site identifier does not match DUT records")
    if model_hint and model_name and _normalize_str(model_name) != model_hint:
        raise HTTPException(status_code=404, detail="Requested model identifier does not match DUT records")

    normalized_selected = selected_station_names or set(criteria_map.keys())
    evaluated_station_names: set[str] = set()
    evaluations: list[StationEvaluation] = []
    station_payload_cache: dict[tuple[int | None, int | None], StationRecordResponseSchema] = {}

    for station_entry in record_data:
        station_name = station_entry.get("name") or "Unknown Station"
        normalized_name = _normalize_str(station_name)
        if normalized_selected and normalized_name not in normalized_selected:
            continue
        station_model_name = station_entry.get("model_name")
        if model_hint and station_model_name and _normalize_str(station_model_name) != model_hint:
            continue
        station_site_name = station_entry.get("site_name")
        if site_hint and station_site_name and _normalize_str(station_site_name) != site_hint:
            continue
        rules = criteria_map.get(normalized_name)
        if not rules:
            continue
        candidate_groups = _extract_candidate_runs(station_entry, start_time, end_time)
        if not candidate_groups:
            continue
        evaluated_station_names.add(normalized_name)
        for (candidate_dut_id, _), runs in candidate_groups.items():
            runs.sort(key=lambda item: _parse_test_date(item.get("test_date")) or datetime.min)
            latest_entry = runs[-1]
            if device_filters:
                if not _matches_device_filters(
                    device_filters,
                    latest_entry.get("device_id"),
                    latest_entry.get("device_id__name") or latest_entry.get("device"),
                ):
                    continue
            resolved_station_id = station_entry.get("id")
            candidate_dut_id_value = latest_entry.get("dut_id") or candidate_dut_id
            dut_identifier = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn") or dut_isn

            if resolved_station_id is None or candidate_dut_id_value is None:
                station_identifier = station_name if resolved_station_id is None else str(resolved_station_id)
                try:
                    resolved_station_id, candidate_dut_id_value = await _resolve_station_and_dut_ids(
                        client,
                        str(station_identifier),
                        str(dut_identifier),
                        site_hint=site_hint,
                        model_hint=model_hint,
                    )
                except HTTPException:
                    continue

            cache_key = (resolved_station_id, candidate_dut_id_value)
            station_payload = station_payload_cache.get(cache_key)
            if station_payload is None:
                try:
                    payload = await client.get_station_records(resolved_station_id, candidate_dut_id_value)
                    station_payload = StationRecordResponseSchema.model_validate(payload)
                    station_payload_cache[cache_key] = station_payload
                except Exception as exc:
                    logger.warning(
                        "Unable to retrieve station records for station %s (%s) and dut %s: %s",
                        station_name,
                        resolved_station_id,
                        candidate_dut_id_value,
                        exc,
                    )
                    continue

            evaluation = _evaluate_station_candidate(
                station_entry,
                station_payload,
                latest_entry,
                runs,
                rules,
                site_name,
                model_name,
            )
            if evaluation and evaluation.dut_isn:
                evaluations.append(evaluation)

    # UPDATED: Enrich evaluations with PA adjusted power measurements
    if evaluations and start_time and end_time:
        await _enrich_evaluations_with_pa_adjusted_power(
            client=client,
            evaluations=evaluations,
            start_time=start_time,
            end_time=end_time,
            model_hint=model_hint,
        )

    return evaluations, site_name, model_name, evaluated_station_names


def _station_matches_filters(filters: dict, station_id: int | None, station_name: str | None) -> bool:
    if not filters:
        return True
    if station_id is not None and str(station_id) in filters["numeric"]:
        return True
    if station_name and _normalize_str(station_name) in filters["normalized"]:
        return True
    return False


def _parse_input_datetime(value: str) -> datetime:
    if value is None:
        raise ValueError("datetime value is required")
    candidate = value.strip()
    if not candidate:
        raise ValueError("datetime value is required")
    candidate = candidate.replace("z", "Z")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    dt = datetime.fromisoformat(candidate)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    trimmed = value.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return trimmed


def _validate_time_window(start_dt: datetime, end_dt: datetime) -> None:
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    if end_dt - start_dt > _MAX_TOP_PRODUCT_WINDOW:
        raise HTTPException(status_code=400, detail="Time window cannot exceed 7 days")


def _parse_score_criteria(raw: str) -> tuple[float, float | None]:
    if raw is None:
        raise HTTPException(status_code=400, detail="criteria_score is required")
    text = raw.strip()
    if not text:
        raise HTTPException(status_code=400, detail="criteria_score is required")
    match = _SCORE_CRITERIA_PATTERN.match(text)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid criteria_score format. Use <min> or <min>-<max>.")
    lower = float(match.group("min"))
    upper_value = match.group("max")
    upper = float(upper_value) if upper_value is not None else None
    if upper is not None and upper < lower:
        raise HTTPException(status_code=400, detail="criteria_score upper bound must be greater than or equal to the lower bound.")
    return lower, upper


async def _fallback_station_device_results(
    client: DUTAPIClient,
    station_id: int,
    station_name: str,
    start_dt: datetime,
    end_dt: datetime,
    device_filters: set[str] | None,
    model_hint: str | None,
) -> list[dict[str, Any]]:
    try:
        devices = await _get_cached_station_devices(client, station_id)
    except HTTPException:
        return []
    except Exception as exc:
        logger.debug("Unable to fetch cached devices for station %s during fallback: %s", station_id, exc)
        return []
    if not devices:
        return []

    start_iso = _format_external_timestamp(start_dt)
    end_iso = _format_external_timestamp(end_dt)
    station_norm = _normalize_str(station_name)
    fallback_entries: list[dict[str, Any]] = []

    for device in devices:
        device_id = _coerce_optional_int(device.get("id") or device.get("device_id"))
        if device_id is None:
            continue
        device_name = device.get("device_name") or device.get("name") or device.get("device")
        if device_filters and not _matches_device_filters(device_filters, device_id, device_name):
            continue
        payload: dict[str, Any] = {
            "device_id": str(device_id),
            "start_time": start_iso,
            "end_time": end_iso,
            "test_result": "ALL",
        }
        if model_hint:
            payload["model"] = model_hint
        try:
            records = await client.get_test_results_by_device(payload)
        except Exception as exc:
            logger.debug(
                "Fallback device history query failed for station %s device %s: %s",
                station_id,
                device_id,
                exc,
            )
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            record_station = record.get("station_id__name")
            if record_station and _normalize_str(record_station) != station_norm:
                continue
            fallback_entries.append(
                {
                    "test_date": record.get("test_date"),
                    "device_id": device_id,
                    "device_id__name": device_name,
                    "device": device_name,
                    "dut_id": record.get("dut_id"),
                    "dut_id__isn": record.get("dut_id__isn"),
                    "dut_isn": record.get("dut_id__isn"),
                    "test_result": record.get("test_result"),
                    "error_item": record.get("error_item"),
                    "test_duration": record.get("test_duration"),
                }
            )
    return fallback_entries


def _extract_upstream_error_detail(response: httpx.Response | None) -> str:
    if response is None:
        return "Upstream request failed"
    try:
        payload = response.json()
        if isinstance(payload, dict):
            for key in ("detail", "message", "error"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    except ValueError:
        pass
    text = response.text.strip()
    return text or "Upstream request failed"


def _format_external_timestamp(dt: datetime) -> str:
    dt_utc = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    return dt_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def _fetch_pa_trend_measurements(
    client: DUTAPIClient,
    station_id: int,
    dut_id: int,
    dut_isn: str,
    test_date: datetime | None,
    site_hint: str | None = None,
    model_hint: str | None = None,
) -> list[dict[str, str | float | None | dict]]:
    """
    Fetch PA adjusted power scored measurements comparing current values against trend data.

    This function fetches PA ADJUSTED_POW test items (calculated from SROM_NEW - SROM_OLD / 256)
    and scores them based on deviation from 24-hour trend mean:
    - Current: PA adjusted power from latest nonvalue record
    - Trend: 24-hour historical mean from PA trend API
    - Scoring: Linear scoring where closer to trend = higher score (max 10.0)
    - Formula: score = 10.0 * (1 - |current - trend_mean| / threshold)

    Args:
        client: DUT API client
        station_id: Station ID to fetch PA items from
        dut_id: DUT ID for the station
        dut_isn: DUT ISN identifier for trend API
        test_date: Test date to determine time window (defaults to 24h window)
        site_hint: Optional site identifier (NOT sent to PA endpoints)
        model_hint: Optional model identifier (NOT sent to PA endpoints)

    Returns:
        List of measurement dicts in format: {"test_item": name, "usl": None, "lsl": None, "actual": value, "score_breakdown": {...}}
        Each dict represents a PA ADJUSTED_POW test item with score_breakdown containing:
        - category: "PA Adjusted Power Trend Comparison"
        - method: "Linear scoring based on deviation from trend mean"
        - target_used: PA trend mean value (24h average)
        - current_value: Current PA adjusted power value
        - deviation_from_mean: current - trend_mean
        - score: Proximity score (0-10, where 10 = perfect match)
        - interpretation: Human-readable score interpretation

    Note:
        - Uses the same scoring logic as /api/dut/records/nonvalue2/scored/{station_id}/{dut_id}
        - Threshold: 5.0 dB (configurable)
        - Score interpretation:
          * 10.0: Perfect match (deviation < 0.1)
          * 9.0-10.0: Excellent (deviation < 1.0)
          * 7.0-9.0: Good (deviation < 2.5)
          * 5.0-7.0: Acceptable (deviation < 5.0)
          * 0.0: Out of range (deviation ≥ 5.0)

    Returns empty list if:
    - No PA SROM items found
    - PA trend or nonvalue API calls fail
    - No matching items between trend and nonvalue data
    """
    # Determine time window for PA trend calculation (24 hours from test_date)
    if test_date is None:
        end_time = datetime.now(UTC)
    else:
        end_time = test_date.astimezone(UTC) if test_date.tzinfo else test_date.replace(tzinfo=UTC)

    start_time = end_time - timedelta(hours=24)

    # Step 1: Fetch current PA adjusted power values from nonvalue record
    try:
        nonvalue_payload = await client.get_latest_nonvalue_record(station_id, dut_id)
        validated_nonvalue = StationRecordResponseSchema.model_validate(nonvalue_payload)
    except Exception as exc:
        logger.debug(
            "Failed to fetch nonvalue records for PA scoring (station=%s, dut=%s): %s",
            station_id,
            dut_id,
            exc,
        )
        return []

    # Process PA SROM data to get adjusted power values
    current_data, has_pa_srom = _process_pa_srom_enhanced(validated_nonvalue)

    if not has_pa_srom:
        logger.debug("No PA SROM test items found for station %s, dut %s", station_id, dut_id)
        return []

    # Extract current adjusted power values
    current_adjusted_power = {}  # {test_item: value}
    for item in current_data:
        if isinstance(item, PAAdjustedPowerDataItemSchema):
            current_adjusted_power[item.test_item] = item.adjusted_value

    if not current_adjusted_power:
        logger.debug("No PA adjusted power values calculated for station %s, dut %s", station_id, dut_id)
        return []

    # Step 2: Fetch PA SROM test items for trend API
    pa_srom_test_items = []
    for row in validated_nonvalue.data:
        if not isinstance(row, list) or len(row) < 1:
            continue
        test_item_name = str(row[0])
        if _is_pa_srom_test_item(test_item_name, "all"):
            pa_srom_test_items.append(test_item_name)

    if not pa_srom_test_items:
        logger.debug("No PA SROM test items found for station %s", station_id)
        return []

    # Step 3: Fetch PA trend data
    start_time_str = start_time.isoformat().replace("+00:00", "Z")
    end_time_str = end_time.isoformat().replace("+00:00", "Z")

    trend_payload = {
        "start_time": start_time_str,
        "end_time": end_time_str,
        "station_id": station_id,
        "test_items": pa_srom_test_items,
        "model": "",  # Empty string as per PA trend API requirements
    }

    try:
        trend_data = await client.get_pa_test_items_trend(trend_payload)
    except Exception as exc:
        logger.warning("Failed to fetch PA trend data for station %s: %s", station_id, exc)
        return []

    if not trend_data:
        logger.debug("No PA trend data available for station %s", station_id)
        return []

    # Step 4: Calculate trend adjusted power from SROM pairs
    old_trend = {}
    new_trend = {}

    for test_item_name, trend_values in trend_data.items():
        if _is_pa_srom_test_item(test_item_name, "old"):
            base_key = re.sub(r"_SROM_OLD_", "_SROM_", test_item_name, flags=re.IGNORECASE)
            old_trend[base_key] = trend_values
        elif _is_pa_srom_test_item(test_item_name, "new"):
            base_key = re.sub(r"_SROM_NEW_", "_SROM_", test_item_name, flags=re.IGNORECASE)
            new_trend[base_key] = trend_values

    # Calculate adjusted power from trend pairs
    trend_adjusted_power = {}
    for base_key in set(old_trend.keys()) & set(new_trend.keys()):
        old_vals = old_trend[base_key]
        new_vals = new_trend[base_key]

        adjusted_item_name = re.sub(r"_SROM_", "_ADJUSTED_POW_", base_key, flags=re.IGNORECASE)

        mean_old = old_vals.get("mean")
        mean_new = new_vals.get("mean")

        if mean_old is not None and mean_new is not None:
            trend_adjusted_power[adjusted_item_name] = round((mean_new - mean_old) / 256, 2)

    # Step 5: Calculate scores by comparing current vs trend
    measurement_rows = []
    threshold = 5.0  # Maximum acceptable deviation

    for test_item, current_value in current_adjusted_power.items():
        trend_mean = trend_adjusted_power.get(test_item)

        if trend_mean is None:
            continue

        # Calculate deviation from trend mean
        deviation_from_mean = current_value - trend_mean
        abs_deviation = abs(deviation_from_mean)

        # Calculate score: closer to trend mean = higher score
        if abs_deviation >= threshold:
            score = 0.0
        else:
            score = 10.0 * (1.0 - abs_deviation / threshold)
            score = max(0.0, min(10.0, score))

        score = round(score, 2)

        # Create detailed breakdown
        breakdown = {
            "category": "PA Adjusted Power Trend Comparison",
            "method": "Linear scoring based on deviation from trend mean",
            "comparison": "current vs trend_mean",
            "threshold": threshold,
            "target_used": round(trend_mean, 2),
            "current_value": round(current_value, 2),
            "trend_mean": round(trend_mean, 2),
            "deviation_from_mean": round(deviation_from_mean, 2),
            "abs_deviation": round(abs_deviation, 2),
            "raw_score": score,
            "final_score": score,
            "formula_latex": r"$\text{Score} = 10 \times \left(1 - \frac{|\text{current} - \text{trend}|}{\text{threshold}}\right)$",
            "interpretation": (
                "Perfect match (0.0 dev)" if abs_deviation < 0.1 else "Excellent match (<1.0 dev)" if abs_deviation < 1.0 else "Good match (<2.5 dev)" if abs_deviation < 2.5 else "Acceptable (<5.0 dev)" if abs_deviation < 5.0 else "Out of range (≥5.0 dev)"
            ),
        }

        measurement_rows.append({"test_item": test_item, "usl": None, "lsl": None, "actual": current_value, "score_breakdown": breakdown})

    # Sort by frequency first, then PA number (so same frequency shows PA1, PA2, PA3, PA4)
    def pa_sort_key(row: dict) -> tuple:
        test_item = str(row.get("test_item", ""))
        # Extract PA number (1-4)
        pa_match = re.search(r"PA([1-4])", test_item, re.IGNORECASE)
        pa_num = int(pa_match.group(1)) if pa_match else 999

        # Extract frequency (e.g., 5985, 6275)
        freq_match = re.search(r"_(\d{4})_", test_item)
        freq = int(freq_match.group(1)) if freq_match else 0

        # Sort by frequency first, then PA number
        return (freq, pa_num)

    measurement_rows.sort(key=pa_sort_key)

    logger.info(
        "Generated %d PA adjusted power scored measurements for station %s (current vs 24h trend comparison)",
        len(measurement_rows),
        station_id,
    )

    return measurement_rows


async def _compute_top_product_response(
    client: DUTAPIClient,
    db: Session,
    dut_isn: str,
    *,
    stations: list[str] | None,
    site_identifier: str | None,
    model_identifier: str | None,
    device_identifiers: list[str] | None,
    test_item_filters: list[str] | None,
    exclude_test_item_filters: list[str] | None,
    station_filters_map: dict[str, StationFilterConfigSchema] | None,  # UPDATED: Added per-station filters
    criteria_rules: dict[str, list[CriteriaRule]] | None,
    criteria_label: str,
    include_pa_trends: bool = False,
) -> TopProductResponseSchema:
    site_name, model_name, record_data = await _fetch_dut_records(client, dut_isn)
    if not record_data:
        raise HTTPException(status_code=404, detail="No test records found for the provided DUT")

    site_hint_value = _normalize_optional_string(site_identifier)
    site_hint_norm = _normalize_str(site_hint_value) if site_hint_value else None
    site_id_hint = None
    if site_identifier and site_identifier.strip().isdigit():
        try:
            site_id_hint = int(site_identifier.strip())
        except ValueError:
            site_id_hint = None

    model_hint_value = _normalize_optional_string(model_identifier)
    model_hint_norm = _normalize_str(model_hint_value) if model_hint_value else None
    model_id_hint = None
    if model_identifier and model_identifier.strip().isdigit():
        try:
            model_id_hint = int(model_identifier.strip())
        except ValueError:
            model_id_hint = None

    device_filters: set[str] | None = None
    if device_identifiers:
        normalized_devices = {_normalize_str(value) for value in device_identifiers if value}
        if normalized_devices:
            device_filters = normalized_devices

    working_criteria_rules = _clone_criteria_rules(criteria_rules)
    default_rules = [CriteriaRule(pattern=re.compile(".*"), usl=None, lsl=None, target=None)]

    station_filters = _prepare_station_filters(stations)
    dut_isn_norm = _normalize_str(dut_isn)
    station_payload_cache: dict[tuple[int, int], StationRecordResponseSchema] = {}
    evaluations: list[StationEvaluation] = []
    response_rows: list[TopProductStationSummarySchema] = []
    include_patterns, include_tokens, include_signatures, include_is_exact = _compile_test_item_patterns(test_item_filters)
    exclude_patterns, exclude_tokens, exclude_signatures, exclude_is_exact = _compile_test_item_patterns(exclude_test_item_filters)

    # OPTIMIZATION: Pre-compile station-specific patterns to avoid recompilation in loop
    station_pattern_cache: dict[str, tuple[list[re.Pattern[str]], list[list[str]], list[TestItemSignature | None], list[bool]]] = {}
    if station_filters_map:
        for station_key, config in station_filters_map.items():
            cache_key_include = f"{station_key}__include"
            cache_key_exclude = f"{station_key}__exclude"

            if config.test_item_filters:
                station_pattern_cache[cache_key_include] = _compile_test_item_patterns(config.test_item_filters)
            if config.exclude_test_item_filters:
                station_pattern_cache[cache_key_exclude] = _compile_test_item_patterns(config.exclude_test_item_filters)

    for station_entry in record_data:
        station_id = _coerce_optional_int(station_entry.get("id"))
        station_name_entry = station_entry.get("name") or "Unknown Station"

        if station_filters and not _station_matches_filters(station_filters, station_id, station_name_entry):
            continue

        # UPDATED: Get per-station filters if available
        station_specific_config = None
        if station_filters_map:
            # Try to match by station ID or name
            station_id_str = str(station_id) if station_id else None
            station_specific_config = station_filters_map.get(station_name_entry) or (station_filters_map.get(station_id_str) if station_id_str else None)

            # Log station filter application for debugging
            if station_specific_config:
                logger.debug(
                    "Applying station-specific filters for DUT %s, station '%s' (ID: %s): include=%s, exclude=%s",
                    dut_isn,
                    station_name_entry,
                    station_id,
                    station_specific_config.test_item_filters,
                    station_specific_config.exclude_test_item_filters,
                )
            else:
                logger.debug(
                    "No station-specific filters found for DUT %s, station '%s' (ID: %s). Available stations in map: %s",
                    dut_isn,
                    station_name_entry,
                    station_id,
                    list(station_filters_map.keys()),
                )

        # UPDATED: Build per-station device filters (prioritize station-specific over global)
        station_device_filters = device_filters
        if station_specific_config and station_specific_config.device_identifiers:
            normalized_station_devices = {_normalize_str(value) for value in station_specific_config.device_identifiers if value}
            if normalized_station_devices:
                station_device_filters = normalized_station_devices

        # UPDATED: Build per-station test item filters (prioritize station-specific over global)
        # Use cached patterns to avoid recompilation
        station_include_patterns, station_include_tokens, station_include_signatures, station_include_is_exact = (include_patterns, include_tokens, include_signatures, include_is_exact)
        station_exclude_patterns, station_exclude_tokens, station_exclude_signatures, station_exclude_is_exact = (exclude_patterns, exclude_tokens, exclude_signatures, exclude_is_exact)

        if station_specific_config:
            # Lookup station identifier for cache key
            station_cache_id = station_name_entry if station_name_entry else (str(station_id) if station_id else None)
            if station_cache_id:
                cache_key_include = f"{station_cache_id}__include"
                cache_key_exclude = f"{station_cache_id}__exclude"

                if station_specific_config.test_item_filters and cache_key_include in station_pattern_cache:
                    station_include_patterns, station_include_tokens, station_include_signatures, station_include_is_exact = station_pattern_cache[cache_key_include]
                if station_specific_config.exclude_test_item_filters and cache_key_exclude in station_pattern_cache:
                    station_exclude_patterns, station_exclude_tokens, station_exclude_signatures, station_exclude_is_exact = station_pattern_cache[cache_key_exclude]

        if model_id_hint is not None:
            entry_model_id = _coerce_optional_int(station_entry.get("model_id"))
            if entry_model_id is not None and entry_model_id != model_id_hint:
                continue
        elif model_hint_norm:
            entry_model_name = station_entry.get("model_name")
            if entry_model_name and _normalize_str(entry_model_name) != model_hint_norm:
                continue

        if site_id_hint is not None:
            entry_site_id = _coerce_optional_int(station_entry.get("site_id"))
            if entry_site_id is not None and entry_site_id != site_id_hint:
                continue
        elif site_hint_norm:
            entry_site_name = station_entry.get("site_name")
            if entry_site_name and _normalize_str(entry_site_name) != _normalize_str(site_hint_norm):
                continue

        station_rules: list[CriteriaRule]
        if working_criteria_rules is None:
            # No criteria file uploaded - use default rules to include all stations
            station_rules = default_rules
        else:
            # Criteria file uploaded - only include stations with matching rules
            station_rules = _select_station_criteria(working_criteria_rules, station_name_entry, model_name)
            logger.info(
                "Criteria check for station='%s' model='%s': found %d rules (total criteria stations: %d)",
                station_name_entry,
                model_name,
                len(station_rules),
                len(working_criteria_rules),
            )
            if not station_rules:
                # No matching criteria for this station - skip it
                logger.debug(
                    "Skipping station '%s' (model: '%s') - no matching criteria rules found. Available keys: %s",
                    station_name_entry,
                    model_name,
                    list(working_criteria_rules.keys()),
                )
                continue

        data_entries = station_entry.get("data") or []
        if not data_entries:
            continue

        candidate_runs = [item for item in data_entries if _matches_dut_isn(dut_isn_norm, station_entry, item)]
        if not candidate_runs:
            candidate_runs = data_entries

        candidate_runs.sort(
            key=lambda entry: _parse_test_date(entry.get("test_date")) or datetime.min,
        )
        latest_entry = candidate_runs[-1]

        # UPDATED: Use station-specific device filters
        if station_device_filters and not _matches_device_filters(
            station_device_filters,
            latest_entry.get("device_id"),
            latest_entry.get("device_id__name") or latest_entry.get("device"),
        ):
            continue

        resolved_station_id = station_id
        resolved_dut_id = latest_entry.get("dut_id")
        if resolved_station_id is None or resolved_dut_id is None:
            station_identifier = station_name_entry if resolved_station_id is None else str(resolved_station_id)
            dut_identifier = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn") or dut_isn
            try:
                resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    str(station_identifier),
                    str(dut_identifier),
                )
            except HTTPException:
                continue

        cache_key = (resolved_station_id, resolved_dut_id)
        station_payload = station_payload_cache.get(cache_key)
        if station_payload is None:
            try:
                payload = await client.get_station_records(resolved_station_id, resolved_dut_id)
                station_payload = StationRecordResponseSchema.model_validate(payload)
                station_payload_cache[cache_key] = station_payload
            except Exception as exc:
                logger.warning(
                    "Unable to retrieve station records for station %s (%s) and dut %s: %s",
                    station_name_entry,
                    resolved_station_id,
                    resolved_dut_id,
                    exc,
                )
                continue

        run_index = _find_run_index(station_payload, latest_entry)
        measurement_matrix = _compress_station_data(station_payload, run_index)

        # FEATURE: PA trend measurement integration (optional)
        # Fetch PA trend measurements only if explicitly requested via include_pa_trends flag
        if include_pa_trends:
            try:
                entry_test_date = _parse_test_date(latest_entry.get("test_date"))
                entry_dut_isn = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn") or latest_entry.get("isn") or dut_isn
                pa_measurements = await _fetch_pa_trend_measurements(
                    client=client,
                    station_id=resolved_station_id,
                    dut_id=resolved_dut_id,
                    dut_isn=entry_dut_isn,
                    test_date=entry_test_date if entry_test_date != datetime.min else None,
                    site_hint=site_identifier,
                    model_hint=model_identifier,
                )
                if pa_measurements:
                    logger.debug(
                        "Adding %d PA trend measurements to station %s (%s)",
                        len(pa_measurements),
                        station_name_entry,
                        resolved_station_id,
                    )
                    # Extend measurement matrix with PA items
                    # Format: [test_item, null, null, nonvalue_actual, trend_target, score]
                    measurement_matrix.extend(pa_measurements)
            except Exception as exc:
                # Non-critical: If PA measurements fail, continue with regular measurements
                logger.warning(
                    "Failed to fetch PA trend measurements for station %s (%s): %s",
                    station_name_entry,
                    resolved_station_id,
                    exc,
                )

        measurement_rows: list[list[str | float | None]] = []
        measurement_scores: list[MeasurementScore] = []

        for idx, row in enumerate(measurement_matrix):
            # Handle both dict format (PA trends) and array format (regular measurements)
            if isinstance(row, dict):
                # PA trend measurement (already in dict format with breakdown)
                test_item = str(row.get("test_item", ""))
                usl = row.get("usl")
                lsl = row.get("lsl")
                actual = row.get("actual")
                breakdown = row.get("score_breakdown", {})
                # PA trends don't need criteria rules
                existing_score = breakdown.get("score") if breakdown else None
                target = breakdown.get("target_used") if breakdown else None

                if actual is None:
                    continue

                # Add PA trend item directly (already formatted)
                measurement_rows.append(row)

                # Create MeasurementScore for overall_score calculation
                score_value = existing_score if existing_score is not None else 0.0
                measurement_scores.append(
                    MeasurementScore(
                        test_item=test_item,
                        usl=usl,
                        lsl=lsl,
                        target=target,
                        actual=actual,
                        deviation=None,
                        score_value=score_value,
                        score_breakdown=breakdown,
                    )
                )
                continue

            if not isinstance(row, list) or len(row) < 4:
                continue
            test_item = str(row[0])
            usl = _to_float(row[1]) if len(row) > 1 else None
            lsl = _to_float(row[2]) if len(row) > 2 else None
            original_row = station_payload.data[idx] if idx < len(station_payload.data) else row
            actual_override, target_override, score_override = _extract_measurement_overrides(original_row, run_index)
            actual = actual_override if actual_override is not None else _to_float(row[3])
            existing_score = score_override if score_override is not None else (_to_float(row[5]) if len(row) > 5 else None)
            if actual is None:
                continue

            rule = _match_station_rule(station_rules, test_item)

            # FEATURE: Exempt PA adjusted power and POW_DIF_ABS measurements from criteria rule requirement
            # PA items are virtual test items that won't have criteria rules, but should always be included
            measurement_category = _detect_measurement_category(test_item)
            is_pa_adjusted_power = measurement_category == "PA_ADJUSTED_POWER"
            is_pow_dif_abs = measurement_category == "PA_POW_DIF_ABS"

            if working_criteria_rules is not None and rule is None and not (is_pa_adjusted_power or is_pow_dif_abs):
                continue

            # UPDATED: Use station-specific test item filters
            if (station_include_patterns or station_include_tokens or station_include_signatures) and not _pattern_matches(test_item, station_include_patterns, station_include_tokens, station_include_signatures, station_include_is_exact):
                continue
            if (station_exclude_patterns or station_exclude_tokens or station_exclude_signatures) and _pattern_matches(test_item, station_exclude_patterns, station_exclude_tokens, station_exclude_signatures, station_exclude_is_exact):
                continue

            target = target_override if target_override is not None else _determine_target_value(rule, usl, lsl, actual, test_item)
            display_usl = rule.usl if rule and rule.usl is not None else usl
            display_lsl = rule.lsl if rule.lsl is not None else lsl
            # UPDATED: Pass test_item for category-specific scoring (EVM, PER, POW_DIF_ABS)
            deviation, computed_score, breakdown = _calculate_measurement_metrics(display_usl, display_lsl, target, actual, test_item)
            # FEATURE: Always compute score for POW_DIF_ABS items (don't use existing_score from external API)
            # POW_DIF_ABS items from external API have score=0, but we need to calculate the actual score
            score_value = computed_score if is_pow_dif_abs else (existing_score if existing_score is not None else computed_score)

            # Ensure target_used is in breakdown
            if breakdown and "target_used" not in breakdown:
                breakdown["target_used"] = target

            measurement_rows.append({"test_item": test_item, "usl": display_usl, "lsl": display_lsl, "actual": actual, "score_breakdown": breakdown})
            measurement_scores.append(
                MeasurementScore(
                    test_item=test_item,
                    usl=display_usl,
                    lsl=display_lsl,
                    target=target,
                    actual=actual,
                    deviation=deviation,
                    score_value=score_value,
                    score_breakdown=breakdown,
                )
            )

        if not measurement_rows:
            continue

        overall_score = round(sum(score.score_value for score in measurement_scores) / max(len(measurement_scores), 1), 2)

        pass_count = sum(1 for run in candidate_runs if run.get("test_result") == 1)
        fail_count = sum(1 for run in candidate_runs if run.get("test_result") == 0)
        test_count = len(candidate_runs)

        device_id = _coerce_optional_int(latest_entry.get("device_id"))
        device_name = latest_entry.get("device_id__name") or latest_entry.get("device")
        test_date = _parse_test_date(latest_entry.get("test_date"))

        evaluation = StationEvaluation(
            station_id=resolved_station_id,
            station_name=station_name_entry,
            dut_isn=latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn") or dut_isn,
            dut_id=_coerce_optional_int(resolved_dut_id),
            device_id=device_id,
            device_name=device_name,
            test_date=test_date,
            test_duration=_to_float(latest_entry.get("test_duration")),
            error_item=latest_entry.get("error_item"),
            pass_count=pass_count,
            fail_count=fail_count,
            retest_count=max(test_count - 1, 0),
            test_count=test_count,
            score=overall_score,
            status=latest_entry.get("test_result"),
            order=_coerce_optional_int(station_entry.get("order")),
            data=measurement_rows,
            measurements=measurement_scores,
            site_name=site_name,
            project_name=station_entry.get("model_name") or model_name,
            model_name=model_name,
            metadata={},
        )
        evaluations.append(evaluation)

        response_rows.append(
            TopProductStationSummarySchema(
                station_id=resolved_station_id,
                station_name=station_name_entry,
                dut_id=evaluation.dut_id,
                isn=evaluation.dut_isn,
                device_id=device_id,
                device=device_name,
                test_date=test_date,
                status=evaluation.status,
                order=evaluation.order,
                test_duration=evaluation.test_duration,
                test_count=test_count,
                pass_count=pass_count,
                fail_count=fail_count,
                error_item=evaluation.error_item,
                data=measurement_rows,
                overall_data_score=overall_score,
                metadata={"measurement_count": len(measurement_rows)},
            )
        )

    if not response_rows:
        raise HTTPException(status_code=404, detail="No evaluation data available for the requested DUT")

    response_rows.sort(
        key=lambda entry: (
            1 if entry.order is None else 0,
            float("inf") if entry.order is None else float(entry.order),
            (entry.station_name or "").lower(),
        )
    )

    _store_top_products(db, evaluations)

    return TopProductResponseSchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        criteria_path=criteria_label,
        test_result=response_rows,
    )


def _compress_station_data(
    payload: StationRecordResponseSchema,
    run_index: int | None = None,
) -> list[list[str | int | float | None]]:
    compressed: list[list[str | int | float | None]] = []
    for row in payload.data:
        if not isinstance(row, list):
            continue
        if len(row) < 4:
            compressed.append(row)
            continue
        value_index = len(row) - 1
        if run_index is not None:
            candidate_index = 3 + run_index
            if candidate_index < len(row):
                value_index = candidate_index
        compressed.append(row[:3] + [row[value_index]])
    return compressed


def _get_entry_value(entry, key: str):
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _find_run_index(payload: StationRecordResponseSchema, entry: dict) -> int | None:
    if not payload.record:
        return None
    target_date = _parse_test_date(entry.get("test_date"))
    target_device = entry.get("device_id__name") or entry.get("device")
    target_isn = _normalize_str(entry.get("dut_id__isn") or entry.get("dut_isn"))
    matched_index = None
    for idx, record_entry in enumerate(payload.record):
        record_date = _parse_test_date(_get_entry_value(record_entry, "test_date"))
        record_device = _get_entry_value(record_entry, "device")
        record_isn = _normalize_str(_get_entry_value(record_entry, "isn") or _get_entry_value(record_entry, "dut_isn"))
        if target_date and record_date and record_date != target_date:
            continue
        if target_device and record_device and record_device != target_device:
            continue
        if target_isn and record_isn and record_isn != target_isn:
            continue
        matched_index = idx
    return matched_index


async def _resolve_site_id(client: DUTAPIClient, identifier: str) -> int:
    ident = str(identifier).strip()
    if ident.isdigit():
        return int(ident)

    try:
        sites = await _get_cached_sites(client)
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.error("Error fetching sites while resolving identifier '%s': %s", identifier, exc)
        raise HTTPException(status_code=503, detail="DUT service unavailable while retrieving site catalogue") from exc
    except Exception as exc:
        logger.error("Unexpected error fetching sites while resolving identifier '%s': %s", identifier, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve sites from external API") from exc

    for site in sites:
        if _normalize_str(site.get("name")) == _normalize_str(ident):
            site_id = _coerce_optional_int(site.get("id"))
            if site_id is not None:
                return site_id
    raise HTTPException(status_code=404, detail="Site not found")


async def _resolve_model_id(client: DUTAPIClient, identifier: str, *, site_hint: str | None = None) -> int:
    ident = str(identifier).strip()
    if ident.isdigit():
        return int(ident)

    try:
        if site_hint:
            site_id = await _resolve_site_id(client, site_hint)
            sites = [{"id": site_id, "name": site_hint}]
        else:
            sites = await _get_cached_sites(client)
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.error("Error fetching sites while resolving model '%s': %s", identifier, exc)
        raise HTTPException(status_code=503, detail="DUT service unavailable while retrieving model catalogue") from exc
    except Exception as exc:
        logger.error("Unexpected error fetching sites while resolving model '%s': %s", identifier, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve sites from external API") from exc

    for site in sites:
        site_id = _coerce_optional_int(site.get("id"))
        if site_id is None:
            continue
        try:
            models = await _get_cached_models(client, site_id)
        except Exception as exc:
            logger.warning("Error fetching models for site %s while resolving model '%s': %s", site_id, identifier, exc)
            continue
        for model in models:
            if _normalize_str(model.get("name")) == _normalize_str(ident):
                model_id = _coerce_optional_int(model.get("id"))
                if model_id is not None:
                    return model_id
    raise HTTPException(status_code=404, detail="Model not found")


async def _resolve_station_id(
    client: DUTAPIClient,
    identifier: str,
    *,
    model_hint: str | None = None,
    site_hint: str | None = None,
) -> int:
    ident = str(identifier).strip()
    if ident.isdigit():
        return int(ident)

    try:
        if model_hint:
            model_id = await _resolve_model_id(client, model_hint, site_hint=site_hint)
            target_models = [{"id": model_id}]
            sites = [{"id": None}]
        elif site_hint:
            site_id = await _resolve_site_id(client, site_hint)
            sites = [{"id": site_id, "name": site_hint}]
            target_models = None
        else:
            sites = await _get_cached_sites(client)
            target_models = None
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.error("Error fetching sites while resolving station '%s': %s", identifier, exc)
        raise HTTPException(status_code=503, detail="DUT service unavailable while retrieving station catalogue") from exc
    except Exception as exc:
        logger.error("Unexpected error fetching sites while resolving station '%s': %s", identifier, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve stations from external API") from exc

    for site in sites:
        site_id = _coerce_optional_int(site.get("id")) if site.get("id") is not None else None
        if site_id is None and target_models is None:
            continue
        try:
            models = target_models or await _get_cached_models(client, site_id)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning(
                "Error fetching models for site %s while resolving station '%s': %s",
                site_id,
                identifier,
                exc,
            )
            continue
        for model in models:
            model_id = _coerce_optional_int(model.get("id"))
            if model_id is None:
                continue
            try:
                stations = await _get_cached_stations(client, model_id)
            except Exception as exc:
                logger.warning(
                    "Error fetching stations for model %s while resolving station '%s': %s",
                    model_id,
                    identifier,
                    exc,
                )
                continue
            for station in stations:
                if _normalize_str(station.get("name")) == _normalize_str(ident):
                    station_id = _coerce_optional_int(station.get("id"))
                    if station_id is not None:
                        return station_id
    raise HTTPException(status_code=404, detail="Station not found")


async def _resolve_station_and_dut_ids(
    client: DUTAPIClient,
    station_identifier: str,
    dut_identifier: str,
    *,
    site_hint: str | None = None,
    model_hint: str | None = None,
) -> tuple[int, int]:
    station_id_numeric = int(station_identifier) if station_identifier.isdigit() else None
    original_dut = dut_identifier
    dut_id_numeric = int(dut_identifier) if dut_identifier.isdigit() else None

    if station_id_numeric is not None and dut_id_numeric is not None:
        return station_id_numeric, dut_id_numeric

    try:
        _, _, record_data = await _fetch_dut_records(
            client,
            original_dut,
            site_hint=site_hint,
            model_hint=model_hint,
        )
    except HTTPException as exc:
        raise HTTPException(status_code=404, detail="No DUT records found for the supplied identifier") from exc

    # Try exact match first (ISN matches the station)
    match = _match_station_entry(record_data, station_identifier, dut_identifier)
    if match:
        return match.station_id, match.station_dut_id

    # Fall back to loose match (find any DUT that tested on that station)
    match = _match_station_entry_loose(record_data, station_identifier)
    if match:
        return match.station_id, match.station_dut_id

    raise HTTPException(status_code=404, detail="Unable to resolve station/dut identifiers for the provided values")


def _parse_test_date(value: datetime | str | None) -> datetime:
    if not value:
        return datetime.min
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value
    try:
        value_norm = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(value_norm)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed
    except Exception:
        return datetime.min


def _matches_dut_isn(dut_isn_norm: str, station: dict, item: dict) -> bool:
    if not dut_isn_norm:
        return False

    candidate_values: list[Any] = []
    item_fields = [
        "dut_id__isn",
        "dut_id__ssid",
        "dut_id__ssn",
        "dut_id__mac",
        "dut_id__serial",
        "dut_id__identifier",
        "dut_id__isn_ssn",
        "dut_isn",
        "dut_id",
    ]
    station_fields = [
        "dut_isn",
        "dut_id__isn",
        "dut_id__ssid",
        "dut_id__ssn",
        "dut_id__mac",
        "dut_id",
    ]

    candidate_values.extend(item.get(field) for field in item_fields if item.get(field) is not None)
    candidate_values.extend(station.get(field) for field in station_fields if station.get(field) is not None)

    for value in candidate_values:
        normalized = _normalize_str(value)
        if normalized and normalized == dut_isn_norm:
            return True
    return False


@router.post(
    "/history/device-results",
    tags=["DUT_Management"],
    summary="Retrieve detailed test results for devices",
    response_model=list[TestResultRecordSchema],
    responses={
        200: {"description": "Device test results returned from the upstream DUT API."},
        400: {"description": "Invalid parameters or validation error."},
    },
)
async def get_test_results_by_device(
    params: DeviceResultQueryParams = Depends(_parse_device_results_form),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Proxy the upstream API that returns detailed device test results based on time windows.

    Accepts individual form fields for one or more device identifiers.
    """
    try:
        start_dt = _parse_input_datetime(params.start_time)
        end_dt = _parse_input_datetime(params.end_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {exc}") from exc

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")

    site_hint = _normalize_optional_string(params.site_id)
    model_hint = _normalize_optional_string(params.model_id)
    station_hint = _normalize_optional_string(params.station_identifier)

    aggregated_results: list[dict[str, Any]] = []
    resolved_cache: dict[str, str] = {}

    for current_device in params.device_ids:
        try:
            query = TestResultQuerySchema(
                site_id=params.site_id,
                model_id=params.model_id,
                device_id=current_device,
                start_time=start_dt,
                end_time=end_dt,
                test_result=params.test_result,
                model=params.model,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid query parameters: {exc}") from exc

        lookup_key = _normalize_str(current_device)
        resolved_device_id = resolved_cache.get(lookup_key)
        if resolved_device_id is None:
            resolved_device_id = await _resolve_device_identifier(
                client,
                query.device_id,
                station_hint=station_hint,
                model_hint=model_hint,
                site_hint=site_hint,
            )
            resolved_cache[lookup_key] = resolved_device_id

        normalized_model = _normalize_optional_string(query.model)
        payload = {
            "device_id": resolved_device_id,
            "start_time": _format_external_timestamp(query.start_time),
            "end_time": _format_external_timestamp(query.end_time),
            "test_result": query.test_result.upper(),
            "model": normalized_model,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        try:
            results = await client.get_test_results_by_device(payload)
        except Exception as exc:
            logger.error("Error fetching device test results for payload %s: %s", payload, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not isinstance(results, list):
            raise HTTPException(status_code=502, detail="Unexpected response format from external API")
        aggregated_results.extend(results)

    return aggregated_results


@router.post(
    "/history/model-summary",
    tags=["DUT_Management"],
    summary="Retrieve aggregate model summary for all stations",
    response_model=ModelSummarySchema,
    responses={
        200: {"description": "Aggregate model summary as reported by the upstream API."},
        400: {"description": "Invalid parameters or schema validation error."},
    },
)
async def get_model_summary(
    site_id: Annotated[
        str,
        Form(description="Site identifier or name used to narrow model lookup.", min_length=1),
    ],
    model_id: Annotated[str, Form(description="Model ID to query for summary statistics (e.g., 44 or HH5K)")],
    start_time: Annotated[str, Form(description="Start timestamp in ISO format (e.g., 2023-01-01T00:00:00Z or 2023-01-01T00:00:00)")],
    end_time: Annotated[str, Form(description="End timestamp in ISO format (e.g., 2023-01-02T00:00:00Z or 2023-01-02T00:00:00)")],
    model: Annotated[
        str,
        Form(description="Optional model name/identifier (e.g., null)"),
    ] = "",
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Proxy the upstream model summary aggregation endpoint.

    Accepts individual form fields for model summary query.
    """
    site_id = _require_non_empty(site_id, "site_id")
    model_id = _require_non_empty(model_id, "model_id")
    start_time = _require_non_empty(start_time, "start_time")
    end_time = _require_non_empty(end_time, "end_time")

    # Parse datetime strings
    try:
        start_dt = _parse_input_datetime(start_time)
        end_dt = _parse_input_datetime(end_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {exc}") from exc

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")

    # Build the query object
    normalized_model = _normalize_optional_string(model)
    site_hint = _normalize_optional_string(site_id)
    raw_model_id = model_id.strip()
    try:
        resolved_model_id = int(raw_model_id)
    except ValueError:
        resolved_model_id = await _resolve_model_id(client, raw_model_id, site_hint=site_hint)

    try:
        summary_query = ModelSummaryRequestSchema(
            model_id=resolved_model_id,
            start_time=start_dt,
            end_time=end_dt,
            model=normalized_model,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid query parameters: {exc}") from exc

    payload_model = _normalize_optional_string(summary_query.model)
    payload = {
        "model_id": summary_query.model_id,
        "start_time": _format_external_timestamp(summary_query.start_time),
        "end_time": _format_external_timestamp(summary_query.end_time),
        "model": payload_model,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    try:
        summary = await client.get_model_summary(payload)
    except Exception as exc:
        logger.error("Error fetching model summary for payload %s: %s", payload, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not isinstance(summary, dict):
        raise HTTPException(status_code=502, detail="Unexpected response format from external API")
    try:
        return ModelSummarySchema.model_validate(summary)
    except ValidationError as exc:
        logger.error("Model summary validation failed: %s", exc)
        raise HTTPException(status_code=502, detail="Unexpected response format from external API") from exc


@router.get(
    "/history/latest-tests",
    tags=["DUT_Management"],
    summary="Retrieve latest test data per station for a DUT",
    response_model=LatestTestsResponseSchema,
    responses={
        200: {"description": "Latest test entry per station that has tested the specified DUT."},
        404: {"description": "No test data found matching the provided criteria."},
    },
)
@cache(expire=60)
async def get_dut_latest_tests(
    dut_isn: str = Query(
        ...,
        description="DUT ISN identifier (string used with the external DUT API) (e.g., 10235789 or 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the DUT history (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the DUT history (e.g., 44 or HH5K).",
    ),
    stations: list[str] | None = Query(default=None, description="Optional list of station identifiers (IDs or names) to filter results. (e.g., 144 or Wireless_Test_6G)"),  # noqa: B008
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    if not record_data:
        raise HTTPException(status_code=404, detail="No test records found for the provided DUT")

    station_filters = _prepare_station_filters(stations)
    dut_isn_norm = _normalize_str(dut_isn)
    latest_records: list[StationLatestRecordSchema] = []
    station_record_cache: dict[tuple[int, int], StationRecordResponseSchema] = {}

    for station in record_data:
        station_id = station.get("id")
        station_name = station.get("name") or "Unknown Station"

        if station_filters and not _station_matches_filters(station_filters, station_id, station_name):
            continue

        data_entries = station.get("data") or []
        matching_entries = [item for item in data_entries if _matches_dut_isn(dut_isn_norm, station, item)]

        candidate_entries = matching_entries or data_entries

        if not candidate_entries:
            continue

        latest_entry = None
        latest_date = datetime.min
        for entry in candidate_entries:
            parsed_date = _parse_test_date(entry.get("test_date"))
            if parsed_date is None:
                continue
            if parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = entry

        if latest_entry is None:
            continue

        resolved_station_id = station_id
        candidate_dut_id = latest_entry.get("dut_id") or station.get("dut_id")

        if resolved_station_id is None or candidate_dut_id is None:
            station_identifier = station_name if resolved_station_id is None else str(resolved_station_id)
            dut_identifier = str(candidate_dut_id) if candidate_dut_id is not None else latest_entry.get("dut_id__isn") or station.get("dut_isn") or dut_isn
            try:
                resolved_station_id, candidate_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    str(station_identifier),
                    str(dut_identifier),
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
            except HTTPException:
                continue

        cache_key = (resolved_station_id, candidate_dut_id)
        station_payload = station_record_cache.get(cache_key)
        if station_payload is None:
            try:
                raw_payload = await client.get_station_records(resolved_station_id, candidate_dut_id)
                station_payload = StationRecordResponseSchema.model_validate(raw_payload)
                station_record_cache[cache_key] = station_payload
            except Exception as exc:
                logger.warning(
                    "Unable to retrieve station records for station %s (%s) and dut %s: %s",
                    station_name,
                    resolved_station_id,
                    candidate_dut_id,
                    exc,
                )
                station_payload = StationRecordResponseSchema(record=[], data=[])
        run_index = _find_run_index(station_payload, latest_entry)
        trimmed_data = _compress_station_data(station_payload, run_index)

        latest_records.append(
            StationLatestRecordSchema(
                station_id=resolved_station_id,
                station_name=station_name,
                test_date=latest_entry.get("test_date"),
                device_id=latest_entry.get("device_id"),
                device=latest_entry.get("device_id__name") or latest_entry.get("device"),
                test_duration=latest_entry.get("test_duration"),
                dut_id=candidate_dut_id,
                isn=latest_entry.get("dut_id__isn") or station.get("dut_isn") or dut_isn,
                status=latest_entry.get("test_result"),
                order=station.get("order"),
                error_item=latest_entry.get("error_item"),
                data=trimmed_data,
            )
        )

    if not latest_records:
        raise HTTPException(status_code=404, detail="No test records found for the provided filters")

    def _latest_record_sort_key(entry: StationLatestRecordSchema):
        order_flag, order_value, _ = _order_sort_components(entry, name_getter=lambda value: value.station_name)
        parsed_date = _parse_test_date(entry.test_date)
        if not parsed_date or parsed_date == datetime.min:
            timestamp_key = float("inf")
        else:
            timestamp_key = -parsed_date.timestamp()
        fallback_name = (entry.station_name or "").lower()
        return (order_flag, order_value, timestamp_key, fallback_name)

    latest_records.sort(key=_latest_record_sort_key)
    return LatestTestsResponseSchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        station_count=len(latest_records),
        record_data=latest_records,
    )


@router.get(
    "/history/identifiers",
    tags=["DUT_Management"],
    summary="List DUT identifiers linked to an ISN",
    response_model=DUTIdentifierListSchema,
)
@cache(expire=60)
async def get_dut_identifiers(
    dut_isn: str = Query(..., description="DUT ID or ISN/SSN/MAC identifier. (e.g., 260884980003907)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the DUT history (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the DUT history (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    identifiers: dict[tuple[int | None, str | None], DUTIdentifierSchema] = {}

    for station in record_data:
        station_id = station.get("id")
        station_name = station.get("name")
        station_isn = station.get("dut_isn") or dut_isn
        station_dut_id = station.get("dut_id")

        if station_dut_id is not None:
            key = (station_dut_id, station_name)
            identifiers.setdefault(
                key,
                DUTIdentifierSchema(
                    dut_id=station_dut_id,
                    station_id=station_id,
                    station_name=station_name,
                    dut_isn=station_isn,
                ),
            )

        for item in station.get("data") or []:
            key = (item.get("dut_id"), station_name)
            item_isn = item.get("dut_id__isn") or station_isn
            if key not in identifiers:
                identifiers[key] = DUTIdentifierSchema(
                    dut_id=item.get("dut_id"),
                    station_id=station_id,
                    station_name=station_name,
                    dut_isn=item_isn,
                )
            elif not identifiers[key].dut_isn:
                identifiers[key].dut_isn = item_isn

    return DUTIdentifierListSchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        identifiers=list(identifiers.values()),
    )


@router.get(
    "/history/progression",
    tags=["DUT_Management"],
    summary="Show station progression for a DUT",
    response_model=DUTProgressSchema,
)
@cache(expire=60)
async def get_dut_progression(
    dut_isn: str = Query(..., description="DUT ID or ISN/SSN/MAC identifier. (e.g., 260884980003907)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the DUT history (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the DUT history (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    if not record_data:
        raise HTTPException(status_code=404, detail="No test records found for the provided DUT")

    stations = []
    for station in record_data:
        data_entries = station.get("data") or []
        stations.append(
            StationProgressSchema(
                station_id=station.get("id"),
                station_name=station.get("name") or "Unknown Station",
                status=station.get("status"),
                tested=bool(data_entries),
                test_runs=len(data_entries),
            )
        )

    return DUTProgressSchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        stations=stations,
    )


@router.get(
    "/history/results",
    tags=["DUT_Management"],
    summary="Summarise station results for a DUT",
    response_model=DUTRunSummarySchema,
)
@cache(expire=60)
async def get_dut_results(
    dut_isn: str = Query(..., description="DUT ID or ISN/SSN/MAC identifier. (e.g., 260884980003907)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the DUT history (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the DUT history (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    if not record_data:
        raise HTTPException(status_code=404, detail="No test records found for the provided DUT")

    station_summaries: list[StationRunSummarySchema] = []

    for station in record_data:
        data_entries = station.get("data") or []
        pass_runs = sum(1 for item in data_entries if _status_from_result(item.get("test_result")) == "PASS")
        fail_runs = sum(1 for item in data_entries if _status_from_result(item.get("test_result")) == "FAIL")

        results = [
            DeviceTestResultSchema(
                test_date=item.get("test_date"),
                test_result=item.get("test_result"),
                status=_status_from_result(item.get("test_result")),
                error_item=item.get("error_item"),
                test_duration=item.get("test_duration"),
            )
            for item in data_entries
        ]

        station_summaries.append(
            StationRunSummarySchema(
                station_id=station.get("id"),
                station_name=station.get("name") or "Unknown Station",
                test_runs=len(data_entries),
                pass_runs=pass_runs,
                fail_runs=fail_runs,
                results=results,
            )
        )

    return DUTRunSummarySchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        stations=station_summaries,
    )


@router.get(
    "/history/isns",
    tags=["DUT_Management"],
    summary="List DUT ISN variants",
    response_model=DUTISNVariantListSchema,
)
@cache(expire=60)
async def get_dut_isn_variants(
    dut_isn: str = Query(..., description="DUT ID or ISN/SSN/MAC identifier. (e.g., 260884980003907)"),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the DUT history (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the DUT history (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )
    seen: set[str] = set()
    ordered_variants: list[str] = []

    def _append_variant(value: str | None) -> None:
        if not value:
            return
        variant = str(value)
        if variant in seen:
            return
        seen.add(variant)
        ordered_variants.append(variant)

    _append_variant(str(dut_isn))

    for station in record_data:
        _append_variant(station.get("dut_isn"))
        for item in station.get("data") or []:
            _append_variant(item.get("dut_id__isn"))

    return DUTISNVariantListSchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        isns=ordered_variants,
    )


@router.get(
    "/summary",
    tags=["DUT_Management"],
    summary="Summarise DUT history by ISN",
    response_model=DUTTestSummarySchema,
    responses={
        200: {"description": "Aggregated view of DUT test history across stations and devices."},
        404: {"description": "No test records found for the provided DUT identifier."},
    },
)
@cache(expire=60)
async def get_dut_summary(
    dut_isn: str = Query(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier. (e.g., 260884980003907)",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to filter the DUT history (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to filter the DUT history (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Aggregate DUT information including stations, devices, and result counts.
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")
    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )

    if not record_data:
        raise HTTPException(status_code=404, detail="No test records found for the provided DUT")

    station_map: dict[object, dict] = {}

    for station in record_data:
        station_id = station.get("id")
        station_name = station.get("name") or "Unknown Station"
        key = station_id or station_name

        entry = station_map.setdefault(
            key,
            {
                "station_id": station_id,
                "station_name": station_name,
                "svn_name": station.get("svn_name"),
                "svn_url": station.get("svn_url"),
                "dut_id": station.get("dut_id"),
                "dut_isn": station.get("dut_isn") or dut_isn,
                "test_runs": 0,
                "devices": defaultdict(
                    lambda: {
                        "device_id": None,
                        "device_name": None,
                        "total_runs": 0,
                        "pass_runs": 0,
                        "fail_runs": 0,
                        "results": [],
                    }
                ),
            },
        )

        if station.get("dut_id"):
            entry["dut_id"] = station["dut_id"]
        if station.get("dut_isn"):
            entry["dut_isn"] = station["dut_isn"]

        data_entries = station.get("data") or []
        entry["test_runs"] += len(data_entries)

        for item in data_entries:
            device_id = item.get("device_id")
            device_name = item.get("device_id__name") or item.get("device")
            device_key = (device_id, device_name)

            device_summary = entry["devices"][device_key]
            device_summary["device_id"] = device_id
            device_summary["device_name"] = device_name
            device_summary["total_runs"] += 1

            status = _status_from_result(item.get("test_result"))
            if status == "PASS":
                device_summary["pass_runs"] += 1
            elif status == "FAIL":
                device_summary["fail_runs"] += 1

            device_summary["results"].append(
                {
                    "test_date": item.get("test_date"),
                    "test_result": item.get("test_result"),
                    "status": status,
                    "error_item": item.get("error_item"),
                    "test_duration": item.get("test_duration"),
                }
            )

    stations: list[StationTestSummarySchema] = []
    for station_data in station_map.values():
        devices = [
            DeviceSummarySchema(
                device_id=dev["device_id"],
                device_name=dev["device_name"],
                total_runs=dev["total_runs"],
                pass_runs=dev["pass_runs"],
                fail_runs=dev["fail_runs"],
                results=[DeviceTestResultSchema(**res) for res in dev["results"]],
            )
            for dev in station_data["devices"].values()
        ]
        stations.append(
            StationTestSummarySchema(
                station_id=station_data["station_id"],
                station_name=station_data["station_name"],
                svn_name=station_data["svn_name"],
                svn_url=station_data["svn_url"],
                dut_id=station_data["dut_id"],
                dut_isn=station_data["dut_isn"],
                test_runs=station_data["test_runs"],
                devices=devices,
            )
        )

    summary = DUTTestSummarySchema(
        dut_isn=dut_isn,
        site_name=site_name,
        model_name=model_name,
        station_count=len(stations),
        stations=stations,
    )
    return summary


# ==================== PA Test Items Adjusted Power Calculation ====================


def _extract_pa_test_items_from_measurements(measurements: list[MeasurementRow]) -> tuple[list[str], dict[str, str]]:
    """Extract PA SROM test items from measurement rows.

    Args:
        measurements: List of measurement rows from station records

    Returns:
        Tuple of (test_items_list, base_name_to_original_mapping)
        - test_items_list: List of PA SROM test item names (OLD and NEW)
        - base_name_to_original_mapping: Maps base names to original test item names
          (used to create virtual adjusted power test item names)

    Examples:
        Input measurements with:
        - WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80
        - WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80
        - WiFi_TX1_POW_2412_11B_CCK11_B20

        Output:
        - test_items: ['WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80', 'WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80']
        - mapping: {'WiFi_PA1_5985_11AX_MCS9_B80': 'WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80'}
    """
    pa_test_items: list[str] = []
    base_name_mapping: dict[str, str] = {}

    for row in measurements:
        if _is_pa_srom_test_item(row.name):
            pa_test_items.append(row.name)
            # Map base name to original for creating virtual test item names
            base_name = _extract_pa_base_name(row.name)
            if base_name and base_name not in base_name_mapping:
                base_name_mapping[base_name] = row.name

    return pa_test_items, base_name_mapping


async def _fetch_and_calculate_pa_adjusted_power(
    client: DUTAPIClient,
    station_id: int,
    pa_test_items: list[str],
    start_time: datetime,
    end_time: datetime,
    model_hint: str | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch PA trend data and calculate adjusted power values.

    Args:
        client: DUT API client
        station_id: Station ID
        pa_test_items: List of PA SROM test item names
        start_time: Start of evaluation window
        end_time: End of evaluation window
        model_hint: Optional model identifier for the query

    Returns:
        Dictionary mapping base test item names to dictionaries containing MID and MEAN adjusted power values
        Example: {
            'WiFi_PA1_5985_11AX_MCS9_B80': {'mid': 0.37, 'mean': 0.35},
            'WiFi_PA2_6275_11AC_VHT40_MCS9': {'mid': 0.42, 'mean': 0.40}
        }

    Raises:
        HTTPException: If PA trend API call fails or date range exceeds 7 days
    """
    if not pa_test_items:
        return {}

    # Validate date range (max 7 days as per external API constraint)
    time_diff = end_time - start_time
    if time_diff > timedelta(days=7):
        logger.warning("PA adjusted power date range exceeds 7 days, skipping PA calculation")
        return {}

    # Prepare payload for external DUT API with Z-format timestamps
    payload = {
        "start_time": start_time.isoformat().replace("+00:00", "Z"),
        "end_time": end_time.isoformat().replace("+00:00", "Z"),
        "station_id": station_id,
        "test_items": pa_test_items,
        "model": model_hint or "",
    }

    # Fetch trend data from external API
    try:
        trend_data = await client.get_pa_test_items_trend(payload)
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Failed to fetch PA trend data for station %s: status=%s body=%s",
            station_id,
            exc.response.status_code,
            exc.response.text,
        )
        return {}
    except Exception as exc:
        logger.warning("Unexpected error fetching PA trend data for station %s: %s", station_id, exc)
        return {}

    if not trend_data:
        logger.debug("No PA trend data returned for station %s", station_id)
        return {}

    # Group items by base name and calculate adjusted power
    paired_items: dict[str, dict[str, Any]] = {}
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

    # Calculate adjusted power for each complete pair
    adjusted_power_values: dict[str, dict[str, float]] = {}
    for base_name, values in paired_items.items():
        # Calculate using the existing helper function
        adjusted_power_result = _calculate_adjusted_power(
            old_mid=values["old_mid"],
            old_mean=values["old_mean"],
            new_mid=values["new_mid"],
            new_mean=values["new_mean"],
        )

        # Store both MID and MEAN values if available
        result_dict: dict[str, float] = {}
        if adjusted_power_result.get("adjusted_mid") is not None:
            result_dict["mid"] = adjusted_power_result["adjusted_mid"]
            logger.debug("Calculated PA adjusted power (MID) for %s: %.2f", base_name, result_dict["mid"])

        if adjusted_power_result.get("adjusted_mean") is not None:
            result_dict["mean"] = adjusted_power_result["adjusted_mean"]
            logger.debug("Calculated PA adjusted power (MEAN) for %s: %.2f", base_name, result_dict["mean"])

        if result_dict:
            adjusted_power_values[base_name] = result_dict

    return adjusted_power_values


def _extract_pa_base_name(test_item_name: str) -> str | None:
    """Extract base test item name by removing SROM_OLD or SROM_NEW suffix.

    Examples:
        WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80 -> WiFi_PA1_5985_11AX_MCS9_B80
        WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80 -> WiFi_PA1_5985_11AX_MCS9_B80
        WiFi_PA2_SROM_OLD_6275_11AC_VHT40_MCS9 -> WiFi_PA2_6275_11AC_VHT40_MCS9
    """
    if not test_item_name:
        return None

    # Pattern: PA{1-4}_SROM_OLD or PA{1-4}_SROM_NEW
    old_pattern = re.compile(r"(PA[1-4])_SROM_OLD_(.+)", re.IGNORECASE)
    new_pattern = re.compile(r"(PA[1-4])_SROM_NEW_(.+)", re.IGNORECASE)

    old_match = old_pattern.search(test_item_name)
    if old_match:
        # Reconstruct: prefix_PA{n}_suffix
        prefix = test_item_name[: old_match.start(1)]
        pa_num = old_match.group(1)
        suffix = old_match.group(2)
        return f"{prefix}{pa_num}_{suffix}"

    new_match = new_pattern.search(test_item_name)
    if new_match:
        prefix = test_item_name[: new_match.start(1)]
        pa_num = new_match.group(1)
        suffix = new_match.group(2)
        return f"{prefix}{pa_num}_{suffix}"

    return None


def _calculate_adjusted_power(old_mid: float | None, old_mean: float | None, new_mid: float | None, new_mean: float | None) -> dict[str, float | None]:
    """Calculate adjusted power values using formula: (NEW - OLD) / 256.

    Returns:
        {
            "adjusted_mid": float | None,
            "adjusted_mean": float | None,
            "raw_mid_difference": float | None,
            "raw_mean_difference": float | None
        }
    """
    result: dict[str, float | None] = {"adjusted_mid": None, "adjusted_mean": None, "raw_mid_difference": None, "raw_mean_difference": None}

    if old_mid is not None and new_mid is not None:
        raw_mid_diff = new_mid - old_mid
        result["raw_mid_difference"] = raw_mid_diff
        result["adjusted_mid"] = round(raw_mid_diff / 256, 2)

    if old_mean is not None and new_mean is not None:
        raw_mean_diff = new_mean - old_mean
        result["raw_mean_difference"] = raw_mean_diff
        result["adjusted_mean"] = round(raw_mean_diff / 256, 2)

    return result


@router.post(
    "/pa/trend",
    tags=["DUT_Management"],
    summary="Calculate adjusted power from PA SROM OLD/NEW test items",
    response_model=AdjustedPowerResponseSchema,
    responses={200: {"description": "Adjusted power calculations for PA test items."}, 400: {"description": "Invalid request or date range exceeds 7 days."}, 404: {"description": "No trend data available."}},
)
async def calculate_pa_adjusted_power(request: AdjustedPowerRequestSchema, client: DUTAPIClient = dut_client_dependency):
    """
    Calculate adjusted power values from PA test items trend data.

    This endpoint:
    1. Fetches PA test items trend data (mean/mid values) from external DUT API
    2. Pairs SROM_OLD and SROM_NEW test items by their base names
    3. Calculates adjusted power using formula: (NEW - OLD) / 256
    4. Rounds results to 2 decimal places

    **Formula:**
    - Adjusted Mid Power = (NEW_mid - OLD_mid) / 256
    - Adjusted Mean Power = (NEW_mean - OLD_mean) / 256

    **Example:**
    ```
    Input test items:
    - WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80: {mid: 11219.0, mean: 11227}
    - WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80: {mid: 11313.0, mean: 11308}

    Output for WiFi_PA1_5985_11AX_MCS9_B80:
    - adjusted_mid: 0.37 = (11313.0 - 11219.0) / 256
    - adjusted_mean: 0.32 = (11308 - 11227) / 256
    ```

    **Constraints:**
    - Maximum date range: 7 days
    - Test items must include both OLD and NEW variants
    """
    # Validate date range (max 7 days as per external API constraint)
    time_diff = request.end_time - request.start_time
    if time_diff > timedelta(days=7):
        raise HTTPException(status_code=400, detail="Date range exceeds maximum allowed 7 days")

    if time_diff.total_seconds() < 0:
        raise HTTPException(status_code=400, detail="end_time must be after start_time")

    # Prepare payload for external DUT API with Z-format timestamps
    payload = {
        "start_time": request.start_time.isoformat().replace("+00:00", "Z"),
        "end_time": request.end_time.isoformat().replace("+00:00", "Z"),
        "station_id": request.station_id,
        "test_items": request.test_items,
        "model": request.model or "",
    }

    # Fetch trend data from external API
    try:
        trend_data = await client.get_pa_test_items_trend(payload)
    except httpx.HTTPStatusError as exc:
        logger.error("Failed to fetch PA trend data: status=%s body=%s", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=exc.response.status_code, detail=f"External API error: {exc.response.text}") from exc
    except Exception as exc:
        logger.error("Unexpected error fetching PA trend data: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch trend data: {exc!s}") from exc

    if not trend_data:
        raise HTTPException(status_code=404, detail="No trend data returned from external API")

    # Group items by base name
    paired_items: dict[str, dict[str, Any]] = {}
    unpaired_items: list[str] = []

    for item_name, item_data in trend_data.items():
        base_name = _extract_pa_base_name(item_name)
        if not base_name:
            logger.warning("Could not extract base name from test item: %s", item_name)
            unpaired_items.append(item_name)
            continue

        if base_name not in paired_items:
            paired_items[base_name] = {"old_item_name": None, "new_item_name": None, "old_values": None, "new_values": None}

        if "_SROM_OLD_" in item_name.upper():
            paired_items[base_name]["old_item_name"] = item_name
            paired_items[base_name]["old_values"] = item_data
        elif "_SROM_NEW_" in item_name.upper():
            paired_items[base_name]["new_item_name"] = item_name
            paired_items[base_name]["new_values"] = item_data

    # Calculate adjusted power for each pair and build new response structure
    result_data: list[dict[str, Any]] = []

    for base_name, pair_data in paired_items.items():
        old_values = pair_data.get("old_values") or {}
        new_values = pair_data.get("new_values") or {}
        old_item_name = pair_data.get("old_item_name")
        new_item_name = pair_data.get("new_item_name")

        # Check if we have both OLD and NEW
        if not old_item_name or not new_item_name:
            error_msg = f"Missing pair: OLD={old_item_name}, NEW={new_item_name}"
            logger.warning("Incomplete pair for %s: %s", base_name, error_msg)

            if old_item_name:
                unpaired_items.append(old_item_name)
            if new_item_name:
                unpaired_items.append(new_item_name)
            continue

        # Calculate adjusted power
        adjusted_power_calc = _calculate_adjusted_power(
            old_mid=old_values.get("mid"),
            old_mean=old_values.get("mean"),
            new_mid=new_values.get("mid"),
            new_mean=new_values.get("mean"),
        )

        # Build adjusted_power_trend with proper naming
        adjusted_power_trend = {}
        if adjusted_power_calc.get("adjusted_mid") is not None:
            mid_name = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POW_MID_", base_name, count=1, flags=re.IGNORECASE)
            adjusted_power_trend[mid_name] = adjusted_power_calc["adjusted_mid"]

        if adjusted_power_calc.get("adjusted_mean") is not None:
            mean_name = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POW_MEAN_", base_name, count=1, flags=re.IGNORECASE)
            adjusted_power_trend[mean_name] = adjusted_power_calc["adjusted_mean"]

        # Build response item with OLD/NEW SROM values as top-level keys
        data_item = {
            old_item_name: old_values,
            new_item_name: new_values,
            "adjusted_power_trend": adjusted_power_trend,
            "error": None,
        }

        result_data.append(data_item)

    logger.info("PA adjusted power calculation completed: %d paired items, %d unpaired items", len(result_data), len(unpaired_items))

    return {
        "station_id": request.station_id,
        "start_time": request.start_time,
        "end_time": request.end_time,
        "data": result_data,
        "unpaired_items": list(set(unpaired_items)),
    }


@router.get(
    "/pa/trend/auto",
    tags=["DUT_Management"],
    summary="[nonvalue-trend] Get PA SROM trend values (mid/mean) for DUT stations",
    response_model=list[PATrendStationDataSchema],
    responses={
        200: {"description": "PA SROM trend values (mid/mean) for the DUT stations."},
        400: {"description": "Invalid parameters or time window exceeds 7 days."},
        404: {"description": "No DUT records found or no PA test items available."},
    },
)
@cache(expire=60)
async def get_pa_trend_auto(
    dut_isn: list[str] = Query(..., description="DUT ISN identifier(s) (e.g., 261534750003154 or DM2527470036123). Can provide multiple."),
    station_id: list[str] = Query(..., description="Station identifier(s) (numeric ID or name) (e.g., 144 or Wireless_Test_6G). Can provide multiple."),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the lookup (e.g., 44 or HH5K).",
    ),
    start_time: datetime | None = Query(
        None,
        description="Optional start of time window for PA trend calculation (ISO format, e.g., 2023-01-01T00:00:00Z). If not provided, defaults to 24 hours before DUT ISN test_date.",
    ),
    end_time: datetime | None = Query(
        None,
        description="Optional end of time window for PA trend calculation (ISO format, e.g., 2023-01-01T23:59:59Z). If not provided, defaults to DUT ISN test_date.",
    ),
    srom_filter: Literal["all", "old", "new"] = Query(
        default="all",
        description="Filter PA SROM items: 'all' (both OLD and NEW), 'old' (only SROM_OLD), 'new' (only SROM_NEW)",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Retrieves PA SROM trend values (mid/mean) for DUT(s) and station(s).

    This endpoint:
    1. Fetches DUT records for the specified ISN(s)
    2. Extracts PA SROM (OLD/NEW) test items from nonvalue records for each station
    3. Calls PA trend API to get mid/mean values for each test item
    4. Returns the raw mid/mean values for each station (no adjusted power calculations)

    Supports multiple station_id and dut_isn parameters (can use IDs or names).

    **Time Window:**
    - If start_time and end_time are not provided, automatically uses 24-hour window from test_date
    - Maximum allowed window: 7 days
    - If test_date is not available, uses current time - 24 hours
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    # Collect all station records
    station_records = []

    # Process each combination of dut_isn and station_id
    for dut in dut_isn:
        dut = dut.strip()

        for station in station_id:
            station = station.strip()

            try:
                logger.info(f"[PA Trend Auto] Processing dut={dut}, station={station}")

                # Fetch DUT records to get station metadata
                site_name, model_name, record_data = await _fetch_dut_records(
                    client,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                if not record_data:
                    logger.warning(f"[PA Trend Auto] No DUT records found for dut={dut}")
                    continue

                # Resolve IDs
                resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    station,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
                logger.info(f"[PA Trend Auto] Resolved: station_id={resolved_station_id}, dut_id={resolved_dut_id}")

                # Find the station in the record data to get the correct station name
                station_name_str = f"Station {resolved_station_id}"
                isn_val = None
                test_date_val = None
                device_val = None

                for entry in record_data:
                    if _coerce_optional_int(entry.get("id")) == resolved_station_id:
                        station_name_str = entry.get("name") or station_name_str
                        # Get latest data entry for metadata
                        data_entries = entry.get("data") or []
                        if data_entries:
                            # Find latest entry
                            latest_entry = None
                            latest_date = datetime.min
                            for data_entry in data_entries:
                                parsed_date = _parse_test_date(data_entry.get("test_date"))
                                if parsed_date and parsed_date >= latest_date:
                                    latest_date = parsed_date
                                    latest_entry = data_entry

                            if latest_entry:
                                test_date_val = latest_entry.get("test_date")
                                device_val = latest_entry.get("device")
                                isn_val = latest_entry.get("isn")
                        break

                logger.info(f"[PA Trend Auto] Station name: {station_name_str}, ISN: {isn_val}")

                # Fetch nonvalue data
                payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
                logger.info(f"[PA Trend Auto] Fetched nonvalue payload with {len(payload.get('data', []))} data rows")

                validated = StationRecordResponseSchema.model_validate(payload)

                # Extract PA SROM test items
                station_pa_test_items: list[str] = []

                for data_row in validated.data:
                    if not data_row or len(data_row) < 2:
                        continue

                    test_item_name = str(data_row[0]).strip()
                    if not _is_pa_srom_test_item(test_item_name, srom_filter):
                        continue

                    station_pa_test_items.append(test_item_name)

                if not station_pa_test_items:
                    logger.warning(f"[PA Trend Auto] No PA SROM test items found for station={station}, dut={dut}")
                    continue

                logger.info(f"[PA Trend Auto] Found {len(station_pa_test_items)} PA SROM test items")

                # Determine time window
                if start_time is None or end_time is None:
                    parsed_test_date = _parse_test_date(test_date_val)
                    if parsed_test_date:
                        if parsed_test_date.tzinfo is None:
                            window_end = parsed_test_date.replace(tzinfo=UTC)
                        else:
                            window_end = parsed_test_date.astimezone(UTC)
                    else:
                        window_end = datetime.now(UTC)

                    window_start = window_end - timedelta(hours=24)
                    actual_start = start_time or window_start
                    actual_end = end_time or window_end
                else:
                    if start_time.tzinfo is None:
                        actual_start = start_time.replace(tzinfo=UTC)
                    else:
                        actual_start = start_time
                    if end_time.tzinfo is None:
                        actual_end = end_time.replace(tzinfo=UTC)
                    else:
                        actual_end = end_time

                    time_diff = actual_end - actual_start
                    if time_diff > timedelta(days=7):
                        raise HTTPException(status_code=400, detail="Time window exceeds maximum allowed 7 days")
                    if time_diff.total_seconds() < 0:
                        raise HTTPException(status_code=400, detail="end_time must be after start_time")

                # Fetch PA trend data
                start_time_str = actual_start.isoformat().replace("+00:00", "Z")
                end_time_str = actual_end.isoformat().replace("+00:00", "Z")

                payload = {
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "station_id": resolved_station_id,
                    "test_items": station_pa_test_items,
                    "model": model_hint or "",
                }

                trend_data = await client.get_pa_test_items_trend(payload)

                if not trend_data:
                    logger.warning(f"[PA Trend Auto] No trend data returned for station={station}, dut={dut}")
                    continue

                # Convert trend data to list of items
                trend_items = []
                for item_name, item_data in trend_data.items():
                    trend_items.append(
                        PATrendStationItemSchema(
                            test_item_name=item_name,
                            mid=item_data.get("mid"),
                            mean=item_data.get("mean"),
                        )
                    )

                # Sort trend items by frequency, antenna, SROM type, and name
                trend_items.sort(key=lambda item: _get_pa_srom_sort_key(item.test_item_name))

                # Apply srom_filter to trend items
                if srom_filter != "all":
                    trend_items = [item for item in trend_items if _is_pa_srom_test_item(item.test_item_name, srom_filter)]

                if not trend_items:
                    logger.warning(f"[PA Trend Auto] No trend items after filtering for station={station}, dut={dut}")
                    continue

                logger.info(f"[PA Trend Auto] Created {len(trend_items)} trend items")

                # Create station record
                station_record = PATrendStationDataSchema(
                    station_id=resolved_station_id,
                    station_name=station_name_str,
                    test_date=test_date_val,
                    device=device_val,
                    isn=isn_val or dut,
                    trend_items=trend_items,
                )
                station_records.append(station_record)
                logger.info(f"[PA Trend Auto] Added station record")

            except Exception as exc:
                logger.error(f"[PA Trend Auto] Error processing station={station}, dut={dut}: {exc}", exc_info=True)
                continue

    if not station_records:
        logger.error(f"[PA Trend Auto] No station records found for dut_isn={dut_isn}, station_id={station_id}")
        raise HTTPException(status_code=404, detail="No PA SROM trend data found for any station/DUT combination")

    logger.info(f"[PA Trend Auto] Returning {len(station_records)} station records")
    return station_records


@router.get(
    "/pa/trend/decimal",
    tags=["DUT_Management"],
    summary="[nonvalue-trend] Get PA SROM adjusted trend values of decimal from trend data",
    response_model=list[PATrendStationDataSchema],
    responses={
        200: {"description": "PA SROM adjusted trend values for the DUT stations."},
        400: {"description": "Invalid parameters or time window exceeds 7 days."},
        404: {"description": "No DUT records found or no PA test items available."},
    },
)
@cache(expire=60)
async def get_pa_trend_dex(
    dut_isn: list[str] = Query(..., description="DUT ISN identifier(s) (e.g., 261534750003154 or DM2527470036123). Can provide multiple."),
    station_id: list[str] = Query(..., description="Station identifier(s) (numeric ID or name) (e.g., 144 or Wireless_Test_6G). Can provide multiple."),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the lookup (e.g., 44 or HH5K).",
    ),
    start_time: datetime | None = Query(
        None,
        description="Optional start of time window for PA trend calculation (ISO format, e.g., 2023-01-01T00:00:00Z). If not provided, defaults to 24 hours before DUT ISN test_date.",
    ),
    end_time: datetime | None = Query(
        None,
        description="Optional end of time window for PA trend calculation (ISO format, e.g., 2023-01-01T23:59:59Z). If not provided, defaults to DUT ISN test_date.",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get PA SROM adjusted trend values using the formula: (SROM_NEW - SROM_OLD)/256.

    This endpoint:
    1. Fetches DUT records for the specified ISN(s)
    2. Extracts PA SROM (OLD and NEW) test items from nonvalue records for each station
    3. Calls PA trend API to get mid/mean values for each test item
    4. Pairs SROM_NEW and SROM_OLD by frequency/protocol
    5. Calculates adjusted values: (SROM_NEW - SROM_OLD)/256 for both mid and mean
    6. Returns adjusted trend values with naming: WiFi_PA{X}_TREND_ADJUSTED_POW_{frequency}_{protocol}

    Supports multiple station_id and dut_isn parameters (can use IDs or names).

    **Time Window:**
    - If start_time and end_time are not provided, automatically uses 24-hour window from test_date
    - Maximum allowed window: 7 days
    - If test_date is not available, uses current time - 24 hours

    **Formula:**
    - Adjusted Mid = (SROM_NEW mid - SROM_OLD mid) / 256
    - Adjusted Mean = (SROM_NEW mean - SROM_OLD mean) / 256
    - Results are rounded to 2 decimal places
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    # Collect all station records
    station_records = []

    # Process each combination of dut_isn and station_id
    for dut in dut_isn:
        dut = dut.strip()

        for station in station_id:
            station = station.strip()

            try:
                logger.info(f"[PA Trend Dex] Processing dut={dut}, station={station}")

                # Fetch DUT records to get station metadata
                site_name, model_name, record_data = await _fetch_dut_records(
                    client,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                if not record_data:
                    logger.warning(f"[PA Trend Dex] No DUT records found for dut={dut}")
                    continue

                # Resolve IDs
                resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    station,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
                logger.info(f"[PA Trend Dex] Resolved: station_id={resolved_station_id}, dut_id={resolved_dut_id}")

                # Find the station in the record data
                station_name_str = f"Station {resolved_station_id}"
                isn_val = None
                test_date_val = None
                device_val = None

                for entry in record_data:
                    if _coerce_optional_int(entry.get("id")) == resolved_station_id:
                        station_name_str = entry.get("name") or station_name_str
                        data_entries = entry.get("data") or []
                        if data_entries:
                            latest_entry = None
                            latest_date = datetime.min
                            for data_entry in data_entries:
                                parsed_date = _parse_test_date(data_entry.get("test_date"))
                                if parsed_date and parsed_date >= latest_date:
                                    latest_date = parsed_date
                                    latest_entry = data_entry

                            if latest_entry:
                                test_date_val = latest_entry.get("test_date")
                                device_val = latest_entry.get("device")
                                isn_val = latest_entry.get("isn")
                        break

                logger.info(f"[PA Trend Dex] Station: {station_name_str}, ISN: {isn_val}")

                # Fetch nonvalue records
                payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
                logger.info(f"[PA Trend Dex] Fetched nonvalue payload with {len(payload.get('data', []))} data rows")

                validated = StationRecordResponseSchema.model_validate(payload)

                # Extract PA SROM test items (both OLD and NEW)
                pa_test_items = []
                for row in validated.data:
                    if not isinstance(row, list) or len(row) < 1:
                        continue
                    test_item_name = str(row[0]) if row else ""
                    if _is_pa_srom_test_item(test_item_name, "all"):
                        pa_test_items.append(test_item_name)

                if not pa_test_items:
                    logger.warning(f"[PA Trend Dex] No PA SROM items for station={station}, dut={dut}")
                    continue

                logger.info(f"[PA Trend Dex] Found {len(pa_test_items)} PA SROM test items")

                # Determine time window
                if start_time is None or end_time is None:
                    parsed_test_date = _parse_test_date(test_date_val)
                    if parsed_test_date:
                        if parsed_test_date.tzinfo is None:
                            window_end = parsed_test_date.replace(tzinfo=UTC)
                        else:
                            window_end = parsed_test_date.astimezone(UTC)
                    else:
                        window_end = datetime.now(UTC)

                    window_start = window_end - timedelta(hours=24)
                    actual_start = start_time or window_start
                    actual_end = end_time or window_end
                else:
                    actual_start = start_time.replace(tzinfo=UTC) if start_time.tzinfo is None else start_time
                    actual_end = end_time.replace(tzinfo=UTC) if end_time.tzinfo is None else end_time

                    time_diff = actual_end - actual_start
                    if time_diff > timedelta(days=7):
                        raise HTTPException(status_code=400, detail="Time window exceeds 7 days")
                    if time_diff.total_seconds() < 0:
                        raise HTTPException(status_code=400, detail="end_time must be after start_time")

                # Fetch PA trend data
                start_time_str = actual_start.isoformat().replace("+00:00", "Z")
                end_time_str = actual_end.isoformat().replace("+00:00", "Z")

                trend_payload = {
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "station_id": resolved_station_id,
                    "test_items": pa_test_items,
                    "model": model_hint or "",
                }

                trend_data = await client.get_pa_test_items_trend(trend_payload)

                if not trend_data:
                    logger.warning(f"[PA Trend Dex] No trend data for station={station}, dut={dut}")
                    continue

                # Organize by SROM type
                old_items = {}
                new_items = {}

                for item_name, item_data in trend_data.items():
                    mid_value = item_data.get("mid")
                    mean_value = item_data.get("mean")

                    if "_SROM_OLD_" in item_name.upper():
                        base_key = re.sub(r"_SROM_OLD_", "_SROM_", item_name, flags=re.IGNORECASE)
                        old_items[base_key] = {"mid": mid_value, "mean": mean_value}
                    elif "_SROM_NEW_" in item_name.upper():
                        base_key = re.sub(r"_SROM_NEW_", "_SROM_", item_name, flags=re.IGNORECASE)
                        new_items[base_key] = {"mid": mid_value, "mean": mean_value}

                # Calculate adjusted values
                trend_items = []
                for base_key in sorted(set(old_items.keys()) & set(new_items.keys())):
                    old_data = old_items[base_key]
                    new_data = new_items[base_key]

                    adjusted_mid = None
                    if old_data["mid"] is not None and new_data["mid"] is not None:
                        adjusted_mid = round((new_data["mid"] - old_data["mid"]) / 256, 2)

                    adjusted_mean = None
                    if old_data["mean"] is not None and new_data["mean"] is not None:
                        adjusted_mean = round((new_data["mean"] - old_data["mean"]) / 256, 2)

                    adjusted_item_name = re.sub(r"_SROM_", "_TREND_ADJUSTED_POW_", base_key, flags=re.IGNORECASE)

                    trend_items.append(
                        PATrendStationItemSchema(
                            test_item_name=adjusted_item_name,
                            mid=adjusted_mid,
                            mean=adjusted_mean,
                        )
                    )

                trend_items.sort(key=lambda item: _get_pa_srom_sort_key(item.test_item_name))

                if not trend_items:
                    logger.warning(f"[PA Trend Dex] No adjusted items for station={station}, dut={dut}")
                    continue

                logger.info(f"[PA Trend Dex] Created {len(trend_items)} adjusted trend items")

                # Create station record
                station_record = PATrendStationDataSchema(
                    isn=isn_val or dut,
                    station_name=station_name_str,
                    station_id=resolved_station_id,
                    device=device_val,
                    test_date=test_date_val,
                    trend_items=trend_items,
                )
                station_records.append(station_record)
                logger.info(f"[PA Trend Dex] Added station record")

            except Exception as exc:
                logger.error(f"[PA Trend Dex] Error processing station={station}, dut={dut}: {exc}", exc_info=True)
                continue

    if not station_records:
        logger.error(f"[PA Trend Dex] No station records found for dut_isn={dut_isn}, station_id={station_id}")
        raise HTTPException(status_code=404, detail="No PA SROM adjusted trend data found")

    logger.info(f"[PA Trend Dex] Returning {len(station_records)} station records")
    return station_records


@router.get(
    "/pa/trend/diff",
    tags=["DUT_Management"],
    summary="[nonvalue-trend] Calculate PA SROM difference from trend data",
    response_model=list[PADiffStationDataSchema],
    responses={
        200: {"description": "PA SROM difference values (SROM_NEW - SROM_OLD) for the DUT stations."},
        400: {"description": "Invalid parameters or time window exceeds 7 days."},
        404: {"description": "No DUT records found or no PA test items available."},
    },
)
@cache(expire=60)
async def get_pa_srom_diff(
    dut_isn: list[str] = Query(..., description="DUT ISN identifier(s) (e.g., 261534750003154 or DM2527470036123). Can provide multiple."),
    station_id: list[str] = Query(..., description="Station identifier(s) (numeric ID or name) (e.g., 144 or Wireless_Test_6G). Can provide multiple."),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Calculate PA SROM difference (SROM_NEW - SROM_OLD) / 256 from trend data.

    This endpoint:
    1. Fetches PA trend data for the specified DUT(s) and station(s)
    2. Pairs SROM_NEW and SROM_OLD test items by frequency/protocol
    3. Calculates difference: (SROM_NEW - SROM_OLD) / 256 for both mid and mean
    4. Returns difference values with naming pattern: WiFi_PA{X}_DIFF_{frequency}_{protocol}

    Supports multiple station_id and dut_isn parameters (can use IDs or names).

    Example:
        WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80 (mean: 10.5) - WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80 (mean: 10.24)
        Calculation: (10.5 - 10.24) / 256 = 0.001015625
        Result: {"test_item_name": "WiFi_PA1_DIFF_5985_11AX_MCS9_B80", "mid": 0.00, "mean": 0.00}
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    # Collect all station records
    station_records = []

    # Process each combination of dut_isn and station_id
    for dut in dut_isn:
        dut = dut.strip()

        for station in station_id:
            station = station.strip()

            try:
                logger.info(f"[PA Diff] Processing dut={dut}, station={station}")

                # Fetch DUT records to get station metadata
                site_name, model_name, record_data = await _fetch_dut_records(
                    client,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                if not record_data:
                    logger.warning(f"[PA Diff] No DUT records found for dut={dut}")
                    continue

                # Resolve IDs
                resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    station,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
                logger.info(f"[PA Diff] Resolved: station_id={resolved_station_id}, dut_id={resolved_dut_id}")

                # Find the station in the record data
                station_name_str = f"Station {resolved_station_id}"
                isn_val = None
                test_date_val = None
                device_val = None

                for entry in record_data:
                    if _coerce_optional_int(entry.get("id")) == resolved_station_id:
                        station_name_str = entry.get("name") or station_name_str
                        data_entries = entry.get("data") or []
                        if data_entries:
                            latest_entry = None
                            latest_date = datetime.min
                            for data_entry in data_entries:
                                parsed_date = _parse_test_date(data_entry.get("test_date"))
                                if parsed_date and parsed_date >= latest_date:
                                    latest_date = parsed_date
                                    latest_entry = data_entry

                            if latest_entry:
                                test_date_val = latest_entry.get("test_date")
                                device_val = latest_entry.get("device")
                                isn_val = latest_entry.get("isn")
                        break

                logger.info(f"[PA Diff] Station: {station_name_str}, ISN: {isn_val}")

                # Fetch nonvalue records
                payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
                validated = StationRecordResponseSchema.model_validate(payload)

                # Extract all PA SROM test items (both OLD and NEW)
                pa_test_items = []
                for row in validated.data:
                    if not isinstance(row, list) or len(row) < 1:
                        continue
                    test_item_name = str(row[0]) if row else ""
                    if _is_pa_srom_test_item(test_item_name, "all"):
                        pa_test_items.append(test_item_name)

                if not pa_test_items:
                    logger.warning(f"[PA Diff] No PA SROM items for station={station}, dut={dut}")
                    continue

                # Determine time window (24 hours from test_date)
                parsed_test_date = _parse_test_date(test_date_val)
                if parsed_test_date:
                    if parsed_test_date.tzinfo is None:
                        window_end = parsed_test_date.replace(tzinfo=UTC)
                    else:
                        window_end = parsed_test_date.astimezone(UTC)
                else:
                    window_end = datetime.now(UTC)

                window_start = window_end - timedelta(hours=24)

                # Fetch PA trend data
                start_time_str = window_start.isoformat().replace("+00:00", "Z")
                end_time_str = window_end.isoformat().replace("+00:00", "Z")

                trend_payload = {
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "station_id": resolved_station_id,
                    "test_items": pa_test_items,
                    "model": model_hint or "",
                }

                trend_data = await client.get_pa_test_items_trend(trend_payload)

                if not trend_data:
                    logger.warning(f"[PA Diff] No trend data for station={station}, dut={dut}")
                    continue

                # Organize by SROM type
                old_items = {}
                new_items = {}

                for item_name, item_data in trend_data.items():
                    mid_value = item_data.get("mid")
                    mean_value = item_data.get("mean")

                    if "_SROM_OLD_" in item_name.upper():
                        base_key = re.sub(r"_SROM_OLD_", "_SROM_", item_name, flags=re.IGNORECASE)
                        old_items[base_key] = {"mid": mid_value, "mean": mean_value}
                    elif "_SROM_NEW_" in item_name.upper():
                        base_key = re.sub(r"_SROM_NEW_", "_SROM_", item_name, flags=re.IGNORECASE)
                        new_items[base_key] = {"mid": mid_value, "mean": mean_value}

                # Calculate differences
                trend_diff_items = []
                for base_key in sorted(set(old_items.keys()) & set(new_items.keys())):
                    old_data = old_items[base_key]
                    new_data = new_items[base_key]

                    diff_mid = None
                    if old_data["mid"] is not None and new_data["mid"] is not None:
                        diff_mid = round((new_data["mid"] - old_data["mid"]) / 256, 2)

                    diff_mean = None
                    if old_data["mean"] is not None and new_data["mean"] is not None:
                        diff_mean = round((new_data["mean"] - old_data["mean"]) / 256, 2)

                    diff_item_name = re.sub(r"_SROM_", "_DIFF_", base_key, flags=re.IGNORECASE)

                    trend_diff_items.append(
                        PATrendStationItemSchema(
                            test_item_name=diff_item_name,
                            mid=diff_mid,
                            mean=diff_mean,
                        )
                    )

                trend_diff_items.sort(key=lambda item: _get_pa_srom_sort_key(item.test_item_name))

                if not trend_diff_items:
                    logger.warning(f"[PA Diff] No diff items for station={station}, dut={dut}")
                    continue

                logger.info(f"[PA Diff] Created {len(trend_diff_items)} diff items")

                # Create station record
                station_record = PADiffStationDataSchema(
                    isn=isn_val or dut,
                    station_name=station_name_str,
                    station_id=resolved_station_id,
                    device=device_val,
                    test_date=test_date_val,
                    trend_diff_items=trend_diff_items,
                )
                station_records.append(station_record)
                logger.info(f"[PA Diff] Added station record")

            except Exception as exc:
                logger.error(f"[PA Diff] Error processing station={station}, dut={dut}: {exc}", exc_info=True)
                continue

    if not station_records:
        raise HTTPException(status_code=404, detail="No PA SROM difference data found")

    return station_records


@router.get(
    "/pa/trend/new-adjusted",
    tags=["DUT_Management"],
    summary="[nonvalue-trend] Calculate PA latest adjusted power (SROM_NEW trend - SROM_NEW latest)",
    response_model=list[PAActualAdjustedStationDataSchema],
    responses={
        200: {"description": "PA actual adjusted power values for the DUT."},
        400: {"description": "Invalid parameters or time window exceeds 7 days."},
        404: {"description": "No DUT records found or no PA test items available."},
    },
)
@cache(expire=60)
async def get_pa_actual_adjusted(
    dut_isn: list[str] = Query(..., description="DUT ISN identifier(s) (e.g., 261534750003154 or DM2527470036123). Can provide multiple."),
    station_id: list[str] = Query(..., description="Station identifier(s) (numeric ID or name) (e.g., 144 or Wireless_Test_6G). Can provide multiple."),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Calculate PA actual adjusted power (SROM_NEW trend MEAN - SROM_NEW latest decimal value) / 256.

    This endpoint:
    1. Fetches PA trend data (SROM_NEW only) for the specified DUT(s) and station(s)
    2. Fetches latest PA SROM_NEW values (hex converted to decimal)
    3. Calculates difference: (SROM_NEW trend (mean) - SROM_NEW latest (decimal)) / 256
    4. Returns actual adjusted power values with renamed test items (NEW_ADJUSTED)

    Supports multiple station_id and dut_isn parameters (can use IDs or names).

    Example:
        SROM_NEW trend mean: 10.5, SROM_NEW latest: 10 (0x0A)
        Calculation: (10.5 - 10) / 256 = 0.001953125
        Result: {"test_item_name": "WiFi_PA1_NEW_ADJUSTED_5985_11AX_MCS9_B80", "value": 0.00}
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    # Collect all station records
    station_records = []

    # Process each combination of dut_isn and station_id
    for dut in dut_isn:
        dut = dut.strip()

        for station in station_id:
            station = station.strip()

            try:
                # Fetch DUT records
                site_name, model_name, record_data = await _fetch_dut_records(
                    client,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                if not record_data:
                    continue

                # Resolve station_id
                resolved_station_id = await _resolve_station_id(client, station, site_hint=site_hint, model_hint=model_hint)

                # Find the station in the record data
                station_entry = None
                station_name_str = "Unknown Station"
                for entry in record_data:
                    if _coerce_optional_int(entry.get("id")) == resolved_station_id:
                        station_entry = entry
                        station_name_str = entry.get("name") or station_name_str
                        break

                if station_entry is None:
                    continue

                # Get latest data entry
                data_entries = station_entry.get("data") or []
                if not data_entries:
                    continue

                # Find latest entry
                latest_entry = None
                latest_date = datetime.min
                for entry in data_entries:
                    parsed_date = _parse_test_date(entry.get("test_date"))
                    if parsed_date and parsed_date >= latest_date:
                        latest_date = parsed_date
                        latest_entry = entry

                if latest_entry is None:
                    continue

                test_date = latest_entry.get("test_date")
                device = latest_entry.get("device")
                isn = latest_entry.get("isn")

                # Resolve dut_isn to get the proper DUT ID for nonvalue records
                _, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    station,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                # Fetch latest SROM_NEW values (hex to decimal)
                nonvalue_payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
                validated = StationRecordResponseSchema.model_validate(nonvalue_payload)

                # Process and extract SROM_NEW items with decimal values
                latest_srom_new_values = {}
                for row in validated.data:
                    if not isinstance(row, list) or len(row) < 2:
                        continue

                    test_item_name = str(row[0]) if row else ""

                    # Filter for SROM_NEW only
                    if not _is_pa_srom_test_item(test_item_name, "new"):
                        continue

                    latest_value = row[-1] if len(row) > 1 else None
                    decimal_value = _convert_hex_to_decimal(latest_value)

                    if decimal_value is not None:
                        latest_srom_new_values[test_item_name] = decimal_value

                if not latest_srom_new_values:
                    continue

                # Determine time window (24 hours from test_date)
                parsed_test_date = _parse_test_date(test_date)
                if parsed_test_date:
                    if parsed_test_date.tzinfo is None:
                        window_end = parsed_test_date.replace(tzinfo=UTC)
                    else:
                        window_end = parsed_test_date.astimezone(UTC)
                else:
                    window_end = datetime.now(UTC)

                window_start = window_end - timedelta(hours=24)

                # Fetch PA trend data for SROM_NEW items only
                start_time_str = window_start.isoformat().replace("+00:00", "Z")
                end_time_str = window_end.isoformat().replace("+00:00", "Z")

                payload = {
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "station_id": resolved_station_id,
                    "test_items": list(latest_srom_new_values.keys()),
                    "model": model_hint or "",
                }

                trend_data = await client.get_pa_test_items_trend(payload)

                if not trend_data:
                    continue

                # Calculate actual adjusted power (trend mean - latest decimal)
                station_data = []
                for item_name, latest_decimal in latest_srom_new_values.items():
                    trend_item = trend_data.get(item_name)
                    if trend_item is None:
                        continue

                    mean_value = trend_item.get("mean")
                    if mean_value is None:
                        continue

                    adjusted_value = (mean_value - latest_decimal) / 256

                    # Rename from WiFi_PA1_SROM_NEW_... to WiFi_PA1_NEW_ADJUSTED_...
                    adjusted_item_name = re.sub(r"_SROM_NEW_", "_NEW_ADJUSTED_", item_name, flags=re.IGNORECASE)

                    station_data.append(PAItemValueSchema(test_item_name=adjusted_item_name, value=round(adjusted_value, 2)))

                # Sort station data
                station_data.sort(key=lambda item: _get_pa_srom_sort_key(item.test_item_name))

                # Create station record
                station_record = PAActualAdjustedStationDataSchema(
                    isn=isn or dut,  # Use ISN from latest entry or fallback to dut
                    station_name=station_name_str,
                    station_id=resolved_station_id,
                    device=device,
                    test_date=test_date,
                    site_name=site_name,
                    model_name=model_name,
                    time_window_start=start_time_str,
                    time_window_end=end_time_str,
                    data=station_data,
                )
                station_records.append(station_record)

            except Exception as exc:
                logger.warning(f"Error processing dut={dut}, station={station}: {exc}")
                continue

    if not station_records:
        raise HTTPException(status_code=404, detail="No PA adjusted data found for any DUT/station combination")

    return station_records


@router.get(
    "/pa/nonvalue",
    tags=["DUT_Management"],
    summary="[nonvalue] Get actual latest PA SROM test items (both OLD and NEW) with hex-to-decimal conversion",
    response_model=list[PAStationDataSchema],
)
@cache(expire=30)
async def get_latest_pa_srom_all_decimal(
    station_id: list[str] = Query(..., description="Station identifier(s) (numeric ID or name) (e.g., 144 or Wireless_Test_6G). Can provide multiple."),
    dut_id: list[str] = Query(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier(s) (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907). Can provide multiple.",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    srom_filter: Literal["all", "old", "new"] = Query(
        default="all",
        description="Filter PA SROM items: 'all' (both OLD and NEW), 'old' (only SROM_OLD), 'new' (only SROM_NEW)",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get latest PA SROM test items (both SROM_OLD and SROM_NEW patterns) with hexadecimal values converted to decimal.

    Filters test items matching PA{1-4}_SROM_OLD or PA{1-4}_SROM_NEW patterns and converts their hexadecimal
    values to decimal integers. Returns only the latest test record.

    Supports multiple station_id and dut_id parameters (can use IDs or names).

    Example:
        Input: ["WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "0x235e"]
        Output: {"test_item_name": "WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", "value": 9054.0}
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    # Collect all station records
    station_records = []

    # Process each combination of dut_id and station_id
    for dut in dut_id:
        dut = dut.strip()

        for station in station_id:
            station = station.strip()

            try:
                logger.info(f"[PA Nonvalue] Processing dut={dut}, station={station}")

                # Fetch DUT records to get station metadata
                site_name, model_name, record_data = await _fetch_dut_records(
                    client,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                if not record_data:
                    logger.warning(f"[PA Nonvalue] No DUT records found for dut={dut}")
                    continue

                # Resolve IDs
                resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    station,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
                logger.info(f"[PA Nonvalue] Resolved: station_id={resolved_station_id}, dut_id={resolved_dut_id}")

                # Find the station in the record data to get the correct station name
                station_name_str = f"Station {resolved_station_id}"
                isn_val = None
                test_date_val = None
                device_val = None

                for entry in record_data:
                    if _coerce_optional_int(entry.get("id")) == resolved_station_id:
                        station_name_str = entry.get("name") or station_name_str
                        # Get latest data entry for metadata
                        data_entries = entry.get("data") or []
                        if data_entries:
                            # Find latest entry
                            latest_entry = None
                            latest_date = datetime.min
                            for data_entry in data_entries:
                                parsed_date = _parse_test_date(data_entry.get("test_date"))
                                if parsed_date and parsed_date >= latest_date:
                                    latest_date = parsed_date
                                    latest_entry = data_entry

                            if latest_entry:
                                test_date_val = latest_entry.get("test_date")
                                device_val = latest_entry.get("device")
                                isn_val = latest_entry.get("isn")
                        break

                logger.info(f"[PA Nonvalue] Station name: {station_name_str}, ISN: {isn_val}")

                # Fetch nonvalue data
                payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
                logger.info(f"[PA Nonvalue] Fetched nonvalue payload with {len(payload.get('data', []))} data rows")

                validated = StationRecordResponseSchema.model_validate(payload)

                # Process SROM data
                result = _process_pa_srom_data(validated, pattern_type=srom_filter)
                logger.info(f"[PA Nonvalue] Processed SROM data: {len(result.data)} items, {len(result.record)} records")

                # Sort the data array by test_item_name
                if result.data:
                    result.data.sort(key=lambda row: _get_pa_srom_sort_key(row[0]) if row else (float("inf"), 0, 0, ""))

                # Convert to PAItemValueSchema format
                station_data = []
                for row in result.data:
                    if len(row) >= 2:
                        station_data.append(PAItemValueSchema(test_item_name=row[0], value=row[1]))

                if not station_data:
                    logger.warning(f"[PA Nonvalue] No PA SROM data after filtering for station={station}, dut={dut}")
                    continue

                logger.info(f"[PA Nonvalue] Created {len(station_data)} PA items")

                # Create station record
                station_record = PAStationDataSchema(
                    isn=isn_val or dut,  # Use ISN from latest entry or fallback to dut
                    station_name=station_name_str,
                    station_id=resolved_station_id,
                    device=device_val,
                    test_date=test_date_val,
                    site_name=site_name,
                    model_name=model_name,
                    data=station_data,
                )
                station_records.append(station_record)
                logger.info(f"[PA Nonvalue] Added station record")

            except Exception as exc:
                logger.error(f"[PA Nonvalue] Error processing station={station}, dut={dut}: {exc}", exc_info=True)
                continue

    if not station_records:
        logger.error(f"[PA Nonvalue] No station records found for dut_id={dut_id}, station_id={station_id}")
        raise HTTPException(status_code=404, detail="No PA SROM data found for any station/DUT combination")

    logger.info(f"[PA Nonvalue] Returning {len(station_records)} station records")
    return station_records


@router.get(
    "/pa/nonvalue/adjusted",
    tags=["DUT_Management"],
    summary="[nonvalue] Get actual latest PA SROM adjusted values from a DUT",
    response_model=list[PAStationDataSchema],
)
@cache(expire=30)
async def get_latest_pa_srom_adjusted(
    station_id: list[str] = Query(..., description="Station identifier(s) (numeric ID or name) (e.g., 144 or Wireless_Test_6G). Can provide multiple."),
    dut_id: list[str] = Query(
        ...,
        description="DUT ID or ISN/SSN/MAC identifier(s) (numeric DUT ID or DUT ISN) (e.g., 10235789 or 260884980003907). Can provide multiple.",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the station lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the station lookup (e.g., 44 or HH5K).",
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Get latest PA SROM adjusted values calculated as (SROM_NEW - SROM_OLD) / 256.

    Fetches the latest nonvalue record and calculates the adjusted difference for matching
    PA SROM items. Returns only items where both SROM_OLD and SROM_NEW values exist.

    Supports multiple station_id and dut_id parameters (can use IDs or names).

    Example:
        Input:
            SROM_OLD: ["WiFi_PA1_SROM_OLD_2412_11B_CCK11_B20", 9054]
            SROM_NEW: ["WiFi_PA1_SROM_NEW_2412_11B_CCK11_B20", 9310]
        Output: {"test_item_name": "WiFi_PA1_ADJUSTED_2412_11B_CCK11_B20", "value": 1.00}
    """
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    # Collect all station records
    station_records = []

    # Process each combination of dut_id and station_id
    for dut in dut_id:
        dut = dut.strip()

        for station in station_id:
            station = station.strip()

            try:
                logger.info(f"[PA Adjusted] Processing dut={dut}, station={station}")

                # Fetch DUT records to get station metadata
                site_name, model_name, record_data = await _fetch_dut_records(
                    client,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )

                if not record_data:
                    logger.warning(f"[PA Adjusted] No DUT records found for dut={dut}")
                    continue

                # Resolve IDs
                resolved_station_id, resolved_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    station,
                    dut,
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
                logger.info(f"[PA Adjusted] Resolved: station_id={resolved_station_id}, dut_id={resolved_dut_id}")

                # Find the station in the record data to get the correct station name
                station_name_str = f"Station {resolved_station_id}"
                isn_val = None
                test_date_val = None
                device_val = None

                for entry in record_data:
                    if _coerce_optional_int(entry.get("id")) == resolved_station_id:
                        station_name_str = entry.get("name") or station_name_str
                        # Get latest data entry for metadata
                        data_entries = entry.get("data") or []
                        if data_entries:
                            # Find latest entry
                            latest_entry = None
                            latest_date = datetime.min
                            for data_entry in data_entries:
                                parsed_date = _parse_test_date(data_entry.get("test_date"))
                                if parsed_date and parsed_date >= latest_date:
                                    latest_date = parsed_date
                                    latest_entry = data_entry

                            if latest_entry:
                                test_date_val = latest_entry.get("test_date")
                                device_val = latest_entry.get("device")
                                isn_val = latest_entry.get("isn")
                        break

                logger.info(f"[PA Adjusted] Station name: {station_name_str}, ISN: {isn_val}")

                # Fetch nonvalue data
                payload = await client.get_latest_nonvalue_record(resolved_station_id, resolved_dut_id)
                logger.info(f"[PA Adjusted] Fetched nonvalue payload with {len(payload.get('data', []))} data rows")

                validated = StationRecordResponseSchema.model_validate(payload)

                # Process both OLD and NEW SROM data
                processed = _process_pa_srom_data(validated, pattern_type="all")
                logger.info(f"[PA Adjusted] Processed SROM data: {len(processed.data)} items")

                # Organize items by SROM type
                old_items = {}
                new_items = {}

                for row in processed.data:
                    if len(row) < 2:
                        continue
                    item_name = row[0]
                    item_value = row[1]

                    # Extract base key by removing SROM_OLD or SROM_NEW
                    if "_SROM_OLD_" in item_name.upper():
                        base_key = re.sub(r"_SROM_OLD_", "_SROM_", item_name, flags=re.IGNORECASE)
                        old_items[base_key] = item_value
                    elif "_SROM_NEW_" in item_name.upper():
                        base_key = re.sub(r"_SROM_NEW_", "_SROM_", item_name, flags=re.IGNORECASE)
                        new_items[base_key] = item_value

                logger.info(f"[PA Adjusted] Found {len(old_items)} OLD items, {len(new_items)} NEW items")

                # Calculate adjusted values (SROM_NEW - SROM_OLD) / 256
                station_data = []
                for base_key in sorted(set(old_items.keys()) & set(new_items.keys())):
                    adjusted_value = (new_items[base_key] - old_items[base_key]) / 256

                    # Create adjusted item name: WiFi_PA{X}_ADJUSTED_{frequency}_{protocol}
                    adjusted_item_name = re.sub(r"_SROM_", "_ADJUSTED_", base_key, flags=re.IGNORECASE)

                    station_data.append(PAItemValueSchema(test_item_name=adjusted_item_name, value=round(adjusted_value, 2)))

                if not station_data:
                    logger.warning(f"[PA Adjusted] No adjusted data after calculation for station={station}, dut={dut}")
                    continue

                # Sort station data
                station_data.sort(key=lambda item: _get_pa_srom_sort_key(item.test_item_name))

                logger.info(f"[PA Adjusted] Created {len(station_data)} adjusted items")

                # Create station record
                station_record = PAStationDataSchema(
                    isn=isn_val or dut,  # Use ISN from latest entry or fallback to dut
                    station_name=station_name_str,
                    station_id=resolved_station_id,
                    device=device_val,
                    test_date=test_date_val,
                    site_name=site_name,
                    model_name=model_name,
                    data=station_data,
                )
                station_records.append(station_record)
                logger.info(f"[PA Adjusted] Added station record")

            except Exception as exc:
                logger.error(f"[PA Adjusted] Error processing station={station}, dut={dut}: {exc}", exc_info=True)
                continue

    if not station_records:
        logger.error(f"[PA Adjusted] No station records found for dut_id={dut_id}, station_id={station_id}")
        raise HTTPException(status_code=404, detail="No PA adjusted data found for any station/DUT combination")

    logger.info(f"[PA Adjusted] Returning {len(station_records)} station records")
    return station_records


@router.get(
    "/pa/adjusted-power",
    tags=["DUT_Management"],
    summary="[nonvalue-trend] Get PA SROM test items with adjusted power calculations",
    response_model=CombinedPAAdjustedPowerSchema,
    responses={
        200: {"description": "PA SROM test items and adjusted power calculations for the DUT."},
        400: {"description": "Invalid parameters or time window exceeds 7 days."},
        404: {"description": "No DUT records found or no PA test items available."},
    },
)
@cache(expire=60)
async def get_combined_pa_srom_items(
    dut_isn: str = Query(..., description="DUT ISN identifier (e.g., 261534750003154 or DM2527470036123)"),
    station_identifiers: list[str] | None = Query(
        default=None,
        description="Optional list of station identifiers (IDs or names) to filter (e.g., 145 or Wireless_Test_6G). If not provided, all stations with PA SROM items will be included.",
    ),
    site_identifier: str | None = Query(
        None,
        description="Optional site identifier or name to narrow the lookup (e.g., 2 or PTB).",
    ),
    model_identifier: str | None = Query(
        None,
        description="Optional model identifier or name to narrow the lookup (e.g., 44 or HH5K).",
    ),
    start_time: datetime | None = Query(
        None,
        description="Optional start of time window for PA trend calculation (ISO format, e.g., 2023-01-01T00:00:00Z). If not provided, defaults to 24 hours before DUT ISN test_date.",
    ),
    end_time: datetime | None = Query(
        None,
        description="Optional end of time window for PA trend calculation (ISO format, e.g., 2023-01-01T23:59:59Z). If not provided, defaults to DUT ISN test_date.",
    ),
    threshold: float = Query(
        5.0,
        description="Threshold for PA adjusted power scoring (default: 5.0 dB). Score = 10 × (1 - deviation/threshold).",
        gt=0.0,
    ),
    client: DUTAPIClient = dut_client_dependency,
):
    """
    Combined endpoint that retrieves PA SROM test items and calculates adjusted power.

    This endpoint:
    1. Fetches DUT records for the specified ISN
    2. Extracts PA SROM (OLD/NEW) test items from nonvalue records for each station
    3. Gets the latest test values for each SROM test item
    4. Calculates PA adjusted power using the PA trend API
    5. Returns both the raw SROM values and calculated adjusted power (MID and MEAN)

    **Time Window:**
    - If start_time and end_time are not provided, automatically uses 24-hour window from test_date
    - Maximum allowed window: 7 days
    - If test_date is not available, uses current time - 24 hours
    """
    # Trim whitespace from dut_isn
    dut_isn = dut_isn.strip()

    # Fetch DUT records
    site_hint = _select_identifier(site_identifier, label="site")
    model_hint = _select_identifier(model_identifier, label="model")

    site_name, model_name, record_data = await _fetch_dut_records(
        client,
        dut_isn,
        site_hint=site_hint,
        model_hint=model_hint,
    )

    if not record_data:
        raise HTTPException(status_code=404, detail="No test records found for the provided DUT")

    # Prepare station filters
    station_filters = _prepare_station_filters(station_identifiers)

    # Collect PA SROM items from each station
    stations_pa_items: list[dict[str, Any]] = []
    all_pa_test_items: dict[int, list[str]] = {}  # station_id -> list of SROM test items

    for station_entry in record_data:
        station_id = _coerce_optional_int(station_entry.get("id"))
        station_name_entry = station_entry.get("name") or "Unknown Station"

        # Apply station filters
        if station_filters and not _station_matches_filters(station_filters, station_id, station_name_entry):
            continue

        if station_id is None:
            logger.warning("Skipping station with no ID: %s", station_name_entry)
            continue

        # Get latest data entry to determine test_date and dut_id
        data_entries = station_entry.get("data") or []
        if not data_entries:
            continue

        # Find latest entry
        latest_entry = None
        latest_date = datetime.min
        for entry in data_entries:
            parsed_date = _parse_test_date(entry.get("test_date"))
            if parsed_date and parsed_date >= latest_date:
                latest_date = parsed_date
                latest_entry = entry

        if latest_entry is None:
            continue

        logger.debug(
            "Station %s (%s): DUT records test_date=%s",
            station_id,
            station_name_entry,
            latest_entry.get("test_date"),
        )

        # Resolve DUT ID for this station
        candidate_dut_id = latest_entry.get("dut_id")
        if candidate_dut_id is None:
            dut_identifier = latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn") or dut_isn
            try:
                _, candidate_dut_id = await _resolve_station_and_dut_ids(
                    client,
                    str(station_id),
                    str(dut_identifier),
                    site_hint=site_hint,
                    model_hint=model_hint,
                )
            except HTTPException:
                logger.warning("Unable to resolve DUT ID for station %s and DUT %s", station_id, dut_identifier)
                continue

        # Fetch nonvalue records for this station
        try:
            nonvalue_payload = await client.get_station_nonvalue_records(station_id, candidate_dut_id)
            validated_nonvalue = StationRecordResponseSchema.model_validate(nonvalue_payload)
        except Exception as exc:
            logger.warning(
                "Failed to fetch nonvalue records for station %s (%s) and dut %s: %s",
                station_name_entry,
                station_id,
                candidate_dut_id,
                exc,
            )
            continue

        # Extract PA SROM test items (filter for PA{1-4}_SROM_OLD and PA{1-4}_SROM_NEW)
        pa_srom_items: list[dict[str, Any]] = []
        station_pa_test_items: list[str] = []

        for data_row in validated_nonvalue.data:
            if not data_row or len(data_row) < 2:
                continue

            test_item_name = str(data_row[0]).strip()
            if not _is_pa_srom_test_item(test_item_name):
                continue

            # Get latest value (last element in the row)
            latest_value = str(data_row[-1]) if data_row[-1] is not None else None

            pa_srom_items.append({"test_item_name": test_item_name, "latest_value": latest_value})
            station_pa_test_items.append(test_item_name)

        # Only include station if it has PA SROM items
        if not pa_srom_items:
            continue

        # Store PA test items for this station
        all_pa_test_items[station_id] = station_pa_test_items

        stations_pa_items.append(
            {
                "station_id": station_id,
                "station_name": station_name_entry,
                "test_date": latest_entry.get("test_date"),
                "device": latest_entry.get("device_id__name") or latest_entry.get("device"),
                "isn": latest_entry.get("dut_id__isn") or latest_entry.get("dut_isn"),
            }
        )

    if not stations_pa_items:
        raise HTTPException(status_code=404, detail="No PA SROM test items found in any station for the provided DUT")

    # Determine time window for PA trend calculation
    # Use the LATEST test_date from all stations (most recent test)
    latest_test_date: datetime | None = None
    latest_station_id: int | None = None
    for station_data in stations_pa_items:
        test_date = station_data.get("test_date")
        if test_date:
            parsed = _parse_test_date(test_date)
            if parsed and (latest_test_date is None or parsed > latest_test_date):
                latest_test_date = parsed
                latest_station_id = station_data.get("station_id")

    logger.info(
        "Time window calculation - Using test_date from station %s: %s",
        latest_station_id,
        latest_test_date.isoformat() if latest_test_date else "None",
    )

    # Set time window
    if start_time is None or end_time is None:
        # Default: 24 hours window ending at latest test_date
        if latest_test_date is not None:
            # Ensure test_date is timezone-aware (UTC)
            if latest_test_date.tzinfo is None:
                window_end = latest_test_date.replace(tzinfo=UTC)
            else:
                window_end = latest_test_date.astimezone(UTC)
        else:
            # Fallback to current time if no test_date
            window_end = datetime.now(UTC)

        window_start = window_end - timedelta(hours=24)

        if start_time is None:
            start_time = window_start
        if end_time is None:
            end_time = window_end
    else:
        # Ensure user-provided times are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)

        # Validate provided time window
        time_diff = end_time - start_time
        if time_diff > timedelta(days=7):
            raise HTTPException(status_code=400, detail="Time window exceeds maximum allowed 7 days")
        if time_diff.total_seconds() < 0:
            raise HTTPException(status_code=400, detail="end_time must be after start_time")

    logger.info(
        "PA trend time window for DUT %s: start=%s, end=%s (based on latest_test_date=%s)",
        dut_isn,
        start_time.isoformat() if start_time else None,
        end_time.isoformat() if end_time else None,
        latest_test_date.isoformat() if latest_test_date else None,
    )
    logger.debug(
        "Time window details - Window size: %.2f hours, Auto-calculated: %s",
        (end_time - start_time).total_seconds() / 3600,
        "YES (from test_date)" if latest_test_date is not None else "NO (user provided)",
    )

    # Calculate PA adjusted power for each station
    all_adjusted_power_items: list[dict[str, Any]] = []
    all_unpaired_items: list[str] = []

    for station_id, test_items in all_pa_test_items.items():
        if not test_items:
            continue

        # Prepare payload for PA trend API
        # Format timestamps with 'Z' suffix for UTC (external API expects this format)
        start_time_str = start_time.isoformat().replace("+00:00", "Z")
        end_time_str = end_time.isoformat().replace("+00:00", "Z")

        payload = {
            "start_time": start_time_str,
            "end_time": end_time_str,
            "station_id": station_id,
            "test_items": test_items,
            "model": model_hint or "",
        }

        logger.info(
            "Fetching PA trend data for station %s with %d test items in time window %s to %s",
            station_id,
            len(test_items),
            start_time_str,
            end_time_str,
        )
        logger.debug("PA trend payload: %s", payload)
        logger.debug("PA trend test items: %s", test_items)

        # Fetch trend data
        try:
            trend_data = await client.get_pa_test_items_trend(payload)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "PA trend API error for station %s: status=%s, response=%s",
                station_id,
                exc.response.status_code,
                exc.response.text[:500] if exc.response else "N/A",
            )
            continue
        except Exception as exc:
            logger.warning("Failed to fetch PA trend data for station %s: %s", station_id, exc)
            continue

        if not trend_data:
            logger.warning(
                "No PA trend data returned for station %s (empty response from external API). "
                "Possible causes: (1) Time window has no data in trend DB, (2) Test items don't exist in trend DB, "
                "(3) External API needs more time to aggregate data. "
                "Requested test items: %s",
                station_id,
                test_items[:5],  # Show first 5 items
            )
            continue

        logger.info("PA trend data for station %s: %d items returned (expected %d)", station_id, len(trend_data), len(test_items))
        logger.debug("PA trend data keys: %s", list(trend_data.keys()))

        # Check if we got data for any of the requested items
        missing_items = [item for item in test_items if item not in trend_data]
        if missing_items:
            logger.warning(
                "Station %s: External API did not return data for %d/%d test items. Missing items (first 5): %s",
                station_id,
                len(missing_items),
                len(test_items),
                missing_items[:5],
            )

        # Group items by base name and calculate adjusted power
        paired_items: dict[str, dict[str, Any]] = {}

        for item_name, item_data in trend_data.items():
            base_name = _extract_pa_base_name(item_name)
            if not base_name:
                logger.warning("Could not extract base name from: %s", item_name)
                all_unpaired_items.append(item_name)
                continue

            if base_name not in paired_items:
                paired_items[base_name] = {"old_item_name": None, "new_item_name": None, "old_values": None, "new_values": None}

            if "_SROM_OLD_" in item_name.upper():
                paired_items[base_name]["old_item_name"] = item_name
                paired_items[base_name]["old_values"] = item_data
            elif "_SROM_NEW_" in item_name.upper():
                paired_items[base_name]["new_item_name"] = item_name
                paired_items[base_name]["new_values"] = item_data

        logger.debug("Station %s: Found %d potential base name pairs", station_id, len(paired_items))

        # Calculate adjusted power for complete pairs and build restructured response
        completed_pairs_count = 0
        for base_name, pair_data in paired_items.items():
            old_values = pair_data.get("old_values") or {}
            new_values = pair_data.get("new_values") or {}
            old_item_name = pair_data.get("old_item_name")
            new_item_name = pair_data.get("new_item_name")

            # Check if we have both OLD and NEW
            if not old_item_name or not new_item_name:
                logger.warning(
                    "Incomplete pair for %s: old=%s, new=%s",
                    base_name,
                    old_item_name,
                    new_item_name,
                )
                if old_item_name:
                    all_unpaired_items.append(old_item_name)
                if new_item_name:
                    all_unpaired_items.append(new_item_name)
                continue

            # Calculate adjusted power
            adjusted_power_calc = _calculate_adjusted_power(
                old_mid=old_values.get("mid"),
                old_mean=old_values.get("mean"),
                new_mid=new_values.get("mid"),
                new_mean=new_values.get("mean"),
            )

            # Build adjusted power test items with proper naming
            # Only include MEAN items (not MID) per new requirements
            # Convert: WiFi_PA1_5985_11AX_MCS9_B80 -> WiFi_PA1_ADJUSTED_POW_MEAN_5985_11AX_MCS9_B80
            adjusted_power_test_items = {}

            if adjusted_power_calc.get("adjusted_mean") is not None:
                mean_name = re.sub(r"(PA[1-4])_", r"\1_ADJUSTED_POW_MEAN_", base_name, count=1, flags=re.IGNORECASE)
                adjusted_power_test_items[mean_name] = adjusted_power_calc["adjusted_mean"]

            # Skip items with no MEAN value
            if not adjusted_power_test_items:
                continue

            # Build response item with test_pattern and adjusted_power_test_items
            response_item = {
                "test_pattern": base_name,
                "adjusted_power_test_items": adjusted_power_test_items,
            }

            all_adjusted_power_items.append(response_item)
            completed_pairs_count += 1

        logger.info(
            "Station %s: Completed %d pairs, %d unpaired items",
            station_id,
            completed_pairs_count,
            len([item for item in all_unpaired_items if item]),
        )

    logger.info(
        "Combined PA SROM items fetch completed for DUT %s: %d stations, %d adjusted power items, %d unpaired items",
        dut_isn,
        len(stations_pa_items),
        len(all_adjusted_power_items),
        len(all_unpaired_items),
    )

    # Sort adjusted_power_trend by frequency (low to high), then by PA number (PA1, PA2, PA3, PA4)
    def extract_sort_key(item: dict) -> tuple[int, int]:
        """Extract frequency and PA number for sorting.

        Returns (frequency, pa_number) tuple.
        Example: WiFi_PA2_6335_... -> (6335, 2)
        """
        pattern = item.get("test_pattern", "")

        # Extract frequency (4 digits)
        freq_match = re.search(r"_(\d{4})_", pattern)
        frequency = int(freq_match.group(1)) if freq_match else 0

        # Extract PA number (1-4)
        pa_match = re.search(r"PA(\d)", pattern, re.IGNORECASE)
        pa_number = int(pa_match.group(1)) if pa_match else 0

        return (frequency, pa_number)

    all_adjusted_power_items.sort(key=extract_sort_key)

    return {
        "dut_isn": dut_isn,
        "site_name": site_name,
        "model_name": model_name,
        "stations": stations_pa_items,
        "adjusted_power_trend": all_adjusted_power_items,
        "time_window_start": start_time.isoformat().replace("+00:00", "Z"),
        "time_window_end": end_time.isoformat().replace("+00:00", "Z"),
        "unpaired_items": list(set(all_unpaired_items)),
    }


# ============================================================================
# Cache Management Endpoints
# ============================================================================


@router.get(
    "/cache/stats",
    tags=["DUT_Management"],
    summary="Get cache statistics",
    response_model=dict,
)
@cache(expire=30)
async def get_cache_statistics():
    """
    Retrieve cache performance statistics.

    Returns:
        Cache statistics including hit rate, memory usage, and connection status
    """
    from app.utils.dut_cache import get_cache_stats

    try:
        stats = get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Error fetching cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/cache/invalidate",
    tags=["DUT_Management"],
    summary="Invalidate cache entries",
    response_model=dict,
)
async def invalidate_cache_endpoint(
    dut_isn: str | None = Query(None, description="DUT ISN to invalidate (invalidates all related caches)"),
    pattern: str | None = Query(None, description="Redis key pattern to invalidate (e.g., 'dut:records:*')"),
):
    """
        Invalidate cache entries by DUT ISN or pattern.

        Args:
            dut_isn: Specific DUT ISN to invalidate (clears all related caches)
            pattern: Redis key pattern for targeted invalidation
    "
        Returns:
            Number of cache keys deleted

        Examples:
            - Invalidate all caches for DUT: ?dut_isn=ABC123
            - Invalidate all records: ?pattern=dut:records:*
            - Invalidate all caches: (no parameters)
    """
    from app.utils.dut_cache import invalidate_dut_cache

    try:
        deleted = invalidate_dut_cache(dut_isn=dut_isn, pattern=pattern)
        return {
            "deleted": deleted,
            "dut_isn": dut_isn,
            "pattern": pattern,
            "message": f"Invalidated {deleted} cache entries",
        }
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Custom Formula Evaluation ====================


class CustomFormulaEvaluationRequest(BaseModel):
    """Request schema for custom formula evaluation"""

    formula_latex: str
    parameters: dict[str, float]
    measurement_values: dict[str, float]


class CustomFormulaEvaluationResponse(BaseModel):
    """Response schema for custom formula evaluation"""

    score: float
    formula_used: str
    success: bool
    error: str | None = None


@router.post("/evaluate-custom-formula", response_model=CustomFormulaEvaluationResponse)
async def evaluate_custom_formula(request: CustomFormulaEvaluationRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    """
    Evaluate a custom LaTeX formula with provided parameters and measurement values.

    This endpoint safely evaluates mathematical formulas defined in LaTeX notation.
    It supports standard mathematical operations and functions.

    **Security**: Uses restricted eval with only math functions - no code injection possible.

    Args:
        request: Contains formula_latex, parameters, and measurement_values

    Returns:
        Calculated score and formula information

    Example:
        ```json
        {
            "formula_latex": "a \\times x + b",
            "parameters": {"a": 1.5, "b": 10.0},
            "measurement_values": {"x": 5.0}
        }
        ```
        Returns: score = 1.5 * 5.0 + 10.0 = 17.5
    """
    import math
    import re

    try:
        # Strip LaTeX delimiters
        latex_formula = request.formula_latex.strip()
        latex_formula = re.sub(r"^(\$\$?|\\\[|\\\()", "", latex_formula)
        latex_formula = re.sub(r"(\$\$?|\\\]|\\\))$", "", latex_formula)

        # Convert LaTeX to Python expression (simplified conversion)
        python_expr = latex_formula

        # LaTeX to Python conversions
        conversions = [
            (r"\\times", "*"),
            (r"\\div", "/"),
            (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))"),
            (r"\\sqrt\{([^}]+)\}", r"math.sqrt(\1)"),
            (r"\\ln\{([^}]+)\}", r"math.log(\1)"),
            (r"\\ln\(([^)]+)\)", r"math.log(\1)"),
            (r"\\log\{([^}]+)\}", r"math.log10(\1)"),
            (r"\\log\(([^)]+)\)", r"math.log10(\1)"),
            (r"\\exp\{([^}]+)\}", r"math.exp(\1)"),
            (r"\\exp\(([^)]+)\)", r"math.exp(\1)"),
            (r"e\^\{([^}]+)\}", r"math.exp(\1)"),
            (r"e\^([a-zA-Z0-9_]+)", r"math.exp(\1)"),
            (r"\^2", "**2"),
            (r"\^3", "**3"),
            (r"\^([0-9]+)", r"**\1"),
            (r"\^\{([^}]+)\}", r"**(\1)"),
            (r"\\text\{[^}]*\}", ""),  # Remove text labels
            (r"\\mathrm\{[^}]*\}", ""),  # Remove mathrm
        ]

        for pattern, replacement in conversions:
            python_expr = re.sub(pattern, replacement, python_expr)

        # Clean up extra spaces and operators
        python_expr = python_expr.strip()

        # Build safe evaluation namespace
        safe_namespace = {
            "math": math,
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
            "round": round,
            "__builtins__": {},  # Disable all builtins for security
        }

        # Add parameters to namespace
        safe_namespace.update(request.parameters)

        # Add measurement values to namespace
        safe_namespace.update(request.measurement_values)

        # Evaluate the expression
        logger.info(f"Evaluating formula: {python_expr}")
        logger.info(f"With namespace: {list(safe_namespace.keys())}")

        result = eval(python_expr, safe_namespace)

        # Ensure result is a number
        if not isinstance(result, (int, float)):
            raise ValueError(f"Formula evaluation returned non-numeric result: {type(result)}")

        return CustomFormulaEvaluationResponse(score=float(result), formula_used=request.formula_latex, success=True, error=None)

    except Exception as e:
        logger.error(f"Error evaluating custom formula: {e}")
        logger.error(f"Formula: {request.formula_latex}")
        logger.error(f"Parameters: {request.parameters}")
        logger.error(f"Measurements: {request.measurement_values}")
        logger.error(traceback.format_exc())

        return CustomFormulaEvaluationResponse(score=0.0, formula_used=request.formula_latex, success=False, error=str(e))


# ========================================
# Test Log Download Endpoint
# ========================================


@router.post("/test-log/download")
async def download_test_log(request: TestLogDownloadRequest, current_user: Annotated[dict, Depends(get_current_user)]):
    """
    Download test log(s) from External2 (iPLAS) API.

    This endpoint acts as a proxy to iPLAS API's download_attachment endpoint.
    It fetches the test log file(s) as base64-encoded ZIP, decodes it, and returns
    it as a downloadable file to the frontend.

    Supports multiple test logs in a single request.

    Args:
        request: TestLogDownloadRequest containing list of info objects, site, and project
        current_user: Authenticated user (from JWT token)

    Returns:
        StreamingResponse with ZIP file content

    Raises:
        HTTPException: If iPLAS API call fails or returns an error
    """
    try:
        # Get iPLAS API configuration from environment
        base_url = os.getenv("DUT2_API_BASE_URL", "http://10.176.33.89:32678/api/v1")
        token = os.getenv("DUT2_API_TOKEN")
        timeout = int(os.getenv("DUT2_API_TIMEOUT", "30"))

        if not token:
            logger.error("DUT2_API_TOKEN not configured in environment")
            raise HTTPException(status_code=500, detail="External API token not configured")

        # Construct iPLAS API URL
        url = f"{base_url}/file/{request.site}/{request.project}/download_attachment"

        # Prepare request body for iPLAS API
        payload = {
            "info": [
                {
                    "isn": info.isn,
                    "time": info.time,  # Format: YYYY/MM/DD HH:MM:SS
                    "deviceid": info.deviceid,
                    "station": info.station,
                }
                for info in request.info_list
            ],
            "token": token,
        }

        logger.info(f"Calling iPLAS API: {url}")
        logger.info(f"Request payload: {payload} ({len(request.info_list)} test log(s))")

        # Call iPLAS API
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        # Parse response
        response_data = response.json()
        logger.info(f"iPLAS API response status: {response_data.get('statuscode')}")

        # Check for API-level errors
        if response_data.get("statuscode") != 200:
            error_msg = response_data.get("message", "Unknown error from iPLAS API")
            logger.error(f"iPLAS API returned error: {error_msg}")
            raise HTTPException(status_code=502, detail=f"External API error: {error_msg}")

        # Extract base64 content and filename
        data = response_data.get("data", {})
        base64_content = data.get("content")

        # Generate filename
        if len(request.info_list) == 1:
            # Single download: use ISN_station format
            info = request.info_list[0]
            default_filename = f"{info.isn}_{info.station}.zip"
        else:
            # Multiple downloads: use generic name with count
            default_filename = f"test_logs_{len(request.info_list)}_records.zip"

        filename = data.get("filename", default_filename)

        if not base64_content:
            logger.error("No content in iPLAS API response")
            raise HTTPException(status_code=502, detail="External API returned empty content")

        # Decode base64 to binary
        try:
            file_content = base64.b64decode(base64_content)
            logger.info(f"Decoded file size: {len(file_content)} bytes")
        except Exception as e:
            logger.error(f"Failed to decode base64 content: {e}")
            raise HTTPException(status_code=502, detail="Failed to decode file content from External API") from e

        # Return as streaming response
        return StreamingResponse(BytesIO(file_content), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{filename}"', "Content-Length": str(len(file_content))})

    except httpx.TimeoutException as e:
        logger.error(f"Timeout calling iPLAS API after {timeout}s")
        raise HTTPException(status_code=504, detail="External API request timed out") from e
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from iPLAS API: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"External API HTTP error: {e.response.status_code}") from e
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in download_test_log: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e
