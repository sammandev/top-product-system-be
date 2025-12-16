from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_CRITERIA_PATH = Path(os.getenv("DUT_CRITERIA_PATH", "reference/conf_dut_criteria.ini"))


@dataclass(slots=True)
class CriteriaRule:
    pattern: re.Pattern[str]
    usl: float | None
    lsl: float | None
    target: float | None
    gap: float | None = None  # Added gap/margin support


# Updated pattern to support both old <USL,LSL> ===> "Target" and new <USL,LSL,Target,Gap> formats
_CRITERIA_LINE_PATTERN_OLD = re.compile(r'^\s*"(?P<test>.+?)"\s*<(?P<usl>[^,]*),(?P<lsl>[^>]*)>\s*===>\s*"(?P<target>.*)"\s*$')
_CRITERIA_LINE_PATTERN_NEW = re.compile(r'^\s*"(?P<test>.+?)"\s*<(?P<usl>[^,]*),(?P<lsl>[^,]*),(?P<target>[^,]*),(?P<gap>[^>]*)>\s*$')


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


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


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


def strip_test_item_digits(value: str) -> str:
    tokens = _tokenize_test_item(value)
    cleaned: list[str] = []
    for token in tokens:
        match = re.match(r"^(tx|rx|pa|ant|rssi)(\d{1,2})$", token)
        if match:
            cleaned.append(match.group(1))
            continue
        cleaned.append(token)
    return "_".join(cleaned)


def parse_criteria_line(line: str) -> CriteriaRule | None:
    # Try new format first: <USL,LSL,Target,Gap>
    match = _CRITERIA_LINE_PATTERN_NEW.match(line.strip())
    if match:
        test_pattern = match.group("test")
        usl = _to_float(match.group("usl"))
        lsl = _to_float(match.group("lsl"))
        target = _to_float(match.group("target"))
        gap = _to_float(match.group("gap"))
        try:
            compiled = re.compile(test_pattern, re.IGNORECASE)
        except re.error:
            return None
        return CriteriaRule(pattern=compiled, usl=usl, lsl=lsl, target=target, gap=gap)
    
    # Try old format: <USL,LSL> ===> "Target"
    match = _CRITERIA_LINE_PATTERN_OLD.match(line.strip())
    if match:
        test_pattern = match.group("test")
        usl = _to_float(match.group("usl"))
        lsl = _to_float(match.group("lsl"))
        target = _to_float(match.group("target"))
        try:
            compiled = re.compile(test_pattern, re.IGNORECASE)
        except re.error:
            return None
        return CriteriaRule(pattern=compiled, usl=usl, lsl=lsl, target=target, gap=None)
    
    return None


def parse_criteria_content(lines: Iterable[str]) -> dict[str, list[CriteriaRule]]:
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
        rule = parse_criteria_line(line)
        if rule is None:
            continue
        key = normalize_key(current_station)
        station_rules.setdefault(key, []).append(rule)
    return station_rules


@lru_cache(maxsize=8)
def load_criteria_from_path(path: str | Path) -> dict[str, list[CriteriaRule]]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Criteria file not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    return parse_criteria_content(lines)


def load_default_criteria() -> dict[str, list[CriteriaRule]]:
    try:
        return load_criteria_from_path(DEFAULT_CRITERIA_PATH)
    except FileNotFoundError:
        return {}


def load_criteria_from_bytes(data: bytes) -> dict[str, list[CriteriaRule]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
    return parse_criteria_content(text.splitlines())


def select_station_rules(
    criteria_map: dict[str, list[CriteriaRule]] | None,
    station_name: str | None,
    model_name: str | None = None,
) -> list[CriteriaRule]:
    if not criteria_map:
        return []
    normalized_station = normalize_key(station_name)
    normalized_model = normalize_key(model_name)
    selected: list[CriteriaRule] = []
    for raw_key, rules in criteria_map.items():
        if not rules:
            continue
        key = normalize_key(raw_key)
        if "|" in key:
            model_part, station_part = key.split("|", 1)
            if station_part == normalized_station and (not model_part or model_part == normalized_model):
                selected.extend(rules)
        elif key == normalized_station:
            selected.extend(rules)
    if selected:
        return selected
    # fallback: return all rules when no station specified or no match found
    if not normalized_station:
        aggregated: list[CriteriaRule] = []
        for rules in criteria_map.values():
            aggregated.extend(rules)
        return aggregated
    return []


def match_rule(rules: list[CriteriaRule], test_item: str) -> CriteriaRule | None:
    if not rules:
        return None
    sanitized_name = strip_test_item_digits(test_item)
    for rule in rules:
        if rule.pattern.search(test_item):
            return rule
        if rule.pattern.search(sanitized_name):
            return rule
        sanitized_pattern = strip_test_item_digits(rule.pattern.pattern)
        if sanitized_pattern and sanitized_pattern in sanitized_name:
            return rule
    return None


def determine_target_value(
    rule: CriteriaRule | None,
    usl: float | None,
    lsl: float | None,
    actual: float | None,
    test_item_name: str | None = None,
) -> float | None:
    if actual is None:
        return None
    if rule is not None:
        # Check if rule has explicit target - this takes highest priority
        if rule.target is not None:
            return rule.target
        candidate_usl = rule.usl if rule.usl is not None else usl
        candidate_lsl = rule.lsl if rule.lsl is not None else lsl
    else:
        candidate_usl = usl
        candidate_lsl = lsl
    # For PER test items, default to 0.0 instead of midpoint (but can be overridden by rule.target above)
    if test_item_name and ("PER" in test_item_name.upper() or "_PER_" in test_item_name.upper()):
        return 0.0
    if candidate_usl is not None and candidate_lsl is not None:
        return (candidate_usl + candidate_lsl) / 2
    if candidate_usl is not None:
        return candidate_usl
    if candidate_lsl is not None:
        return candidate_lsl
    return actual
