from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SpecPayload:
    json_spec: dict[str, Any]

    @property
    def has_json(self) -> bool:
        return bool(self.json_spec)

    @property
    def has_criteria(self) -> bool:
        return False

    @property
    def criteria_rules(self) -> list[Any]:
        return []


def load_spec_payload(data: bytes, filename: str | None = None) -> SpecPayload:
    """
    Parse uploaded specification bytes. Accepts:
    - JSON structures matching the historical compare/multi-DUT configuration format (dict).
    - New simplified JSON array format with test_pattern, usl, lsl, target, gap fields.
    """
    if not data:
        raise ValueError("spec file is empty")

    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Unable to parse spec file as JSON: {exc}") from exc
    
    # Support both old dict format and new array format
    if isinstance(payload, list):
        # New array format: convert to old dict format for backward compatibility
        # Each item has: {test_pattern, usl, lsl, target, gap}
        # Convert to old format for now
        json_spec = {"rules": payload}  # Wrap in dict
        return SpecPayload(json_spec=json_spec)
    elif isinstance(payload, dict):
        # Old dict format
        return SpecPayload(json_spec=payload)
    else:
        raise ValueError("JSON spec must be an object or array")
