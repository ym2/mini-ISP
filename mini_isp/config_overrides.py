from __future__ import annotations

from typing import Any, Dict, List


def _coerce_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected KEY=VALUE): {item}")
        key, value = item.split("=", 1)
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise ValueError(f"Invalid override key: {item}")
        cursor: Dict[str, Any] = config
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            if not isinstance(cursor[part], dict):
                raise ValueError(f"Override path crosses non-dict key: {part}")
            cursor = cursor[part]
        cursor[parts[-1]] = _coerce_value(value)
    return config
