from __future__ import annotations

import re
from typing import Any

_OBJECT_MEMORY_ADDRESS_RE = re.compile(r"<(?P<class>[^<>\n]+?) object at 0x[0-9a-fA-F]+>")


def stabilize_runtime_value(value: Any) -> Any:
    if isinstance(value, str):
        return _stable_object_repr(value)
    if isinstance(value, dict):
        return {str(key): stabilize_runtime_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [stabilize_runtime_value(item) for item in value]
    if isinstance(value, list):
        return [stabilize_runtime_value(item) for item in value]
    if isinstance(value, set):
        return sorted(stabilize_runtime_value(item) for item in value)
    return value


def stabilize_expected_output_signature(expected: dict[str, Any]) -> dict[str, Any]:
    stabilized = stabilize_runtime_value(expected)
    return stabilized if isinstance(stabilized, dict) else dict(expected)


def contains_unstable_object_repr(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_OBJECT_MEMORY_ADDRESS_RE.search(value))
    if isinstance(value, dict):
        return any(contains_unstable_object_repr(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_unstable_object_repr(item) for item in value)
    return False


def _stable_object_repr(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        class_name = str(match.group("class") or "object").split(".")[-1]
        return f"<{class_name} object>"

    return _OBJECT_MEMORY_ADDRESS_RE.sub(replace, text)
