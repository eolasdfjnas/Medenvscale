from __future__ import annotations

from copy import deepcopy
from typing import Any

from medenvscale.utils import stable_hash

WEAK_TEST_PATTERNS = [
    "assert inserted_solution_code.strip()",
    "assert solution.strip()",
    "assert code.strip()",
    "assert True",
    "assert 'TODO' not in",
    "compile(",
]


def is_weak_smoke_test(test_code: str) -> bool:
    normalized = test_code.replace(" ", "")
    weak_patterns = [
        "assertinserted_solution_code.strip()",
        "assertsolution.strip()",
        "assertcode.strip()",
        "asserttrue",
        "ortrue",
        "compile(",
    ]
    return any(pattern in normalized for pattern in weak_patterns)


def normalize_verifier_delta(raw_delta: dict[str, Any] | Any, owner_id: str = "") -> dict[str, Any]:
    payload = raw_delta.model_dump() if hasattr(raw_delta, "model_dump") else deepcopy(raw_delta or {})
    if not isinstance(payload, dict):
        raise TypeError("verifier_delta must be a dict")

    normalized = deepcopy(payload)
    normalized["new_checks"] = [_normalize_check_item(item, owner_id) for item in _ensure_list(payload.get("new_checks"))]
    normalized["new_hidden_tests"] = [
        _normalize_hidden_test_item(item, owner_id) for item in _ensure_list(payload.get("new_hidden_tests"))
    ]
    normalized["exception_tests"] = [
        _normalize_exception_test_item(item, owner_id) for item in _ensure_list(payload.get("exception_tests"))
    ]
    normalized["numeric_tolerance_tests"] = [
        _normalize_numeric_tolerance_item(item, owner_id) for item in _ensure_list(payload.get("numeric_tolerance_tests"))
    ]
    normalized["array_close_tests"] = [
        _normalize_generic_test_item(item, owner_id, prefix="array_close") for item in _ensure_list(payload.get("array_close_tests"))
    ]
    normalized["dataframe_equal_tests"] = [
        _normalize_generic_test_item(item, owner_id, prefix="dataframe_equal")
        for item in _ensure_list(payload.get("dataframe_equal_tests"))
    ]
    normalized["file_output_tests"] = [
        _normalize_generic_test_item(item, owner_id, prefix="file_output") for item in _ensure_list(payload.get("file_output_tests"))
    ]
    normalized["object_state_tests"] = [
        _normalize_generic_test_item(item, owner_id, prefix="object_state") for item in _ensure_list(payload.get("object_state_tests"))
    ]
    normalized["static_checks"] = [
        _normalize_static_check_item(item, owner_id) for item in _ensure_list(payload.get("static_checks"))
    ]
    normalized["expected_failure_modes"] = [str(item) for item in _ensure_list(payload.get("expected_failure_modes")) if str(item).strip()]
    return normalized


def _ensure_list(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, list):
        return items
    raise TypeError("verifier_delta items must be lists")


def _normalize_hidden_test_item(item: Any, owner_id: str) -> dict[str, Any]:
    if isinstance(item, str):
        code = item if ("assert" in item or "def test_" in item) else ""
        normalized = {
            "test_id": _stable_id(owner_id, "hidden", item),
            "name": _stable_id(owner_id, "hidden", item),
            "description": "Recovered from raw string verifier delta",
            "code": code,
            "assertion_code": code,
            "source": "fallback",
            "test_tier": "smoke",
            "counts_as_hidden_test": False,
            "eligible_for_clean_export": False,
            "expected_failure_modes": [],
        }
        return _demote_weak_hidden_test(normalized)
    if not isinstance(item, dict):
        raise TypeError("Hidden test item must be dict or string")

    code = _first_text(item, ["code", "assertion_code", "test_code", "execution"])
    description = _first_text(item, ["description"]) or ""
    name = _first_text(item, ["name", "test_id"]) or _stable_id(owner_id, "hidden", item)
    normalized = dict(item)
    normalized["test_id"] = _first_text(item, ["test_id"]) or name
    normalized["name"] = name
    normalized["description"] = description
    normalized["code"] = code
    normalized["assertion_code"] = item.get("assertion_code") or code
    normalized["source"] = str(item.get("source") or "llm")
    normalized["test_tier"] = str(item.get("test_tier") or "semantic")
    normalized["counts_as_hidden_test"] = bool(item.get("counts_as_hidden_test", True))
    normalized["eligible_for_clean_export"] = bool(item.get("eligible_for_clean_export", True))
    normalized["expected_failure_modes"] = [str(mode) for mode in item.get("expected_failure_modes", []) or []]
    return _demote_weak_hidden_test(normalized)


def _demote_weak_hidden_test(test: dict[str, Any]) -> dict[str, Any]:
    code = str(test.get("code") or "")
    if str(test.get("source")) == "fallback" or is_weak_smoke_test(code):
        test["test_tier"] = "smoke"
        test["counts_as_hidden_test"] = False
        test["eligible_for_clean_export"] = False
    return test


def _normalize_exception_test_item(item: Any, owner_id: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "test_id": _stable_id(owner_id, "exception", item),
            "description": "Recovered exception test from raw string verifier delta",
            "code": item if ("assert" in item or "def test_" in item) else None,
            "input": {},
            "expected_exception": "",
            "source": "fallback",
            "eligible_for_clean_export": False,
        }
    if not isinstance(item, dict):
        raise TypeError("Exception test item must be dict or string")
    normalized = dict(item)
    normalized["test_id"] = _first_text(item, ["test_id", "name"]) or _stable_id(owner_id, "exception", item)
    normalized["description"] = _first_text(item, ["description"]) or ""
    normalized["code"] = item.get("code") or item.get("test_code")
    normalized["input"] = item.get("input") or {}
    normalized["expected_exception"] = str(item.get("expected_exception") or "")
    normalized["source"] = str(item.get("source") or "llm")
    normalized["eligible_for_clean_export"] = bool(item.get("eligible_for_clean_export", True))
    return normalized


def _normalize_numeric_tolerance_item(item: Any, owner_id: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "test_id": _stable_id(owner_id, "numeric_tolerance", item),
            "description": "Recovered numeric tolerance test from raw string verifier delta",
            "code": item if ("assert" in item or "def test_" in item) else None,
            "input": {},
            "expected": None,
            "tolerance": None,
            "source": "fallback",
            "eligible_for_clean_export": False,
        }
    if not isinstance(item, dict):
        raise TypeError("numeric_tolerance_test item must be dict or string")
    normalized = dict(item)
    normalized["test_id"] = _first_text(item, ["test_id", "name"]) or _stable_id(owner_id, "numeric_tolerance", item)
    normalized["description"] = _first_text(item, ["description"]) or ""
    normalized["code"] = item.get("code") or item.get("test_code")
    normalized["input"] = item.get("input") or {}
    normalized["expected"] = item.get("expected")
    normalized["tolerance"] = item.get("tolerance")
    normalized["source"] = str(item.get("source") or "llm")
    normalized["eligible_for_clean_export"] = bool(item.get("eligible_for_clean_export", True))
    return normalized


def _normalize_static_check_item(item: Any, owner_id: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "check_id": _stable_id(owner_id, "static", item),
            "name": _stable_id(owner_id, "static", item),
            "description": "",
            "rule": item,
            "pattern": None,
            "must_match": True,
            "source": "fallback",
        }
    if not isinstance(item, dict):
        raise TypeError("Static check item must be dict or string")
    normalized = dict(item)
    normalized["check_id"] = _first_text(item, ["check_id", "name"]) or _stable_id(owner_id, "static", item)
    normalized["name"] = _first_text(item, ["name", "check_id"]) or normalized["check_id"]
    normalized["description"] = _first_text(item, ["description"]) or ""
    normalized["rule"] = _first_text(item, ["rule", "description"]) or ""
    normalized["pattern"] = item.get("pattern")
    normalized["must_match"] = bool(item.get("must_match", True))
    normalized["source"] = str(item.get("source") or "llm")
    return normalized


def _normalize_check_item(item: Any, owner_id: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "check_id": _stable_id(owner_id, "check", item),
            "name": _stable_id(owner_id, "check", item),
            "kind": "generic",
            "rule": item,
            "source": "fallback",
        }
    if not isinstance(item, dict):
        raise TypeError("Verifier check item must be dict or string")
    normalized = dict(item)
    normalized["check_id"] = _first_text(item, ["check_id", "name"]) or _stable_id(owner_id, "check", item)
    normalized["name"] = _first_text(item, ["name", "check_id"]) or normalized["check_id"]
    normalized["kind"] = _first_text(item, ["kind"]) or "generic"
    normalized["rule"] = _first_text(item, ["rule", "description"]) or normalized["kind"]
    normalized["source"] = str(item.get("source") or "llm")
    return normalized


def _normalize_generic_test_item(item: Any, owner_id: str, prefix: str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "test_id": _stable_id(owner_id, prefix, item),
            "description": "Recovered from raw string verifier delta",
            "code": item if ("assert" in item or "def test_" in item) else None,
            "source": "fallback",
            "eligible_for_clean_export": False,
        }
    if not isinstance(item, dict):
        raise TypeError(f"{prefix} test item must be dict or string")
    normalized = dict(item)
    normalized["test_id"] = _first_text(item, ["test_id", "name"]) or _stable_id(owner_id, prefix, item)
    normalized["description"] = _first_text(item, ["description"]) or ""
    normalized["code"] = item.get("code") or item.get("test_code")
    normalized["source"] = str(item.get("source") or "llm")
    normalized["eligible_for_clean_export"] = bool(item.get("eligible_for_clean_export", True))
    return normalized


def _first_text(item: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _stable_id(owner_id: str, prefix: str, value: Any) -> str:
    return f"{prefix}_{stable_hash({'owner_id': owner_id, 'value': value})[:10]}"
