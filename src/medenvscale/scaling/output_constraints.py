from __future__ import annotations

import math
import numbers
import posixpath
import re
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec


def normalize_output_constraint_spec(environment: ExecutableEnvSpec, operator_instances: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    fallback_requirements: list[str] = []

    for operator in operator_instances:
        operator_id = str(operator.get("operator_id") or "")
        for requirement in operator.get("output_requirements", []) or []:
            text = str(requirement).strip()
            if text:
                fallback_requirements.append(text)
        raw_spec = operator.get("output_constraint_spec") or {}
        for check in raw_spec.get("checks", []) if isinstance(raw_spec, dict) else []:
            normalized = _normalize_check(check, operator_id)
            if normalized and normalized["check_id"] not in seen:
                checks.append(normalized)
                seen.add(normalized["check_id"])
        state_updates = operator.get("state_updates") or {}
        for requirement in _requirements_from_state_updates(state_updates):
            fallback_requirements.append(requirement)

    for index, requirement in enumerate(fallback_requirements, start=1):
        check = _compile_requirement(requirement, index=index)
        if check and check["check_id"] not in seen:
            checks.append(check)
            seen.add(check["check_id"])

    return {
        "checks": checks,
        "require_full_coverage": True,
    }


def merge_output_constraint_specs(*specs: dict[str, Any] | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    require_full_coverage = False
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        require_full_coverage = require_full_coverage or bool(spec.get("require_full_coverage"))
        for check in spec.get("checks", []) if isinstance(spec.get("checks"), list) else []:
            normalized = _normalize_check(check, str(check.get("source_operator_id") or "")) if isinstance(check, dict) else None
            if normalized and normalized["check_id"] not in seen:
                checks.append(normalized)
                seen.add(normalized["check_id"])
    return {
        "checks": checks,
        "require_full_coverage": require_full_coverage,
    }


def output_constraints_from_scaled_oracle_cases(scaled_oracle_cases: list[dict[str, Any]] | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(scaled_oracle_cases or [], start=1):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or case.get("example_id") or f"scaled_oracle_case_{index}")
        source_operator_id = str(case.get("targets_operator_id") or "")
        expected = case.get("expected_output_signature")
        if not isinstance(expected, dict):
            continue
        for check in _compile_expected_output_signature(expected, example_id=case_id, source_operator_id=source_operator_id):
            if check["check_id"] in seen:
                continue
            checks.append(check)
            seen.add(check["check_id"])
    return {
        "checks": checks,
        "require_full_coverage": True,
    }


def output_constraints_from_oracle_examples(oracle_examples: list[dict[str, Any]] | None) -> dict[str, Any]:
    return output_constraints_from_scaled_oracle_cases(oracle_examples)


def check_output_constraints(output_signature: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    failed_checks: list[dict[str, Any]] = []
    passed_checks: list[str] = []
    checks = spec.get("checks", []) if isinstance(spec, dict) else []
    for check in checks:
        ok, reason = _run_check(output_signature, check)
        if ok:
            passed_checks.append(str(check.get("check_id") or ""))
        else:
            failed_checks.append(
                {
                    "check_id": str(check.get("check_id") or ""),
                    "reason": reason,
                    "severity": str(check.get("severity") or "hard"),
                    "source_operator_id": str(check.get("source_operator_id") or ""),
                }
            )
    return {
        "passed": not failed_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
    }


def _normalize_check(check: Any, operator_id: str) -> dict[str, Any] | None:
    if not isinstance(check, dict):
        return None
    check_id = str(check.get("check_id") or check.get("name") or "").strip()
    kind = str(check.get("kind") or "").strip()
    rule = str(check.get("rule") or "").strip()
    params = check.get("params") if isinstance(check.get("params"), dict) else {}
    if not check_id or not kind or not rule:
        return None
    return {
        "check_id": check_id,
        "kind": kind,
        "rule": rule,
        "params": params,
        "severity": str(check.get("severity") or "hard"),
        "source_operator_id": str(check.get("source_operator_id") or operator_id),
    }


def _requirements_from_state_updates(state_updates: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    visible = state_updates.get("visible_state_patch") or {}
    task = state_updates.get("task_state_patch") or {}
    for key in ["output_constraints", "format_constraints", "constraint_hints", "execution_requirements"]:
        value = visible.get(key)
        if isinstance(value, list):
            requirements.extend(str(item).strip() for item in value if str(item).strip())
    extra = task.get("extra_constraints")
    if isinstance(extra, list):
        requirements.extend(str(item).strip() for item in extra if str(item).strip())
    return requirements


def _compile_requirement(requirement: str, index: int) -> dict[str, Any] | None:
    text = requirement.strip()
    lowered = text.lower()
    if not text:
        return None
    if "dict" in lowered and ("key" in lowered or "field" in lowered):
        keys = []
        for token in ["mean", "std", "valid_count", "count", "summary"]:
            if token in lowered:
                keys.append(token)
        return {
            "check_id": f"req_{index}_dict_keys",
            "kind": "return_value",
            "rule": "dict_has_keys",
            "params": {"keys": keys} if keys else {"keys": []},
            "severity": "hard",
            "source_operator_id": "",
        }
    if "dict" in lowered:
        return {
            "check_id": f"req_{index}_return_dict",
            "kind": "return_value",
            "rule": "type_is",
            "params": {"type": "dict"},
            "severity": "hard",
            "source_operator_id": "",
        }
    if "stdout" in lowered or "print" in lowered:
        for token in ["mg", "kg", "filtered mean", "summary", "warning"]:
            if token in lowered:
                return {
                    "check_id": f"req_{index}_stdout_contains",
                    "kind": "stdout",
                    "rule": "contains",
                    "params": {"text": token},
                    "severity": "soft",
                    "source_operator_id": "",
                }
        return {
            "check_id": f"req_{index}_stdout_non_empty",
            "kind": "stdout",
            "rule": "non_empty",
            "params": {},
            "severity": "soft",
            "source_operator_id": "",
        }
    if ".json" in lowered or ".csv" in lowered or "write" in lowered or "file" in lowered:
        path = ""
        for token in text.replace(",", " ").split():
            if token.endswith(".json") or token.endswith(".csv") or token.endswith(".txt"):
                path = token.strip("`'\"")
                break
        return {
            "check_id": f"req_{index}_artifact_exists",
            "kind": "file_artifact",
            "rule": "path_exists",
            "params": {"path": path} if path else {},
            "severity": "hard",
            "source_operator_id": "",
        }
    if "sort" in lowered or "ascending" in lowered or "descending" in lowered:
        return {
            "check_id": f"req_{index}_return_sorted",
            "kind": "return_value",
            "rule": "list_sorted",
            "params": {"order": "ascending" if "descending" not in lowered else "descending"},
            "severity": "hard",
            "source_operator_id": "",
        }
    if "preserve output contract" in lowered or "return an integer result" in lowered:
        return {
            "check_id": f"req_{index}_return_non_empty",
            "kind": "return_value",
            "rule": "non_empty",
            "params": {},
            "severity": "hard",
            "source_operator_id": "",
        }
    return None


def _compile_expected_output_signature(
    expected: dict[str, Any],
    *,
    example_id: str,
    source_operator_id: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    return_type = str(expected.get("return_type") or "").strip()
    skip_repr_type_check = _should_skip_repr_type_check(expected, return_type)
    if return_type and not skip_repr_type_check:
        checks.append(
            {
                "check_id": f"{example_id}_return_type",
                "kind": "return_value",
                "rule": "type_is",
                "params": {"type": return_type},
                "severity": "hard",
                "source_operator_id": source_operator_id,
            }
        )

    return_keys = expected.get("return_keys")
    if isinstance(return_keys, list) and return_keys:
        checks.append(
            {
                "check_id": f"{example_id}_return_keys",
                "kind": "return_value",
                "rule": "dict_has_keys",
                "params": {"keys": [str(item) for item in return_keys]},
                "severity": "hard",
                "source_operator_id": source_operator_id,
            }
        )

    if "return_value" in expected:
        expected_value = expected.get("return_value")
        match_mode = str(expected.get("return_value_match") or "").strip().lower()
        if expected_value is None and return_type != "NoneType" and match_mode != "exact":
            expected_value = _MISSING_RETURN_VALUE
        if expected_value is not _MISSING_RETURN_VALUE:
            rule = "dict_contains" if isinstance(expected_value, dict) and match_mode != "exact" else "equals"
            check_id_suffix = "return_value_contains" if rule == "dict_contains" else "return_value_equals"
            checks.append(
                {
                    "check_id": f"{example_id}_{check_id_suffix}",
                    "kind": "return_value",
                    "rule": rule,
                    "params": {"value": expected_value},
                    "severity": "hard",
                    "source_operator_id": source_operator_id,
                }
            )

    return_value_contains = expected.get("return_value_contains")
    if isinstance(return_value_contains, dict):
        checks.append(
            {
                "check_id": f"{example_id}_return_value_contains",
                "kind": "return_value",
                "rule": "dict_contains",
                "params": {"value": return_value_contains},
                "severity": "hard",
                "source_operator_id": source_operator_id,
            }
        )

    stdout_contains = expected.get("stdout_contains")
    if isinstance(stdout_contains, list):
        for position, text in enumerate(stdout_contains, start=1):
            token = str(text).strip()
            if not token:
                continue
            checks.append(
                {
                    "check_id": f"{example_id}_stdout_contains_{position}",
                    "kind": "stdout",
                    "rule": "contains",
                    "params": {"text": token},
                    "severity": "soft",
                    "source_operator_id": source_operator_id,
                }
            )

    stdout_regex = expected.get("stdout_regex")
    if isinstance(stdout_regex, list):
        for position, pattern in enumerate(stdout_regex, start=1):
            token = str(pattern).strip()
            if not token:
                continue
            checks.append(
                {
                    "check_id": f"{example_id}_stdout_regex_{position}",
                    "kind": "stdout",
                    "rule": "regex_match",
                    "params": {"pattern": token},
                    "severity": "soft",
                    "source_operator_id": source_operator_id,
                }
            )

    file_artifacts = expected.get("file_artifacts")
    if isinstance(file_artifacts, list):
        for position, artifact in enumerate(file_artifacts, start=1):
            path = ""
            if isinstance(artifact, dict):
                path = str(artifact.get("path") or "").strip()
            else:
                path = str(artifact).strip()
            if not path:
                continue
            checks.append(
                {
                    "check_id": f"{example_id}_artifact_exists_{position}",
                    "kind": "file_artifact",
                    "rule": "path_exists",
                    "params": {"path": path},
                    "severity": "hard",
                    "source_operator_id": source_operator_id,
                }
            )

    return checks


def _run_check(output_signature: dict[str, Any], check: dict[str, Any]) -> tuple[bool, str]:
    kind = str(check.get("kind") or "")
    rule = str(check.get("rule") or "")
    params = check.get("params") if isinstance(check.get("params"), dict) else {}
    return_value = output_signature.get("return_value")
    observed_return_type = str(output_signature.get("return_type") or type(return_value).__name__)
    stdout = str(output_signature.get("stdout") or "")
    file_artifacts = output_signature.get("file_artifacts") if isinstance(output_signature.get("file_artifacts"), list) else []

    if kind == "return_value" and rule == "type_is":
        expected = params.get("type")
        actual = observed_return_type
        aliases = {"dict": "dict", "list": "list", "str": "str", "int": "int", "float": "float"}
        ok = actual == aliases.get(str(expected), str(expected))
        return ok, f"expected return type {expected}, got {actual}"
    if kind == "return_value" and rule == "dict_has_keys":
        keys = [str(item) for item in params.get("keys", [])]
        if not isinstance(return_value, dict):
            return False, "return value is not a dict"
        missing = [key for key in keys if key not in return_value]
        return not missing, f"missing dict keys: {missing}"
    if kind == "return_value" and rule == "non_empty":
        ok = return_value not in (None, "", [], {}, ())
        return ok, "return value is empty"
    if kind == "return_value" and rule == "equals":
        expected = _normalize_expected_value(params.get("value"))
        actual = _normalize_expected_value(return_value)
        return _values_match(actual, expected), f"expected return value {expected!r}, got {actual!r}"
    if kind == "return_value" and rule == "dict_contains":
        expected = _normalize_expected_value(params.get("value"))
        actual = _normalize_expected_value(return_value)
        ok = _value_contains(actual, expected)
        return ok, f"expected return value to contain {expected!r}, got {actual!r}"
    if kind == "return_value" and rule == "list_sorted":
        if not isinstance(return_value, list):
            return False, "return value is not a list"
        order = str(params.get("order") or "ascending")
        expected = sorted(return_value, reverse=order == "descending")
        return return_value == expected, f"return list is not {order}"
    if kind == "stdout" and rule == "contains":
        text = str(params.get("text") or "")
        ok = text in stdout or _normalize_comparison_text(text) in _normalize_comparison_text(stdout)
        return ok, f"stdout missing substring: {text}"
    if kind == "stdout" and rule == "non_empty":
        return bool(stdout.strip()), "stdout is empty"
    if kind == "stdout" and rule == "regex_match":
        import re

        pattern = str(params.get("pattern") or "")
        try:
            return bool(re.search(pattern, stdout)), f"stdout does not match regex: {pattern}"
        except re.error as exc:
            return False, f"invalid stdout regex: {pattern} ({exc})"
    if kind == "file_artifact" and rule == "path_exists":
        target = str(params.get("path") or "")
        if target:
            exists = any(_artifact_path_matches(target, str(item.get("path") or "")) for item in file_artifacts if isinstance(item, dict))
            return exists, f"missing file artifact: {target}"
        return bool(file_artifacts), "no file artifacts were produced"
    if kind == "file_artifact" and rule == "path_suffix_exists":
        suffix = str(params.get("suffix") or "")
        exists = any(str(item.get("path") or "").endswith(suffix) for item in file_artifacts if isinstance(item, dict))
        return exists, f"missing file artifact with suffix: {suffix}"
    return False, f"unsupported check kind/rule: {kind}/{rule}"


_MISSING_RETURN_VALUE = object()
_DEFAULT_ABS_TOL = 1e-6
_DEFAULT_REL_TOL = 1e-6
_NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _artifact_path_matches(expected: str, observed: str) -> bool:
    expected_norm = _normalize_artifact_path(expected)
    observed_norm = _normalize_artifact_path(observed)
    if not expected_norm or not observed_norm:
        return False
    if expected_norm == observed_norm:
        return True
    if "/" not in expected_norm:
        return posixpath.basename(observed_norm) == expected_norm
    return False


def _normalize_artifact_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    normalized = posixpath.normpath(normalized)
    if normalized == ".":
        return ""
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _value_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual:
                return False
            if not _value_contains(actual[key], value):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_value_contains(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected))
    return _values_match(actual, expected)


def _values_match(actual: Any, expected: Any) -> bool:
    if _both_numeric(actual, expected):
        return _numbers_close(float(actual), float(expected))
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(_values_match(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return _structured_text_values_match(actual, expected)
        return all(_values_match(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected))
    if isinstance(expected, str) or isinstance(actual, str):
        return _structured_text_values_match(actual, expected)
    return actual == expected


def _normalize_expected_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return _normalize_expected_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_normalize_expected_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_expected_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_expected_value(item) for key, item in value.items()}
    return repr(value)


def _both_numeric(actual: Any, expected: Any) -> bool:
    return (
        isinstance(actual, numbers.Real)
        and isinstance(expected, numbers.Real)
        and not isinstance(actual, bool)
        and not isinstance(expected, bool)
    )


def _numbers_close(actual: float, expected: float) -> bool:
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    return math.isclose(actual, expected, rel_tol=_DEFAULT_REL_TOL, abs_tol=_DEFAULT_ABS_TOL)


def _structured_text_values_match(actual: Any, expected: Any) -> bool:
    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    if _normalize_comparison_text(actual_text) == _normalize_comparison_text(expected_text):
        return True
    return _numeric_structured_text_close(actual_text, expected_text)


def _normalize_comparison_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\r\n", "\n").replace("\r", "\n")).strip()


def _numeric_structured_text_close(actual: str, expected: str) -> bool:
    if not (_looks_like_structured_numeric_text(actual) or _looks_like_structured_numeric_text(expected)):
        return False
    actual_numbers = _extract_numbers(actual)
    expected_numbers = _extract_numbers(expected)
    if len(actual_numbers) < 2 or len(actual_numbers) != len(expected_numbers):
        return False
    actual_words = _semantic_word_tokens(actual)
    expected_words = _semantic_word_tokens(expected)
    if actual_words and expected_words and actual_words != expected_words:
        return False
    return all(_numbers_close(actual_item, expected_item) for actual_item, expected_item in zip(actual_numbers, expected_numbers))


def _extract_numbers(text: str) -> list[float]:
    numbers_found: list[float] = []
    for token in _NUMERIC_TOKEN_RE.findall(str(text)):
        try:
            numbers_found.append(float(token))
        except ValueError:
            continue
    return numbers_found


def _semantic_word_tokens(text: str) -> list[str]:
    ignored = {
        "array",
        "dtype",
        "float",
        "float16",
        "float32",
        "float64",
        "int",
        "int16",
        "int32",
        "int64",
        "np",
        "nan",
    }
    return [token.lower() for token in _WORD_TOKEN_RE.findall(str(text)) if token.lower() not in ignored]


def _looks_like_structured_numeric_text(value: Any) -> bool:
    text = str(value)
    if len(_extract_numbers(text)) < 2:
        return False
    lowered = text.lower()
    if "array(" in lowered or "np." in lowered or "dtype=" in lowered:
        return True
    if "[[" in text or "]]" in text:
        return True
    if "\n" in text and re.search(r"\d\s+\d", text):
        return True
    return False


def _should_skip_repr_type_check(expected: dict[str, Any], return_type: str) -> bool:
    if return_type != "str":
        return False
    return _looks_like_structured_numeric_text(expected.get("return_value"))
