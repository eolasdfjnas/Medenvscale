from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec


def build_output_requirement_metadata(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build structured metadata for the env.output_requirements text table."""

    requirements = _dedupe_strings([str(item).strip() for item in (env.output_requirements or []) if str(item).strip()])
    if not requirements:
        return []

    operators = [item for item in (operator_instances or env.operator_instances or []) if isinstance(item, dict)]
    operator_sources = _operator_requirement_sources(operators)
    seed_requirements = {
        _norm(item)
        for item in ((env.metadata or {}).get("seed_behavior_requirements") or [])
        if str(item).strip()
    }
    counters: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for requirement in requirements:
        requirement_norm = _norm(requirement)
        source = "env.output_requirements"
        operator_id = ""
        axis = "global"
        category = _requirement_category(requirement)

        if requirement_norm in seed_requirements:
            source = "seed_behavior"
            axis = "seed"
            category = "seed_behavior"
        elif requirement_norm in operator_sources:
            source_row = operator_sources[requirement_norm]
            source = "operator.output_requirements"
            operator_id = source_row["operator_id"]
            axis = source_row["axis"]
            category = _requirement_category(requirement)

        counter_key = _safe_id_token(axis or "global")
        counters[counter_key] += 1
        requirement_id = f"req_{counter_key}_{counters[counter_key]:03d}"
        while requirement_id in seen_ids:
            counters[counter_key] += 1
            requirement_id = f"req_{counter_key}_{counters[counter_key]:03d}"
        seen_ids.add(requirement_id)
        rows.append(
            {
                "requirement_id": requirement_id,
                "text": requirement,
                "operator_id": operator_id or None,
                "axis": axis,
                "source": source,
                "category": category,
                "required_coverage": True,
                "visible": True,
            }
        )
    return rows


def requirement_metadata_by_id(env: ExecutableEnvSpec) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("requirement_id") or ""): row
        for row in (env.output_requirement_metadata or [])
        if isinstance(row, dict) and str(row.get("requirement_id") or "")
    }


def infer_covered_requirement_ids(env: ExecutableEnvSpec, case: dict[str, Any]) -> list[str]:
    metadata = [row for row in (env.output_requirement_metadata or []) if isinstance(row, dict)]
    if not metadata:
        return []
    covered_text = _case_requirement_text(case)
    target_ids = {item.strip() for item in str(case.get("targets_operator_id") or "").split(",") if item.strip()}
    case_axes = {item.strip() for item in str(case.get("axis") or "").split(",") if item.strip()}
    inferred: list[str] = []
    for row in metadata:
        req_id = str(row.get("requirement_id") or "")
        if not req_id:
            continue
        row_operator = str(row.get("operator_id") or "")
        row_axis = str(row.get("axis") or "")
        if row_operator and target_ids and row_operator not in target_ids:
            continue
        if row_axis not in {"seed", "global"} and case_axes and row_axis not in case_axes:
            continue
        if requirements_match(str(row.get("text") or ""), covered_text):
            inferred.append(req_id)
    return _dedupe_strings(inferred)


def requirements_match(left: str, right: str) -> bool:
    left_norm = _normalize_requirement(left)
    right_norm = _normalize_requirement(right)
    if not left_norm or not right_norm:
        return False
    if left_norm in right_norm or right_norm in left_norm:
        return True
    left_tokens = _specific_tokens(left_norm)
    right_tokens = _specific_tokens(right_norm)
    if not left_tokens:
        return False
    return len(left_tokens & right_tokens) >= min(2, len(left_tokens))


def requirement_text_match(requirement: str, case: dict[str, Any]) -> bool:
    case_text = _flatten_text(
        [
            case.get("description"),
            case.get("semantic_intent"),
            case.get("target_constraint"),
            case.get("expected_failure_mode"),
            case.get("covered_requirements") or case.get("covers_requirements"),
        ]
    )
    return requirements_match(requirement, case_text)


def requirement_observable_match(requirement: str, case: dict[str, Any]) -> bool:
    observable_text = _flatten_text(
        [
            case.get("setup_code"),
            case.get("call_code"),
            case.get("expected_output_signature"),
            case.get("expected_failure_mode"),
        ]
    )
    req_tokens = _specific_tokens(requirement)
    observable_tokens = _specific_tokens(observable_text)
    if not req_tokens:
        return bool(observable_tokens)
    return bool(req_tokens & observable_tokens)


def _operator_requirement_sources(operators: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for operator in operators:
        operator_id = str(operator.get("operator_id") or "")
        axis = str(operator.get("axis") or "")
        for requirement in operator.get("output_requirements", []) or []:
            text = str(requirement).strip()
            if not text:
                continue
            sources.setdefault(_norm(text), {"operator_id": operator_id, "axis": axis})
    return sources


def _case_requirement_text(case: dict[str, Any]) -> str:
    return _flatten_text(
        [
            case.get("description"),
            case.get("semantic_intent"),
            case.get("target_constraint"),
            case.get("expected_failure_mode"),
            case.get("covered_requirements") or case.get("covers_requirements"),
        ]
    )


def _requirement_category(requirement: str) -> str:
    text = requirement.lower()
    if "seed" in text or "original" in text:
        return "seed_behavior"
    if "format" in text or "stdout" in text or "print" in text:
        return "output_format"
    if "robust" in text or "shortcut" in text or "empty" in text or "missing" in text:
        return "robustness"
    return "scaled_requirement"


def _safe_id_token(value: str) -> str:
    raw = str(value or "global").strip()
    if raw in {"D", "C", "A", "V"}:
        return raw
    token = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return token or "global"


def _normalize_requirement(text: str) -> str:
    return " ".join(_requirement_tokens(str(text)))


def _requirement_tokens(text: str) -> list[str]:
    return [_canonical_requirement_token(token) for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())]


def _canonical_requirement_token(token: str) -> str:
    synonyms = {
        "mutate": "modify",
        "mutated": "modify",
        "mutating": "modify",
        "mutation": "modify",
        "modified": "modify",
        "modifies": "modify",
        "check": "verify",
        "checked": "verify",
        "checks": "verify",
        "prove": "verify",
        "proves": "verify",
        "proving": "verify",
        "test": "verify",
        "tests": "verify",
        "testing": "verify",
        "validated": "validate",
        "validates": "validate",
        "validating": "validate",
    }
    return synonyms.get(token, token)


def _specific_tokens(text: Any) -> set[str]:
    return {
        token
        for token in _requirement_tokens(_flatten_text(text))
        if token not in _generic_tokens()
    }


def _generic_tokens() -> set[str]:
    return {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "be",
        "case",
        "cases",
        "code",
        "constraint",
        "constraints",
        "expected",
        "function",
        "input",
        "new",
        "operator",
        "output",
        "provided",
        "requirement",
        "requirements",
        "return",
        "returns",
        "scaled",
        "should",
        "solution",
        "task",
        "the",
        "to",
        "value",
        "values",
        "with",
    }


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value).lower()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
