from __future__ import annotations

import re
from typing import Any

from medenvscale.scaling.path_safety import analyze_relative_path, extract_code_path_references
from medenvscale.scaling.requirement_registry import (
    infer_covered_requirement_ids,
    requirement_metadata_by_id,
    requirement_observable_match,
    requirement_text_match,
)
from medenvscale.scaling.runtime_value_sanitizer import contains_unstable_object_repr
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.utils import stable_hash


REQUIRED_CASE_FIELDS = [
    "case_id",
    "description",
    "case_kind",
    "targets_operator_id",
    "axis",
    "semantic_intent",
    "target_constraint",
    "expected_failure_mode",
    "call_code",
    "expected_output_signature",
    "covered_requirements",
]

def validate_scaled_oracle_cases(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    cases: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    operators_by_id = {
        str(operator.get("operator_id") or ""): operator
        for operator in operator_instances
        if isinstance(operator, dict) and str(operator.get("operator_id") or "")
    }
    visible_requirements = _visible_requirements(env)
    level = str((env.difficulty.global_level if env.difficulty else "") or "")

    for index, case in enumerate(cases or [], start=1):
        row, failure_reasons = _validate_single_case(
            env=env,
            case=case,
            index=index,
            level=level,
            operators_by_id=operators_by_id,
            visible_requirements=visible_requirements,
        )
        report_rows.append(row)
        if row["valid"]:
            if isinstance(case, dict):
                case["covered_requirement_ids"] = list(row.get("covered_requirement_ids") or [])
                case["bound_requirement_checks"] = list(row.get("bound_requirement_checks") or [])
            validated.append(case)
        else:
            if isinstance(case, dict):
                case["covered_requirement_ids"] = list(row.get("covered_requirement_ids") or [])
                case["bound_requirement_checks"] = list(row.get("bound_requirement_checks") or [])
            invalid.append(case if isinstance(case, dict) else {"case_id": row["case_id"]})

    summary = {
        "env_id": env.env_id,
        "level": level,
        "total_cases": len(cases or []),
        "validated_case_count": len(validated),
        "invalid_case_count": len(invalid),
        "visible_requirements": visible_requirements,
        "operator_ids": sorted(operators_by_id),
    }
    return validated, invalid, report_rows, summary


def _validate_single_case(
    *,
    env: ExecutableEnvSpec,
    case: Any,
    index: int,
    level: str,
    operators_by_id: dict[str, dict[str, Any]],
    visible_requirements: list[str],
) -> tuple[dict[str, Any], list[str]]:
    failure_reasons: list[str] = []
    if not isinstance(case, dict):
        row = _report_row(
            env_id=env.env_id,
            level=level,
            case_id=f"scaled_oracle_case_{index}",
            valid=False,
            failure_reasons=["CASE_NOT_DICT"],
            targets_operator_id="",
            covered_requirements=[],
            covered_requirement_ids=[],
            bound_requirement_checks=[],
            axis="",
        )
        return row, ["CASE_NOT_DICT"]

    case_id = str(case.get("case_id") or f"scaled_oracle_case_{index}")
    case_kind = str(case.get("case_kind") or "")
    is_seed_baseline = case_kind == "seed_baseline"
    is_seed_regression = case_kind == "seed_regression"
    is_seed_case = is_seed_baseline or is_seed_regression
    for field in REQUIRED_CASE_FIELDS:
        if field == "covered_requirements":
            value = case.get("covered_requirements") or case.get("covers_requirements")
        elif field == "case_kind":
            value = case.get("case_kind") or "coverage_extension"
        elif field == "targets_operator_id" and is_seed_case:
            value = "M1"
        else:
            value = case.get(field)
        if field == "expected_output_signature":
            if not isinstance(value, dict) or not value:
                failure_reasons.append("EMPTY_EXPECTED_OUTPUT_SIGNATURE")
        elif field == "covered_requirements":
            if not isinstance(value, list) or not any(str(item).strip() for item in value):
                failure_reasons.append("EMPTY_COVERED_REQUIREMENTS")
        elif not str(value or "").strip():
            failure_reasons.append(f"MISSING_FIELD:{field}")

    targets_operator_id = str(case.get("targets_operator_id") or "")
    target_ids = [item.strip() for item in targets_operator_id.split(",") if item.strip()]
    if not target_ids and not is_seed_case:
        failure_reasons.append("MISSING_TARGET_OPERATOR_ID")
    for operator_id in target_ids:
        if not is_seed_case and operator_id not in operators_by_id:
            failure_reasons.append(f"UNKNOWN_TARGET_OPERATOR_ID:{operator_id}")

    axis = str(case.get("axis") or "")
    declared_axes = {item.strip() for item in axis.split(",") if item.strip()}
    target_axes = {
        str(operators_by_id[operator_id].get("axis") or "")
        for operator_id in target_ids
        if operator_id in operators_by_id
    }
    if declared_axes and target_axes and not declared_axes.issubset(target_axes):
        failure_reasons.append("AXIS_OPERATOR_MISMATCH")

    setup_code = str(case.get("setup_code") or "").strip()
    call_code = str(case.get("call_code") or "").strip()
    if not call_code:
        failure_reasons.append("EMPTY_CALL_CODE")
    elif not _has_result_assignment(call_code):
        failure_reasons.append("CALL_CODE_MUST_ASSIGN_RESULT")

    assertion_code = str(case.get("assertion_code") or "").strip()
    if "assert True" in assertion_code or "assert True" in call_code or "assert True" in setup_code:
        failure_reasons.append("PLACEHOLDER_ASSERTION")

    expected_output_signature = case.get("expected_output_signature")
    if isinstance(expected_output_signature, dict) and not _expected_signature_has_content(expected_output_signature):
        failure_reasons.append("EXPECTED_OUTPUT_SIGNATURE_EMPTY_SHELL")
    if contains_unstable_object_repr(expected_output_signature):
        failure_reasons.append("UNSTABLE_EXPECTED_OUTPUT_OBJECT_MEMORY_ADDRESS")
    failure_reasons.extend(_path_safety_failures(setup_code, call_code, expected_output_signature))

    covered_requirements = [
        str(item).strip()
        for item in ((case.get("covered_requirements") or case.get("covers_requirements")) or [])
        if str(item).strip()
    ]
    metadata_by_id = requirement_metadata_by_id(env)
    explicit_requirement_ids = [
        str(item).strip()
        for item in (case.get("covered_requirement_ids") or [])
        if str(item).strip()
    ]
    inferred_requirement_ids = infer_covered_requirement_ids(env, case) if metadata_by_id else []
    covered_requirement_ids = _dedupe([*explicit_requirement_ids, *inferred_requirement_ids])
    bound_requirement_checks = _bound_requirement_checks(
        case=case,
        covered_requirement_ids=covered_requirement_ids,
        metadata_by_id=metadata_by_id,
        target_ids=target_ids,
        declared_axes=declared_axes,
        is_seed_case=is_seed_case,
    )
    for check in bound_requirement_checks:
        failure_reasons.extend(check.get("failure_reasons", []) or [])
    if metadata_by_id and not is_seed_case and not covered_requirement_ids:
        failure_reasons.append("EMPTY_COVERED_REQUIREMENT_IDS")

    if is_seed_case:
        matched_requirements = list(covered_requirements)
    elif covered_requirements:
        matched_requirements = [item for item in covered_requirements if _matches_visible_requirement(item, visible_requirements)]
        if not matched_requirements:
            failure_reasons.append("COVERED_REQUIREMENTS_NOT_VISIBLE")
        elif not _matches_operator_requirement(case, matched_requirements, target_ids, operators_by_id):
            failure_reasons.append("CASE_DOES_NOT_COVER_NEW_REQUIREMENT")
    else:
        matched_requirements = []
    specificity_failures = [] if is_seed_case else _case_requirement_specificity_failures(case, target_ids, operators_by_id)
    failure_reasons.extend(specificity_failures)

    row = _report_row(
        env_id=env.env_id,
        level=level,
        case_id=case_id,
        valid=not failure_reasons,
        failure_reasons=failure_reasons,
        targets_operator_id=targets_operator_id,
        covered_requirements=covered_requirements,
        covered_requirement_ids=covered_requirement_ids,
        bound_requirement_checks=bound_requirement_checks,
        axis=axis,
    )
    row["matched_requirements"] = matched_requirements
    row["matched_requirement_ids"] = [
        str(check.get("requirement_id") or "")
        for check in bound_requirement_checks
        if check.get("passed")
    ]
    return row, failure_reasons


def _path_safety_failures(setup_code: str, call_code: str, expected_output_signature: Any) -> list[str]:
    failures: list[str] = []
    for source, code in [("setup_code", setup_code), ("call_code", call_code)]:
        for ref in extract_code_path_references(code):
            result = analyze_relative_path(ref.path)
            if not result.safe:
                failures.append(f"UNSAFE_CASE_PATH:{source}:{ref.path}:{result.reason}")
                continue
            if result.has_parent_ref and result.normalizes_to_workdir:
                failures.append(f"UNSAFE_CASE_PATH:{source}:{ref.path}:normalizes_to_workdir")
    if isinstance(expected_output_signature, dict):
        for artifact in expected_output_signature.get("file_artifacts") or []:
            path = str((artifact.get("path") if isinstance(artifact, dict) else artifact) or "").strip()
            result = analyze_relative_path(path, artifact=True)
            if not result.safe:
                code = "DIRECTORY_FILE_ARTIFACT_PATH" if result.directory_artifact else "UNSAFE_FILE_ARTIFACT_PATH"
                failures.append(f"{code}:{path}:{result.reason}")
    return _dedupe(failures)


def _visible_requirements(env: ExecutableEnvSpec) -> list[str]:
    items: list[str] = []
    for requirement in env.output_requirements or []:
        text = str(requirement).strip()
        if text:
            items.append(text)
    visible_state = env.visible_state or {}
    task_state = env.task_state or {}
    for key in [
        "execution_requirements",
        "stepwise_requirements",
        "implicit_requirements",
        "constraint_hints",
        "output_constraints",
        "format_constraints",
        "must_not_assume",
        "robustness_challenges",
    ]:
        value = visible_state.get(key)
        if isinstance(value, list):
            items.extend(str(item).strip() for item in value if str(item).strip())
    for key in ["visible_requirements", "extra_constraints", "required_steps", "execution_steps"]:
        value = task_state.get(key)
        if isinstance(value, list):
            items.extend(str(item).strip() for item in value if str(item).strip())
    items.extend(_additional_requirement_lines(str(env.user_prompt or "")))
    return _dedupe(items)


def _additional_requirement_lines(user_prompt: str) -> list[str]:
    if "Additional requirements:" not in user_prompt:
        return []
    block = user_prompt.split("Additional requirements:", 1)[1].split("Output requirement:", 1)[0]
    lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            stripped = stripped[1:].strip()
        if stripped:
            lines.append(stripped)
    return lines


def _has_result_assignment(call_code: str) -> bool:
    return bool(re.search(r"(^|\n)\s*result\s*=", call_code))


def _expected_signature_has_content(expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "return_value":
            return True
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _matches_visible_requirement(requirement: str, visible_requirements: list[str]) -> bool:
    requirement_norm = _norm(requirement)
    for candidate in visible_requirements:
        candidate_norm = _norm(candidate)
        if requirement_norm and (requirement_norm in candidate_norm or candidate_norm in requirement_norm):
            return True
    return False


def _matches_operator_requirement(
    case: dict[str, Any],
    matched_requirements: list[str],
    target_ids: list[str],
    operators_by_id: dict[str, dict[str, Any]],
) -> bool:
    case_text = " ".join(
        [
            str(case.get("description") or ""),
            str(case.get("case_kind") or ""),
            str(case.get("semantic_intent") or ""),
            str(case.get("target_constraint") or ""),
            " ".join(str(item) for item in ((case.get("covered_requirements") or case.get("covers_requirements")) or [])),
        ]
    )
    case_norm = _norm(case_text)
    if matched_requirements and any(_norm(req) in case_norm or case_norm in _norm(req) for req in matched_requirements if _norm(req)):
        return True
    operator_requirements: list[str] = []
    for operator_id in target_ids:
        operator = operators_by_id.get(operator_id) or {}
        operator_requirements.extend(_operator_requirement_strings(operator))
    for requirement in operator_requirements:
        requirement_norm = _norm(requirement)
        if requirement_norm and (requirement_norm in case_norm or case_norm in requirement_norm):
            return True
    return False


def _case_requirement_specificity_failures(
    case: dict[str, Any],
    target_ids: list[str],
    operators_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    semantic_fields = [
        str(case.get("target_constraint") or ""),
        str(case.get("semantic_intent") or ""),
        str(case.get("description") or ""),
        " ".join(str(item) for item in ((case.get("covered_requirements") or case.get("covers_requirements")) or [])),
    ]
    semantic_text = " ".join(item for item in semantic_fields if item.strip())
    semantic_tokens = _specific_tokens(semantic_text)
    if _is_generic_requirement_text(case.get("target_constraint")) or _is_generic_requirement_text(case.get("semantic_intent")):
        failures.append("CASE_REQUIREMENT_TOO_GENERIC")
    elif len(semantic_tokens) < 2:
        failures.append("CASE_REQUIREMENT_TOO_GENERIC")

    operator_text = " ".join(_operator_specificity_strings(operators_by_id.get(operator_id) or {}) for operator_id in target_ids)
    operator_tokens = _specific_tokens(operator_text)
    new_operator_tokens = operator_tokens - _specific_tokens(_generic_seed_like_text(case))
    if new_operator_tokens:
        operator_tokens = new_operator_tokens
    if operator_tokens and semantic_tokens and not (semantic_tokens & operator_tokens):
        failures.append("CASE_REQUIREMENT_NOT_LINKED_TO_OPERATOR_SEMANTICS")

    observable_text = _flatten_text(
        [
            case.get("setup_code"),
            case.get("call_code"),
            case.get("expected_output_signature"),
            case.get("expected_failure_mode"),
        ]
    )
    observable_tokens = _specific_tokens(observable_text)
    if semantic_tokens and observable_tokens and not (semantic_tokens & observable_tokens):
        failures.append("CASE_EXPECTATION_NOT_LINKED_TO_REQUIREMENT")
    return _dedupe(failures)


def _bound_requirement_checks(
    *,
    case: dict[str, Any],
    covered_requirement_ids: list[str],
    metadata_by_id: dict[str, dict[str, Any]],
    target_ids: list[str],
    declared_axes: set[str],
    is_seed_case: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not metadata_by_id:
        return checks
    for req_id in covered_requirement_ids:
        row = metadata_by_id.get(req_id)
        if not row:
            checks.append(
                {
                    "requirement_id": req_id,
                    "passed": False,
                    "failure_reasons": [f"UNKNOWN_REQUIREMENT_ID:{req_id}"],
                }
            )
            continue
        requirement_text = str(row.get("text") or "")
        operator_id = str(row.get("operator_id") or "")
        axis = str(row.get("axis") or "")
        failure_reasons: list[str] = []
        operator_match = True
        axis_match = True
        if operator_id and target_ids and operator_id not in target_ids:
            operator_match = False
            if not is_seed_case:
                failure_reasons.append(f"REQUIREMENT_OPERATOR_MISMATCH:{req_id}")
        if axis not in {"seed", "global"} and declared_axes and axis not in declared_axes:
            axis_match = False
            if not is_seed_case:
                failure_reasons.append(f"REQUIREMENT_AXIS_MISMATCH:{req_id}")
        text_match = requirement_text_match(requirement_text, case)
        observable_match = requirement_observable_match(requirement_text, case)
        if not text_match and not is_seed_case:
            failure_reasons.append(f"CASE_REQUIREMENT_TEXT_MISMATCH:{req_id}")
        if not observable_match and not is_seed_case:
            failure_reasons.append(f"CASE_EXPECTATION_NOT_LINKED_TO_BOUND_REQUIREMENT:{req_id}")
        checks.append(
            {
                "requirement_id": req_id,
                "requirement": requirement_text,
                "operator_id": operator_id or None,
                "axis": axis,
                "operator_match": operator_match,
                "axis_match": axis_match,
                "text_match": text_match,
                "observable_match": observable_match,
                "passed": not failure_reasons,
                "failure_reasons": failure_reasons,
            }
        )
    return checks


def _operator_specificity_strings(operator: dict[str, Any]) -> str:
    state_updates = operator.get("state_updates") or {}
    verifier_delta = operator.get("verifier_delta") or {}
    parts = [
        operator.get("operator_type"),
        operator.get("transformation_goal"),
        operator.get("rationale"),
        operator.get("output_requirements"),
        state_updates.get("task_state_patch") if isinstance(state_updates, dict) else {},
        state_updates.get("visible_state_patch") if isinstance(state_updates, dict) else {},
        state_updates.get("data_state_patch") if isinstance(state_updates, dict) else {},
        verifier_delta.get("expected_failure_modes") if isinstance(verifier_delta, dict) else [],
    ]
    return _flatten_text(parts)


def _generic_seed_like_text(case: dict[str, Any]) -> str:
    return _flatten_text([case.get("case_kind"), case.get("axis"), case.get("targets_operator_id")])


def _operator_requirement_strings(operator: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for requirement in operator.get("output_requirements", []) or []:
        text = str(requirement).strip()
        if text:
            items.append(text)
    state_updates = operator.get("state_updates") or {}
    visible = state_updates.get("visible_state_patch") or {}
    task = state_updates.get("task_state_patch") or {}
    for key in [
        "execution_requirements",
        "stepwise_requirements",
        "implicit_requirements",
        "constraint_hints",
        "output_constraints",
        "format_constraints",
        "robustness_challenges",
        "must_not_assume",
    ]:
        value = visible.get(key)
        if isinstance(value, list):
            items.extend(str(item).strip() for item in value if str(item).strip())
        elif str(value or "").strip():
            items.append(str(value).strip())
    for key in ["extra_constraints", "required_steps", "execution_steps", "implicit_requirements", "safety_critical_constraints"]:
        value = task.get(key)
        if isinstance(value, list):
            items.extend(str(item).strip() for item in value if str(item).strip())
        elif str(value or "").strip():
            items.append(str(value).strip())
    return _dedupe(items)


def _report_row(
    *,
    env_id: str,
    level: str,
    case_id: str,
    valid: bool,
    failure_reasons: list[str],
    targets_operator_id: str,
    covered_requirements: list[str],
    covered_requirement_ids: list[str],
    bound_requirement_checks: list[dict[str, Any]],
    axis: str,
) -> dict[str, Any]:
    return {
        "env_id": env_id,
        "level": level,
        "case_id": case_id,
        "valid": valid,
        "failure_reasons": failure_reasons,
        "targets_operator_id": targets_operator_id,
        "covered_requirements": covered_requirements,
        "covers_requirements": covered_requirements,
        "covered_requirement_ids": covered_requirement_ids,
        "bound_requirement_checks": bound_requirement_checks,
        "axis": axis,
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _flatten_text(value: Any) -> str:
    parts: list[str] = []
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        for item in value.values():
            parts.append(_flatten_text(item))
        return " ".join(parts)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            parts.append(_flatten_text(item))
        return " ".join(parts)
    return str(value).lower()


def _specific_tokens(value: Any) -> set[str]:
    text = _flatten_text(value)
    raw_tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_*/:.-]*", text)
    tokens = {token.strip(".,;()[]{}'\"") for token in raw_tokens}
    return {token for token in tokens if token and token not in _generic_tokens() and not token.startswith("op_")}


def _generic_tokens() -> set[str]:
    return {
        "a",
        "an",
        "and",
        "any",
        "as",
        "axis",
        "behavior",
        "boundary",
        "case",
        "cases",
        "check",
        "checks",
        "code",
        "computation",
        "constraint",
        "constraints",
        "contract",
        "correct",
        "correctly",
        "data",
        "edge",
        "edge-case",
        "expected",
        "explicit",
        "expose",
        "file",
        "for",
        "format",
        "from",
        "function",
        "gold",
        "handle",
        "handles",
        "input",
        "inputs",
        "intent",
        "introduced",
        "main",
        "must",
        "new",
        "operator",
        "oracle",
        "ordering",
        "output",
        "parameter",
        "parameters",
        "preserve",
        "provided",
        "required",
        "requirement",
        "requirements",
        "respect",
        "return",
        "returns",
        "rule",
        "rules",
        "satisfy",
        "scaled",
        "semantic",
        "solution",
        "task",
        "test",
        "testing",
        "tests",
        "the",
        "this",
        "to",
        "type",
        "value",
        "values",
        "variant",
        "variants",
        "verify",
        "when",
        "with",
    }


def _is_generic_requirement_text(value: Any) -> bool:
    text = _norm(str(value or ""))
    if not text:
        return True
    generic_patterns = [
        "handle scaled input/data variants from the oracle cases",
        "handle scaled input",
        "scaled executable cases must expose",
        "scaled cases add explicit edge-case",
        "scaled cases introduce stronger computation",
        "solution must satisfy new boundary",
        "respect new parameter, boundary, ordering, or return-contract constraints from scaled cases",
        "preserve the required return structure and edge-case behavior",
        "new requirement introduced by this operator",
        "new computation and contract constraints introduced by this operator",
    ]
    if any(pattern in text for pattern in generic_patterns):
        return True
    tokens = _specific_tokens(text)
    return len(tokens) < 2


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        token = stable_hash(item)
        if token in seen:
            continue
        seen.add(token)
        result.append(item)
    return result
