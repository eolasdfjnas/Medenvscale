from __future__ import annotations

import re
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec

AXIS_FAILURE_CODES = {
    "D": [
        "D_NO_DATA_PATCH",
        "D_NO_VISIBLE_INPUT_REQUIREMENT",
        "D_NO_CASE_FOR_DATA_VARIANT",
        "D_GOLD_DOES_NOT_SUPPORT_DATA_VARIANT",
    ],
    "C": [
        "C_NO_EXTRA_CONSTRAINT",
        "C_CONSTRAINT_NOT_VISIBLE",
        "C_NO_CASE_FOR_CONSTRAINT",
        "C_GOLD_DOES_NOT_SATISFY_CONSTRAINT",
    ],
    "A": [
        "A_NO_SHORTCUT_TRAP",
        "A_ROBUSTNESS_NOT_VISIBLE",
        "A_NO_TARGETED_SHORTCUT_CASE",
        "A_NO_CASE_FOR_ROBUSTNESS",
        "A_GOLD_FALLS_INTO_SHORTCUT",
    ],
    "V": [
        "V_NOT_ENOUGH_CASES",
        "V_CASES_NOT_LINKED",
        "V_NO_GOLD_PASS_SIGNAL",
        "V_GOLD_INVARIANT_NOT_DECLARED",
    ],
}


def check_operator_realizations(seed_task: ExecutableEnvSpec, scaled_task: ExecutableEnvSpec) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for operator_instance in scaled_task.operator_instances or []:
        reports.append(check_operator_realization(seed_task=seed_task, scaled_task=scaled_task, operator_instance=operator_instance))
    return reports


def check_operator_realization(
    seed_task: ExecutableEnvSpec,
    scaled_task: ExecutableEnvSpec,
    operator_instance: dict[str, Any],
    hidden_tests: list[dict[str, Any]] | None = None,
    verifier_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    op_id = str(operator_instance.get("operator_id") or "unknown_operator")
    axis = str(operator_instance.get("axis") or "")
    intensity = int(operator_instance.get("operator_intensity") or 1)
    state_updates = operator_instance.get("state_updates") or {}
    final_user_prompt = scaled_task.user_prompt or ""
    seed_prompt_text = _flatten_text([seed_task.user_prompt, seed_task.problem, seed_task.context])
    linked_cases = _linked_cases(op_id, axis, scaled_task.validated_oracle_cases or scaled_task.scaled_oracle_cases or [])
    linked_case_ids = {str(case.get("case_id") or "") for case in linked_cases if isinstance(case, dict)}
    case_reports = [
        report
        for report in (scaled_task.scaled_gold_case_execution_report or [])
        if isinstance(report, dict) and str(report.get("case_id") or "") in linked_case_ids
    ]
    passed_case_reports = [report for report in case_reports if bool(report.get("passed"))]
    failed_case_reports = [report for report in case_reports if not bool(report.get("passed"))]
    valid_linked_cases = _valid_linked_cases(linked_cases)
    requirement_coverage_matrix = _requirement_coverage_matrix(
        operator_id=op_id,
        axis=axis,
        scaled_task=scaled_task,
        linked_cases=linked_cases,
        case_reports=case_reports,
        final_user_prompt=final_user_prompt,
    )
    prompt_exposure = _has_requirement_exposure(
        operator_instance,
        state_updates,
        linked_cases,
        final_user_prompt,
        seed_prompt_text,
    )
    failure_reasons = _check_axis_specific(
        axis=axis,
        operator_instance=operator_instance,
        state_updates=state_updates,
        linked_cases=linked_cases,
        valid_linked_cases=valid_linked_cases,
        passed_case_reports=passed_case_reports,
        failed_case_reports=failed_case_reports,
        gold_state=scaled_task.gold_state or {},
        final_user_prompt=final_user_prompt,
        prompt_exposure=prompt_exposure,
        hidden_tests=hidden_tests or [],
        verifier_specs=verifier_specs or {},
    )
    seed_gold_reused = str(seed_task.gold_solution or "").strip() == str(scaled_task.gold_solution or "").strip()
    answer_invariant = bool((scaled_task.gold_state or {}).get("answer_invariant", axis == "V"))
    if axis in {"D", "C", "A"} and seed_gold_reused and not answer_invariant:
        failure_reasons.append("SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION")
        failure_reasons = _dedupe(failure_reasons)
    warnings = []
    if axis == "V" and intensity >= 2 and len(linked_cases) < 2:
        warnings.append("INTENSITY_2_PREFER_MORE_VERIFIER_SIGNALS")
    severity = _severity_from_findings(failure_reasons, warnings)
    return {
        "sample_id": scaled_task.env_id,
        "operator_id": op_id,
        "axis": axis,
        "intensity": intensity,
        "severity": severity,
        "passed": severity in {"pass", "pass_with_warning"},
        "global_checks": {
            "has_semantic_patch": _has_semantic_patch(state_updates),
            "has_visible_exposure": prompt_exposure,
            "has_verifier_signal": bool(valid_linked_cases),
            "has_gold_alignment": not any(reason.endswith("GOLD_DOES_NOT_SUPPORT_DATA_VARIANT") or reason.endswith("GOLD_DOES_NOT_SATISFY_CONSTRAINT") or reason.endswith("GOLD_FALLS_INTO_SHORTCUT") for reason in failure_reasons),
            "has_baseline_failure_signal": bool(operator_instance.get("expected_failure_modes") or []),
        },
        "axis_checks": {
            "axis_specific_requirement_satisfied": not failure_reasons,
            "matched_hidden_tests": [],
            "matched_visible_fields": _matched_visible_fields(state_updates, final_user_prompt),
            "matched_case_ids": sorted(linked_case_ids),
            "valid_linked_case_ids": sorted(str(case.get("case_id") or "") for case in valid_linked_cases),
            "passed_linked_case_ids": sorted(str(report.get("case_id") or "") for report in passed_case_reports),
            "has_requirement_exposure": prompt_exposure,
            "requirement_coverage_matrix": requirement_coverage_matrix,
        },
        "gold_checks": {
            "scaled_gold_exists": bool(str(scaled_task.gold_solution or "").strip()),
            "semantic_changing_operator": axis in {"D", "C", "A"},
            "gold_changed": bool((scaled_task.gold_state or {}).get("gold_changed", axis in {"D", "C", "A"})),
            "answer_invariant": bool((scaled_task.gold_state or {}).get("answer_invariant", axis == "V")),
            "seed_gold_reused": seed_gold_reused,
            "seed_gold_compatible_with_scaled_task": bool((scaled_task.gold_state or {}).get("seed_gold_compatible_with_scaled_task", axis == "V")),
            "scaled_gold_matches_final_prompt": True,
            "scaled_gold_matches_verifier_specs": not failed_case_reports,
        },
        "intensity_checks": {
            "supported": not failure_reasons,
            "failure_reasons": [],
            "warnings": warnings,
        },
        "failure_reasons": failure_reasons,
        "warnings": warnings,
    }


def _check_axis_specific(
    axis: str,
    operator_instance: dict[str, Any],
    state_updates: dict[str, Any],
    linked_cases: list[dict[str, Any]],
    valid_linked_cases: list[dict[str, Any]],
    passed_case_reports: list[dict[str, Any]],
    failed_case_reports: list[dict[str, Any]],
    gold_state: dict[str, Any],
    final_user_prompt: str,
    prompt_exposure: bool,
    hidden_tests: list[dict[str, Any]],
    verifier_specs: dict[str, Any],
) -> list[str]:
    task_patch = state_updates.get("task_state_patch") or {}
    data_patch = state_updates.get("data_state_patch") or {}
    visible_patch = state_updates.get("visible_state_patch") or {}
    verifier_patch = state_updates.get("verifier_state_patch") or {}
    reasons: list[str] = []
    has_valid_case = bool(valid_linked_cases)
    has_passed_case = bool(passed_case_reports)
    has_failed_case = bool(failed_case_reports)

    if axis == "D":
        if not _has_data_patch(state_updates):
            reasons.append("D_NO_DATA_PATCH")
        if not prompt_exposure:
            reasons.append("D_NO_VISIBLE_INPUT_REQUIREMENT")
        if not has_valid_case:
            reasons.append("D_NO_CASE_FOR_DATA_VARIANT")
        if has_failed_case:
            reasons.append("D_GOLD_DOES_NOT_SUPPORT_DATA_VARIANT")
    elif axis == "C":
        if not _has_constraint_patch(state_updates):
            reasons.append("C_NO_EXTRA_CONSTRAINT")
        if not prompt_exposure:
            reasons.append("C_CONSTRAINT_NOT_VISIBLE")
        if not has_valid_case:
            reasons.append("C_NO_CASE_FOR_CONSTRAINT")
        if has_failed_case:
            reasons.append("C_GOLD_DOES_NOT_SATISFY_CONSTRAINT")
    elif axis == "A":
        if not _has_robustness_patch(state_updates):
            reasons.append("A_NO_SHORTCUT_TRAP")
        if not prompt_exposure:
            reasons.append("A_ROBUSTNESS_NOT_VISIBLE")
        if not has_valid_case:
            reasons.append("A_NO_TARGETED_SHORTCUT_CASE")
        if not has_valid_case:
            reasons.append("A_NO_CASE_FOR_ROBUSTNESS")
        if has_failed_case:
            reasons.append("A_GOLD_FALLS_INTO_SHORTCUT")
    elif axis == "V":
        if not has_valid_case:
            reasons.append("V_NOT_ENOUGH_CASES")
            reasons.append("V_CASES_NOT_LINKED")
        if not has_passed_case:
            reasons.append("V_NO_GOLD_PASS_SIGNAL")
        if not _v_invariant_declared(operator_instance, gold_state):
            reasons.append("V_GOLD_INVARIANT_NOT_DECLARED")
    return _dedupe(reasons)


def _has_semantic_patch(state_updates: dict[str, Any]) -> bool:
    return any(
        bool(state_updates.get(field))
        for field in [
            "task_state_patch",
            "data_state_patch",
            "visible_state_patch",
            "gold_state_patch",
            "verifier_state_patch",
            "test_state_patch",
        ]
    )


def _has_data_patch(state_updates: dict[str, Any]) -> bool:
    data_patch = state_updates.get("data_state_patch") or {}
    visible_patch = state_updates.get("visible_state_patch") or {}
    task_patch = state_updates.get("task_state_patch") or {}
    if any(
        data_patch.get(key)
        for key in [
            "resource_variants",
            "additional_inputs",
            "input_format_variants",
            "data_variants",
            "fixtures",
            "resource_manifest",
        ]
    ):
        return True
    return _contains_any(_flatten_text([data_patch, visible_patch, task_patch]), ["input", "file", "format", "variant", "data", "resource"])


def _has_constraint_patch(state_updates: dict[str, Any]) -> bool:
    task_patch = state_updates.get("task_state_patch") or {}
    visible_patch = state_updates.get("visible_state_patch") or {}
    verifier_patch = state_updates.get("verifier_state_patch") or {}
    if task_patch.get("extra_constraints") or visible_patch.get("output_constraints") or verifier_patch.get("constraint_checks"):
        return True
    return _contains_any(
        _flatten_text([task_patch, visible_patch, verifier_patch]),
        ["constraint", "validate", "validation", "invalid", "error", "raise", "return", "parameter", "params", "bounds", "contract"],
    )


def _has_robustness_patch(state_updates: dict[str, Any]) -> bool:
    task_patch = state_updates.get("task_state_patch") or {}
    visible_patch = state_updates.get("visible_state_patch") or {}
    verifier_patch = state_updates.get("verifier_state_patch") or {}
    if task_patch.get("shortcut_traps") or visible_patch.get("robustness_trap") or visible_patch.get("must_not_assume"):
        return True
    return _contains_any(
        _flatten_text([task_patch, visible_patch, verifier_patch]),
        [
            "robust",
            "adversarial",
            "shortcut",
            "hardcod",
            "misnamed",
            "wrong extension",
            "duplicate",
            "missing",
            "malformed",
            "non-list",
            "mutation",
            "empty",
            "comment",
            "conflict",
            "misleading",
        ],
    )


def _valid_linked_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_cases = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if not str(case.get("call_code") or "").strip():
            continue
        expected = case.get("expected_output_signature")
        if not isinstance(expected, dict) or not expected:
            continue
        valid_cases.append(case)
    return valid_cases


def _has_requirement_exposure(
    operator_instance: dict[str, Any],
    state_updates: dict[str, Any],
    linked_cases: list[dict[str, Any]],
    final_user_prompt: str,
    seed_prompt_text: str = "",
) -> bool:
    additional_text = _additional_requirements_text(final_user_prompt)
    prompt_tokens = _specific_tokens(additional_text or final_user_prompt)
    if not additional_text:
        prompt_tokens -= _specific_tokens(seed_prompt_text)
    if not prompt_tokens:
        return False
    evidence_text = _flatten_text(
        [
            operator_instance.get("transformation_goal"),
            operator_instance.get("rationale"),
            operator_instance.get("output_requirements"),
            state_updates,
            linked_cases,
        ]
    )
    evidence_tokens = _specific_tokens(evidence_text)
    if not evidence_tokens:
        return False
    new_evidence_tokens = evidence_tokens - _specific_tokens(seed_prompt_text)
    if new_evidence_tokens:
        evidence_tokens = new_evidence_tokens
    overlap = prompt_tokens & evidence_tokens
    return len(overlap) >= min(2, len(evidence_tokens))


def _additional_requirements_text(final_user_prompt: str) -> str:
    text = str(final_user_prompt or "")
    match = re.search(r"Additional requirements:\s*(.*?)(?:\n\n[A-Z][A-Za-z ]+:\s*|\Z)", text, flags=re.DOTALL)
    return match.group(1) if match else ""


def _has_verifier_evidence(
    operator_instance: dict[str, Any],
    state_updates: dict[str, Any],
    hidden_tests: list[dict[str, Any]],
    verifier_specs: dict[str, Any],
) -> bool:
    verifier_patch = state_updates.get("verifier_state_patch") or {}
    test_patch = state_updates.get("test_state_patch") or {}
    verifier_delta = operator_instance.get("verifier_delta") or {}
    return bool(
        verifier_patch
        or test_patch
        or hidden_tests
        or (verifier_specs or {}).get("hidden_tests")
        or verifier_delta.get("new_hidden_tests")
        or verifier_delta.get("new_checks")
        or verifier_delta.get("static_checks")
        or operator_instance.get("semantic_test_specs")
    )


def _v_invariant_declared(operator_instance: dict[str, Any], gold_state: dict[str, Any]) -> bool:
    policy = operator_instance.get("gold_update_policy") if isinstance(operator_instance.get("gold_update_policy"), dict) else {}
    state_updates = operator_instance.get("state_updates") or {}
    gold_patch = state_updates.get("gold_state_patch") if isinstance(state_updates.get("gold_state_patch"), dict) else {}
    return bool(
        policy.get("answer_invariant")
        or gold_patch.get("answer_invariant")
        or gold_state.get("answer_invariant")
        or operator_instance.get("semantic_change") is False
    )


def _has_visible_exposure(state_updates: dict[str, Any], final_user_prompt: str) -> bool:
    return bool(_matched_visible_fields(state_updates, final_user_prompt))


def _matched_visible_fields(state_updates: dict[str, Any], final_user_prompt: str) -> list[str]:
    visible_patch = state_updates.get("visible_state_patch") or {}
    task_patch = state_updates.get("task_state_patch") or {}
    matches: list[str] = []
    for key in ["input_description", "resource_complexity_notes", "output_constraints", "format_constraints", "robustness_trap", "must_not_assume", "execution_requirements"]:
        if _value_visible_in_prompt(visible_patch.get(key), final_user_prompt):
            matches.append(f"visible_state_patch.{key}")
    for key in ["extra_constraints", "shortcut_traps"]:
        if _value_visible_in_prompt(task_patch.get(key), final_user_prompt):
            matches.append(f"task_state_patch.{key}")
    return matches


def _linked_cases(op_id: str, axis: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linked: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        targets = str(case.get("targets_operator_id") or "").strip()
        case_axis = str(case.get("axis") or "").strip()
        if targets and any(token.strip() == op_id for token in targets.split(",")):
            linked.append(case)
        elif not targets and case_axis == axis:
            linked.append(case)
    return linked


def _requirement_coverage_matrix(
    *,
    operator_id: str,
    axis: str,
    scaled_task: ExecutableEnvSpec,
    linked_cases: list[dict[str, Any]],
    case_reports: list[dict[str, Any]],
    final_user_prompt: str,
) -> list[dict[str, Any]]:
    metadata = [
        row
        for row in (scaled_task.output_requirement_metadata or [])
        if isinstance(row, dict)
        and str(row.get("operator_id") or "") == operator_id
        and str(row.get("axis") or "") == axis
    ]
    if not metadata:
        return []
    reports_by_case = {str(row.get("case_id") or ""): row for row in case_reports if isinstance(row, dict)}
    matrix: list[dict[str, Any]] = []
    for requirement in metadata:
        req_id = str(requirement.get("requirement_id") or "")
        req_text = str(requirement.get("text") or "")
        linked_for_req = [
            case
            for case in linked_cases
            if req_id in {str(item).strip() for item in (case.get("covered_requirement_ids") or [])}
        ]
        case_ids = [str(case.get("case_id") or "") for case in linked_for_req if str(case.get("case_id") or "")]
        gold_passed = bool(case_ids) and all(bool(reports_by_case.get(case_id, {}).get("passed")) for case_id in case_ids)
        text_match = any(
            any(
                isinstance(check, dict)
                and str(check.get("requirement_id") or "") == req_id
                and bool(check.get("text_match"))
                for check in (case.get("bound_requirement_checks") or [])
            )
            for case in linked_for_req
        )
        observable_match = any(
            any(
                isinstance(check, dict)
                and str(check.get("requirement_id") or "") == req_id
                and bool(check.get("observable_match"))
                for check in (case.get("bound_requirement_checks") or [])
            )
            for case in linked_for_req
        )
        visible_in_prompt = _requirement_text_visible(req_text, final_user_prompt)
        if not case_ids:
            status = "missing_case"
        elif not text_match or not observable_match:
            status = "case_mismatch"
        elif not gold_passed:
            status = "gold_failed"
        else:
            status = "pass"
        matrix.append(
            {
                "operator_id": operator_id,
                "axis": axis,
                "requirement_id": req_id,
                "requirement": req_text,
                "visible_in_prompt": visible_in_prompt,
                "case_ids": case_ids,
                "text_match": text_match,
                "observable_match": observable_match,
                "gold_passed": gold_passed,
                "status": status,
            }
        )
    return matrix


def _linked_case_text(cases: list[dict[str, Any]]) -> str:
    return " ".join(
        " ".join(
            [
                str(case.get("description") or ""),
                str(case.get("semantic_intent") or ""),
                str(case.get("target_constraint") or ""),
                " ".join(str(item) for item in (case.get("covered_requirements") or []) if str(item).strip()),
            ]
        )
        for case in cases
        if isinstance(case, dict)
    ).lower()


def _value_visible_in_prompt(value: Any, final_user_prompt: str) -> bool:
    text = str(final_user_prompt or "").lower()
    if not text or value is None:
        return False
    if isinstance(value, list):
        return any(_requirement_text_visible(item, final_user_prompt) for item in value if str(item).strip())
    return _requirement_text_visible(value, final_user_prompt)


def _requirement_text_visible(value: Any, final_user_prompt: str) -> bool:
    text = str(value or "").strip().lower()
    if not text or _is_generic_requirement(text):
        return False
    prompt = str(final_user_prompt or "").lower()
    if text in prompt:
        return True
    tokens = _specific_tokens(text)
    if not tokens:
        return False
    return len(tokens & _specific_tokens(prompt)) >= min(2, len(tokens))


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


def _contains_any(text: str, needles: list[str]) -> bool:
    normalized = str(text or "").lower()
    return any(needle in normalized for needle in needles)


def _important_tokens(text: Any) -> set[str]:
    stopwords = {
        "the",
        "and",
        "or",
        "with",
        "that",
        "this",
        "must",
        "should",
        "shall",
        "case",
        "cases",
        "test",
        "tests",
        "task",
        "solution",
        "function",
        "return",
        "returns",
        "input",
        "output",
        "file",
        "value",
        "values",
        "expected",
        "include",
        "includes",
        "provided",
        "when",
        "then",
        "than",
        "into",
        "from",
        "for",
        "not",
        "none",
        "true",
        "false",
    }
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", _flatten_text(text))
        if token not in stopwords and len(token) >= 3
    }
    return tokens


def _specific_tokens(text: Any) -> set[str]:
    return {
        token
        for token in _important_tokens(text)
        if token not in _generic_exposure_tokens()
    }


def _generic_exposure_tokens() -> set[str]:
    return {
        "additional",
        "behavior",
        "boundary",
        "case",
        "cases",
        "computation",
        "constraint",
        "constraints",
        "contract",
        "correctly",
        "data",
        "edge",
        "edge-case",
        "executable",
        "expected",
        "explicit",
        "expose",
        "format",
        "gold",
        "handle",
        "input",
        "introduce",
        "introduced",
        "new",
        "operator",
        "oracle",
        "output-contract",
        "ordering",
        "parameter",
        "parameters",
        "preserve",
        "requirement",
        "requirements",
        "required",
        "respect",
        "return-contract",
        "rules",
        "scaled",
        "stronger",
        "structure",
        "testing",
        "variant",
        "variants",
        "verifier",
    }


def _is_generic_requirement(value: Any) -> bool:
    text = str(value or "").strip().lower()
    generic_patterns = [
        "handle scaled input/data variants from the oracle cases",
        "scaled executable cases must expose",
        "scaled cases add explicit edge-case",
        "scaled cases introduce stronger computation",
        "solution must satisfy new boundary",
        "respect new parameter, boundary, ordering, or return-contract constraints from scaled cases",
        "preserve the required return structure and edge-case behavior",
        "new requirement introduced by this operator",
        "new computation and contract constraints introduced by this operator",
    ]
    return any(pattern in text for pattern in generic_patterns)


def _severity_from_findings(failure_reasons: list[str], warnings: list[str]) -> str:
    if failure_reasons:
        return "hard_fail"
    if warnings:
        return "pass_with_warning"
    return "pass"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
