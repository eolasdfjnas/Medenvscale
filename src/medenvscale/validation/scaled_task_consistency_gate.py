from __future__ import annotations

from typing import Any

from medenvscale.scaling.hidden_test_runner import run_hidden_test_execution_check
from medenvscale.scaling.quality_filter import is_semantic_hidden_test
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.utils import stable_hash

from .report_schema import build_gate_result, summarize_operator_report

HARD_FAIL_CODES = {
    "SCALED_GOLD_MISSING",
    "GOLD_NOT_UPDATED",
    "SCALED_GOLD_DOES_NOT_MATCH_FINAL_PROMPT",
    "SCALED_GOLD_DOES_NOT_MATCH_VERIFIER_SPECS",
    "SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION",
    "GOLD_INVARIANT_WITHOUT_JUSTIFICATION",
    "GOLD_CONTRADICTS_NEW_CONSTRAINT",
    "VERIFIER_STATE_EMPTY",
    "WEAK_TESTS_WITHOUT_VERIFIER_BACKUP",
    "REQUIREMENT_CHAIN_BROKEN",
    "PROMPT_GOLD_TEST_VERIFIER_MISMATCH",
    "NO_TEST_COVERS_NEW_REQUIREMENT",
}

VISIBLE_REQUIREMENT_FIELDS = [
    "execution_requirements",
    "stepwise_requirements",
    "implicit_requirements",
    "implicit_clues",
    "constraint_hints",
    "output_constraints",
    "format_constraints",
    "must_not_assume",
    "robustness_challenges",
    "robustness_trap",
]

TASK_REQUIREMENT_FIELDS = [
    "visible_requirements",
    "extra_constraints",
    "required_steps",
    "execution_steps",
    "implicit_requirements",
    "safety_critical_constraints",
    "output_format",
    "input_description",
]


def run_scaled_task_consistency_gate(sample: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    seed_env = _load_env(sample, "seed_task")
    scaled_env = _load_env(sample, "scaled_task")
    operator_reports = sample.get("operator_realization_report") or list(scaled_env.operator_realization_report)
    operator_summary = summarize_operator_report(operator_reports)
    added_requirements = extract_added_requirements(sample, seed_env, scaled_env, operator_reports)
    gold_alignment = check_scaled_gold_alignment(seed_env, scaled_env, operator_reports)
    coverage = check_requirement_test_coverage(added_requirements, scaled_env.hidden_tests)
    verifier_alignment = check_verifier_state_alignment(scaled_env, added_requirements, coverage)
    chain = check_requirement_chain_closure(added_requirements, gold_alignment, coverage, verifier_alignment)

    checks = {
        "added_requirements_extracted": bool(added_requirements),
        "scaled_gold_alignment_passed": not gold_alignment["failure_reasons"],
        "requirement_coverage_passed": not coverage["failure_reasons"],
        "verifier_alignment_passed": not verifier_alignment["failure_reasons"],
        "requirement_chain_passed": not chain["failure_reasons"],
    }

    failure_reasons = [
        *gold_alignment["failure_reasons"],
        *coverage["failure_reasons"],
        *verifier_alignment["failure_reasons"],
        *chain["failure_reasons"],
    ]
    warnings = [
        *gold_alignment["warnings"],
        *coverage["warnings"],
        *verifier_alignment["warnings"],
        *chain["warnings"],
    ]

    if operator_summary["severity"] in {"soft_fail", "hard_fail"}:
        warnings.append("OPERATOR_REALIZATION_ALREADY_FAILED")

    return build_gate_result(
        gate_name="scaled_task_consistency_gate",
        checks=checks,
        failure_reasons=failure_reasons,
        warnings=warnings,
        evidence={
            "matched_requirements": [item["requirement_id"] for item in added_requirements],
            "requirement_to_test_coverage": coverage["coverage_rows"],
            "requirement_chain_checks": chain["rows"],
        },
        hard_fail_codes=HARD_FAIL_CODES,
    )


def extract_added_requirements(
    sample: dict[str, Any],
    seed_env: ExecutableEnvSpec,
    scaled_env: ExecutableEnvSpec,
    operator_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    user_prompt = scaled_env.user_prompt or ""

    for field in VISIBLE_REQUIREMENT_FIELDS:
        requirements.extend(_rows_from_value(scaled_env.visible_state.get(field), f"visible_state.{field}", user_prompt))
    for field in TASK_REQUIREMENT_FIELDS:
        requirements.extend(_rows_from_value(scaled_env.task_state.get(field), f"task_state.{field}", user_prompt))

    for operator in scaled_env.operator_instances or []:
        op_id = str(operator.get("operator_id") or "")
        axis = str(operator.get("axis") or "")
        state_updates = operator.get("state_updates") or {}
        for patch_name in ["visible_state_patch", "task_state_patch"]:
            patch = state_updates.get(patch_name) or {}
            for key, value in patch.items():
                requirements.extend(_rows_from_value(value, f"{patch_name}.{key}", user_prompt, axis=axis, operator_id=op_id))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in requirements:
        token = row["requirement_id"]
        if token in seen:
            continue
        seen.add(token)
        deduped.append(row)
    return deduped


def check_scaled_gold_alignment(
    seed_env: ExecutableEnvSpec,
    scaled_env: ExecutableEnvSpec,
    operator_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    gold_state = scaled_env.gold_state or {}
    scaled_gold = str(scaled_env.gold_solution or "").strip()
    seed_gold = str(seed_env.gold_solution or "").strip()
    semantic_change = any(_semantic_change_report(report) for report in operator_reports) or any(
        str(operator.get("axis") or "") != "V" for operator in scaled_env.operator_instances or []
    )
    answer_invariant = bool(gold_state.get("answer_invariant"))
    gold_changed = bool(gold_state.get("gold_changed"))
    change_reason = str(gold_state.get("gold_change_reason") or "").strip()
    seed_gold_compatible = bool(gold_state.get("seed_gold_compatible_with_scaled_task"))

    if not scaled_gold:
        failures.append("SCALED_GOLD_MISSING")
    if semantic_change and scaled_gold == seed_gold and not answer_invariant:
        failures.append("SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION")
    if semantic_change and scaled_gold == seed_gold and answer_invariant and not change_reason:
        failures.append("GOLD_INVARIANT_WITHOUT_JUSTIFICATION")
    if semantic_change and not gold_changed and not answer_invariant:
        failures.append("GOLD_NOT_UPDATED")
    if answer_invariant and not change_reason:
        warnings.append("GOLD_INVARIANT_WITHOUT_JUSTIFICATION")
    if semantic_change and answer_invariant and not seed_gold_compatible:
        warnings.append("SEED_GOLD_COMPATIBILITY_NOT_DECLARED")

    report_prompt_match = all(bool((report.get("gold_checks") or {}).get("scaled_gold_matches_final_prompt", True)) for report in operator_reports)
    report_verifier_match = all(bool((report.get("gold_checks") or {}).get("scaled_gold_matches_verifier_specs", True)) for report in operator_reports)
    if not report_prompt_match:
        failures.append("SCALED_GOLD_DOES_NOT_MATCH_FINAL_PROMPT")

    hidden_test_check = run_hidden_test_execution_check(scaled_env)
    if hidden_test_check.status != "pass":
        failures.append("SCALED_GOLD_DOES_NOT_MATCH_VERIFIER_SPECS")
    elif not report_verifier_match:
        failures.append("SCALED_GOLD_DOES_NOT_MATCH_VERIFIER_SPECS")

    return {
        "failure_reasons": failures,
        "warnings": warnings,
        "semantic_change": semantic_change,
        "scaled_gold_exists": bool(scaled_gold),
        "seed_gold_reused": scaled_gold == seed_gold,
        "answer_invariant": answer_invariant,
        "gold_changed": gold_changed,
        "seed_gold_compatible": seed_gold_compatible,
    }


def check_requirement_test_coverage(added_requirements: list[dict[str, Any]], hidden_tests: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    coverage_rows: list[dict[str, Any]] = []
    covered = 0
    semantic_tests = [test for test in hidden_tests if isinstance(test, dict) and is_semantic_hidden_test(test)]

    for requirement in added_requirements:
        matched = []
        for test in semantic_tests:
            if _test_matches_requirement(test, requirement):
                matched.append(str(test.get("test_id") or test.get("name") or "hidden_test"))
        coverage_rows.append(
            {
                "requirement_id": requirement["requirement_id"],
                "covered_by": matched,
                "coverage_type": "hidden_test" if matched else "missing",
            }
        )
        if matched:
            covered += 1

    if added_requirements and covered == 0:
        failures.append("NO_TEST_COVERS_NEW_REQUIREMENT")
        failures.append("TESTS_DO_NOT_MATCH_PROMPT_REQUIREMENTS")
    elif added_requirements and covered < len(added_requirements):
        warnings.append("ADDED_REQUIREMENT_NOT_TESTED")

    return {
        "failure_reasons": failures,
        "warnings": warnings,
        "coverage_rows": coverage_rows,
        "covered_count": covered,
    }


def check_verifier_state_alignment(
    scaled_env: ExecutableEnvSpec,
    added_requirements: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    verifier_state = scaled_env.verifier_state or {}
    verifier_spec = scaled_env.verifier_spec or {}
    weak_tests = any(not row["covered_by"] for row in coverage["coverage_rows"])
    checks = verifier_spec.get("checks", []) or []
    static_checks = verifier_spec.get("static_checks", []) or []

    if not verifier_state:
        failures.append("VERIFIER_STATE_EMPTY")
    if weak_tests and not verifier_state and not checks and not static_checks:
        failures.append("WEAK_TESTS_WITHOUT_VERIFIER_BACKUP")

    requirement_supported = 0
    for requirement in added_requirements:
        if _verifier_knows_requirement(requirement, verifier_state, verifier_spec):
            requirement_supported += 1

    if added_requirements and requirement_supported == 0:
        failures.append("VERIFIER_DOES_NOT_KNOW_NEW_REQUIREMENT")
        failures.append("VERIFIER_SPECS_NOT_UPDATED")
    elif added_requirements and requirement_supported < len(added_requirements):
        warnings.append("PARTIAL_VERIFIER_REQUIREMENT_COVERAGE")

    return {
        "failure_reasons": failures,
        "warnings": warnings,
        "supported_requirement_count": requirement_supported,
    }


def check_requirement_chain_closure(
    added_requirements: list[dict[str, Any]],
    gold_alignment: dict[str, Any],
    coverage: dict[str, Any],
    verifier_alignment: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for coverage_row in coverage["coverage_rows"]:
        requirement_id = coverage_row["requirement_id"]
        has_test = bool(coverage_row["covered_by"])
        has_gold = gold_alignment["scaled_gold_exists"] and not gold_alignment["failure_reasons"]
        has_verifier = verifier_alignment["supported_requirement_count"] > 0
        chain_complete = has_gold and (has_test or has_verifier)
        rows.append(
            {
                "requirement_id": requirement_id,
                "has_prompt_exposure": True,
                "has_scaled_gold_support": has_gold,
                "has_hidden_test_support": has_test,
                "has_verifier_support": has_verifier,
                "chain_complete": chain_complete,
            }
        )
        if not chain_complete:
            failures.append("REQUIREMENT_CHAIN_BROKEN")

    if added_requirements and failures:
        failures.append("PROMPT_GOLD_TEST_VERIFIER_MISMATCH")
    if added_requirements and not coverage["covered_count"] and verifier_alignment["supported_requirement_count"] > 0:
        warnings.append("VERIFIER_ONLY_REQUIREMENT_COVERAGE")

    return {
        "failure_reasons": failures,
        "warnings": warnings,
        "rows": rows,
    }


def _load_env(sample: dict[str, Any], key: str) -> ExecutableEnvSpec:
    value = sample.get(key)
    if isinstance(value, ExecutableEnvSpec):
        return value
    if isinstance(value, dict):
        return ExecutableEnvSpec.model_validate(value)
    if key == "scaled_task":
        return ExecutableEnvSpec.model_validate(sample)
    raise ValueError(f"Missing environment payload: {key}")


def _rows_from_value(
    value: Any,
    source: str,
    user_prompt: str,
    axis: str | None = None,
    operator_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            rows.append(_build_requirement_row(text, source, user_prompt, axis=axis, operator_id=operator_id))
        return rows
    if isinstance(value, list):
        for item in value:
            rows.extend(_rows_from_value(item, source, user_prompt, axis=axis, operator_id=operator_id))
        return rows
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_rows_from_value(item, f"{source}.{key}", user_prompt, axis=axis, operator_id=operator_id))
        return rows
    return rows


def _build_requirement_row(
    text: str,
    source: str,
    user_prompt: str,
    axis: str | None = None,
    operator_id: str | None = None,
) -> dict[str, Any]:
    return {
        "requirement_id": f"req_{stable_hash({'source': source, 'text': text, 'axis': axis, 'operator_id': operator_id})[:12]}",
        "source": source,
        "text": text,
        "axis": axis or _infer_axis_from_source(source, text),
        "operator_id": operator_id,
        "prompt_exposed": text.lower()[:32] in user_prompt.lower() if user_prompt else False,
    }


def _infer_axis_from_source(source: str, text: str) -> str | None:
    merged = f"{source} {text}".lower()
    if any(token in merged for token in ["shortcut", "hardcoding", "assume"]):
        return "A"
    if any(token in merged for token in ["format", "constraint", "empty input", "boundary", "return contract"]):
        return "C"
    if any(token in merged for token in ["resource", "input", "variant", "nested", "file", "comment line", "extensionless"]):
        return "D"
    if any(token in merged for token in ["test", "verifier"]):
        return "V"
    return None


def _semantic_change_report(report: dict[str, Any]) -> bool:
    gold_checks = report.get("gold_checks") or {}
    return bool(gold_checks.get("semantic_changing_operator", str(report.get("axis") or "") != "V"))


def _test_matches_requirement(test: dict[str, Any], requirement: dict[str, Any]) -> bool:
    operator_match = not requirement.get("operator_id") or str(test.get("targets_operator_id") or "") == str(requirement.get("operator_id") or "")
    axis_match = not requirement.get("axis") or str(test.get("axis") or "") == str(requirement.get("axis") or "")
    requirement_text = str(requirement.get("text") or "").lower()
    test_text = " ".join(
        str(test.get(key) or "")
        for key in ["semantic_intent", "target_constraint", "expected_failure_mode", "description", "code"]
    ).lower()
    text_match = bool(requirement_text and any(token and token in test_text for token in requirement_text.split()[:6]))
    return operator_match and axis_match and text_match


def _verifier_knows_requirement(requirement: dict[str, Any], verifier_state: dict[str, Any], verifier_spec: dict[str, Any]) -> bool:
    requirement_text = str(requirement.get("text") or "").lower()
    if not requirement_text:
        return False
    blobs = [
        verifier_state,
        verifier_spec.get("checks", []),
        verifier_spec.get("static_checks", []),
        verifier_spec.get("hidden_tests", []),
    ]
    merged = " ".join(str(blob).lower() for blob in blobs if blob)
    return any(token and token in merged for token in requirement_text.split()[:6])
