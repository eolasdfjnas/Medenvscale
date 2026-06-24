from __future__ import annotations

from typing import Any

from medenvscale.scaling.hidden_test_runner import run_hidden_test_execution_check
from medenvscale.scaling.quality_filter import is_semantic_hidden_test
from medenvscale.scaling.verifier_delta_normalizer import is_weak_smoke_test
from medenvscale.schemas import ExecutableEnvSpec

from .report_schema import build_gate_result

HARD_FAIL_CODES = {
    "HIDDEN_TEST_PARSE_ERROR",
    "HIDDEN_TEST_COMPILE_ERROR",
    "HIDDEN_TEST_EXECUTION_ERROR",
    "NO_VALID_SEMANTIC_HIDDEN_TEST",
    "SMOKE_TEST_ONLY",
    "PLACEHOLDER_TEST",
    "ASSERT_TRUE_TEST",
    "M4_COMBINED_ORACLE_CASE_MISSING",
}


def run_hidden_tests_quality_gate(sample: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    env = _env_from_sample(sample)
    hidden_tests = [test for test in env.hidden_tests if isinstance(test, dict)]
    execution = run_hidden_test_execution_check(env)
    counts = _count_semantic_tests(hidden_tests)
    v_intensity = int((env.scaling or env.scaling_plan or {}).get("axis_intensity", {}).get("V", 0))
    scaled_oracle_cases = [item for item in (env.scaled_oracle_cases or []) if isinstance(item, dict)]
    oracle_check = _scaled_oracle_case_check(env, scaled_oracle_cases)

    checks = {
        "compile_execute_passed": execution.status == "pass",
        "weak_test_filter_passed": not counts["weak_failures"],
        "semantic_intent_passed": not counts["semantic_metadata_failures"],
        "count_passed": _semantic_count_sufficient(v_intensity, counts["semantic_effective_count"]),
        "targeting_passed": not counts["targeting_failures"],
        "oracle_example_count_passed": not oracle_check["failure_reasons"],
    }

    failure_reasons: list[str] = []
    warnings: list[str] = []
    evidence = {
        "matched_tests": counts["semantic_test_ids"],
        "matched_operators": counts["linked_operator_ids"],
        "matched_requirements": counts["target_constraints"],
        "semantic_test_count": counts["semantic_effective_count"],
        "v_intensity": v_intensity,
        "scaled_oracle_case_ids": [str(item.get("case_id") or item.get("example_id") or "") for item in scaled_oracle_cases],
        "scaled_oracle_case_count": len(scaled_oracle_cases),
    }

    failure_reasons.extend(_map_execution_errors(execution.errors))
    failure_reasons.extend(counts["weak_failures"])
    failure_reasons.extend(counts["semantic_metadata_failures"])
    failure_reasons.extend(counts["targeting_failures"])
    failure_reasons.extend(oracle_check["failure_reasons"])
    warnings.extend(oracle_check["warnings"])

    if v_intensity and counts["semantic_effective_count"] == 0:
        failure_reasons.append("NO_VALID_SEMANTIC_HIDDEN_TEST")
    elif not _semantic_count_sufficient(v_intensity, counts["semantic_effective_count"]):
        failure_reasons.append("V_NOT_ENOUGH_SEMANTIC_TESTS")

    if not hidden_tests:
        failure_reasons.append("NO_VALID_SEMANTIC_HIDDEN_TEST")

    if counts["missing_expected_failure_mode"]:
        warnings.append("MISSING_EXPECTED_FAILURE_MODE")
    if counts["weak_linkage_warning"]:
        warnings.append("WEAK_OPERATOR_TEST_LINKAGE")

    return build_gate_result(
        gate_name="hidden_tests_quality_gate",
        checks=checks,
        failure_reasons=failure_reasons,
        warnings=warnings,
        evidence=evidence,
        hard_fail_codes=HARD_FAIL_CODES,
    )


def _env_from_sample(sample: dict[str, Any]) -> ExecutableEnvSpec:
    env = sample.get("scaled_task")
    if isinstance(env, ExecutableEnvSpec):
        return env
    if isinstance(sample.get("scaled_task"), dict):
        return ExecutableEnvSpec.model_validate(sample["scaled_task"])
    if isinstance(sample.get("env"), ExecutableEnvSpec):
        return sample["env"]
    return ExecutableEnvSpec.model_validate(sample)


def _map_execution_errors(errors: list[str]) -> list[str]:
    mapped: list[str] = []
    for error in errors:
        if "context_compile" in error or "parse" in error:
            mapped.append("HIDDEN_TEST_PARSE_ERROR")
        elif "compile" in error:
            mapped.append("HIDDEN_TEST_COMPILE_ERROR")
        elif "execution" in error:
            mapped.append("HIDDEN_TEST_EXECUTION_ERROR")
    return mapped


def _count_semantic_tests(hidden_tests: list[dict[str, Any]]) -> dict[str, Any]:
    weak_failures: list[str] = []
    semantic_metadata_failures: list[str] = []
    targeting_failures: list[str] = []
    semantic_test_ids: list[str] = []
    linked_operator_ids: list[str] = []
    target_constraints: list[str] = []
    weak_linkage_warning = False
    missing_expected_failure_mode = False
    semantic_effective_count = 0

    for test in hidden_tests:
        test_id = str(test.get("test_id") or test.get("name") or "hidden_test")
        code = str(test.get("code") or test.get("assertion_code") or "")
        test_tier = str(test.get("test_tier") or "")
        source = str(test.get("source") or "")
        semantic_intent = str(test.get("semantic_intent") or "").strip()
        target_constraint = str(test.get("target_constraint") or "").strip()
        expected_failure_mode = _first_nonempty(
            test.get("expected_failure_mode"),
            ",".join(str(item) for item in test.get("expected_failure_modes", []) or []),
        )
        operator_id = str(test.get("targets_operator_id") or "").strip()
        axis = str(test.get("axis") or "").strip()
        is_semantic = bool(test.get("is_semantic", is_semantic_hidden_test(test)))
        weak = (
            source == "fallback"
            or test_tier == "smoke"
            or bool(test.get("is_placeholder"))
            or _is_weak_test_code(code)
        )

        if weak:
            if "assert True" in code or "assertTrue" in code:
                weak_failures.append("ASSERT_TRUE_TEST")
            elif bool(test.get("is_placeholder")):
                weak_failures.append("PLACEHOLDER_TEST")
            elif test_tier == "smoke" or is_weak_smoke_test(code):
                weak_failures.append("SMOKE_TEST_ONLY")
            elif _is_only_exists_check(code):
                weak_failures.append("ONLY_FUNCTION_EXISTS_CHECK")
            elif _is_only_callability_check(code):
                weak_failures.append("ONLY_CALLABILITY_CHECK")
            elif _is_only_non_empty_output_check(code):
                weak_failures.append("ONLY_NON_EMPTY_OUTPUT_CHECK")
            elif _is_only_no_crash_check(code):
                weak_failures.append("ONLY_NO_CRASH_CHECK")
            elif "assert" not in code:
                weak_failures.append("NO_SEMANTIC_ASSERTION")
            continue

        if not is_semantic:
            semantic_metadata_failures.append("TEST_NOT_SEMANTIC")
        if not semantic_intent and not target_constraint and not expected_failure_mode:
            semantic_metadata_failures.append("TEST_CANNOT_EXPLAIN_TARGET_CONSTRAINT")
        elif not semantic_intent:
            semantic_metadata_failures.append("MISSING_SEMANTIC_INTENT")
        if not expected_failure_mode:
            missing_expected_failure_mode = True

        if not operator_id:
            targeting_failures.append("TEST_NOT_LINKED_TO_OPERATOR")
            weak_linkage_warning = True
        if not axis:
            targeting_failures.append("TEST_NOT_LINKED_TO_AXIS")
            weak_linkage_warning = True
        if not semantic_intent and not target_constraint:
            targeting_failures.append("NO_TEST_FOR_NEW_CONSTRAINT")

        semantic_effective_count += 1
        semantic_test_ids.append(test_id)
        if operator_id:
            linked_operator_ids.append(operator_id)
        if target_constraint:
            target_constraints.append(target_constraint)
        elif semantic_intent:
            target_constraints.append(semantic_intent)

    return {
        "weak_failures": weak_failures,
        "semantic_metadata_failures": semantic_metadata_failures,
        "targeting_failures": targeting_failures,
        "semantic_test_ids": _dedupe(semantic_test_ids),
        "linked_operator_ids": _dedupe(linked_operator_ids),
        "target_constraints": _dedupe(target_constraints),
        "semantic_effective_count": semantic_effective_count,
        "missing_expected_failure_mode": missing_expected_failure_mode,
        "weak_linkage_warning": weak_linkage_warning,
    }


def _semantic_count_sufficient(v_intensity: int, count: int) -> bool:
    if v_intensity <= 0:
        return count >= 1 if count else True
    if v_intensity == 1:
        return count >= 1
    if v_intensity == 2:
        return count >= 2
    return count >= 3


def _scaled_oracle_case_check(env: ExecutableEnvSpec, scaled_oracle_cases: list[dict[str, Any]]) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    difficulty = env.difficulty.global_level if env.difficulty else ""
    semantic_change = bool((env.gold_change_metadata or {}).get("gold_changed"))
    answer_invariant = bool((env.gold_change_metadata or {}).get("answer_invariant"))
    if not semantic_change or answer_invariant:
        return {"failure_reasons": failures, "warnings": warnings}
    if difficulty == "M4":
        if len(scaled_oracle_cases) > 1 and not any("," in str(item.get("axis") or "") for item in scaled_oracle_cases):
            failures.append("M4_COMBINED_ORACLE_CASE_MISSING")
    return {"failure_reasons": failures, "warnings": warnings}


def _is_weak_test_code(code: str) -> bool:
    normalized = code.replace(" ", "")
    return any(
        check(code, normalized)
        for check in [
            lambda original, norm: is_weak_smoke_test(original),
            lambda original, norm: "assert True" in original or "assertTrue" in norm,
            lambda original, norm: _is_only_exists_check(original),
            lambda original, norm: _is_only_callability_check(original),
            lambda original, norm: _is_only_non_empty_output_check(original),
            lambda original, norm: _is_only_no_crash_check(original),
        ]
    )


def _is_only_exists_check(code: str) -> bool:
    normalized = code.replace(" ", "").lower()
    return ("globals()" in normalized or "hasattr(" in normalized) and "assert" in normalized and "==" not in normalized


def _is_only_callability_check(code: str) -> bool:
    normalized = code.replace(" ", "").lower()
    return "callable(" in normalized and "assert" in normalized and "==" not in normalized


def _is_only_non_empty_output_check(code: str) -> bool:
    normalized = code.replace(" ", "").lower()
    return "len(" in normalized and ">0" in normalized and "assert" in normalized and "==" not in normalized


def _is_only_no_crash_check(code: str) -> bool:
    normalized = code.replace(" ", "").lower()
    return "try:" in normalized and "except" in normalized and "assertfalse" in normalized


def _first_nonempty(*values: str) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
