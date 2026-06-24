from __future__ import annotations

from typing import Any

from medenvscale.schemas import ExecutableEnvSpec

BLOCKING_FLAGS = {
    "Hidden test must be a dict",
    "exception_test must be a dict",
    "compile_error",
    "missing structured state update",
    "D axis must add a data/input patch",
    "C axis must add executable constraints",
    "A axis must add a robustness challenge",
    "numeric_tolerance_tests missing required fields",
    "fallback_tool_config",
    "verifier_build_failed",
    "missing semantic state update",
    "hidden_test_compile_error",
    "hidden_test_execution_error",
    "hidden_test_missing_code",
    "hidden_test_context_compile_error",
    "hidden_test_context_execution_error",
    "duplicate_hidden_test_id",
    "fallback_hidden_test_present",
    "smoke_hidden_test_present",
    "semantic_equivalent_to_base_m1",
    "operator_missing_semantic_delta",
    "operator_realization_soft_fail",
    "operator_realization_hard_fail",
    "stage05_gate_soft_fail",
    "stage05_gate_hard_fail",
    "FALLBACK_TOOL_CONFIG",
    "OPERATOR_REALIZATION_HARD_FAIL",
    "OPERATOR_REALIZATION_SOFT_FAIL",
    "SCALED_GOLD_MISSING",
    "SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION",
    "NO_VALID_SEMANTIC_HIDDEN_TEST",
    "NO_VALIDATED_ORACLE_CASES",
    "VALIDATED_ORACLE_CASES_TOO_FEW",
    "NO_VALID_ORACLE_CASE_FOR_OPERATOR",
    "SCALED_ORACLE_CASES_TOO_FEW",
    "M4_COMBINED_ORACLE_CASE_MISSING",
    "SEED_CASE_ADMISSION_FAILED",
}


def is_semantic_hidden_test(test: dict[str, Any]) -> bool:
    return bool(
        test.get("counts_as_hidden_test", True)
        and test.get("eligible_for_clean_export", True)
        and test.get("test_tier", "semantic") != "smoke"
    )


def infer_verifier_mechanisms(env: ExecutableEnvSpec) -> list[str]:
    mechanisms: list[str] = []
    verifier_delta = env.verifier_delta or {}
    verifier_spec = env.verifier_spec or {}
    if verifier_delta.get("new_hidden_tests") or verifier_spec.get("hidden_tests"):
        mechanisms.append("hidden_tests")
    if verifier_delta.get("numeric_tolerance_tests"):
        mechanisms.append("numeric_tolerance")
    if verifier_delta.get("file_output_tests"):
        mechanisms.append("file_output")
    if verifier_delta.get("object_state_tests"):
        mechanisms.append("object_state")
    if verifier_delta.get("dataframe_equal_tests"):
        mechanisms.append("dataframe_equal")
    if verifier_delta.get("array_close_tests"):
        mechanisms.append("array_close")
    if len((verifier_spec or {}).get("checks", [])) >= 3:
        mechanisms.append("multi_check")
    return mechanisms


def validate_v_axis(env: ExecutableEnvSpec) -> list[str]:
    v = int((env.scaling or env.scaling_plan or {}).get("axis_intensity", {}).get("V", 0))
    if v == 0:
        return []
    validated_cases = [case for case in (env.validated_oracle_cases or []) if isinstance(case, dict)]
    v_operator_ids = {
        str(op.get("operator_id") or "").strip()
        for op in (env.operator_instances or [])
        if isinstance(op, dict) and str(op.get("axis") or "").strip() == "V"
    }
    linked_cases = [
        case
        for case in validated_cases
        if not v_operator_ids
        or any(operator_id and operator_id in str(case.get("targets_operator_id") or "") for operator_id in v_operator_ids)
    ]
    required_case_count = 1 if v == 1 else 2 if v == 2 else 3
    if len(linked_cases) < required_case_count:
        return [f"V={v} requires at least {required_case_count} validated oracle cases"]
    return []


def validate_a_axis(env: ExecutableEnvSpec) -> list[str]:
    a = int((env.scaling or env.scaling_plan or {}).get("axis_intensity", {}).get("A", 0))
    if a == 0:
        return []
    visible_state = env.visible_state or {}
    data_state = env.data_state or {}
    gold_state = env.gold_state or {}
    has_challenge = bool(
        visible_state.get("robustness_challenges")
        or visible_state.get("robustness_trap")
        or visible_state.get("must_not_assume")
        or data_state.get("distractors")
        or gold_state.get("must_not_follow_shortcut")
        or (env.task_state or {}).get("shortcut_traps")
    )
    verifier_items = list(env.hidden_tests) + list((env.verifier_spec or {}).get("checks", []))
    has_verifier_rejection = any(
        token in str(item).lower() for item in verifier_items for token in ["shortcut", "duplicate", "conflict", "hardcod", "random"]
    )
    if not has_challenge or not has_verifier_rejection:
        return ["A axis requires real robustness challenge and verifier rejection"]
    return []


def validate_d_axis(env: ExecutableEnvSpec) -> list[str]:
    d = int((env.scaling or env.scaling_plan or {}).get("axis_intensity", {}).get("D", 0))
    if d == 0:
        return []
    data_state = env.data_state or {}
    has_data_patch = bool(
        data_state.get("resource_variants")
        or data_state.get("additional_inputs")
        or data_state.get("input_format_variants")
    )
    if not has_data_patch:
        return ["D axis requires real input/data complexity patches"]
    return []


def validate_c_axis(env: ExecutableEnvSpec) -> list[str]:
    c = int((env.scaling or env.scaling_plan or {}).get("axis_intensity", {}).get("C", 0))
    if c == 0:
        return []
    task_state = env.task_state or {}
    visible_state = env.visible_state or {}
    verifier_state = env.verifier_state or {}
    has_constraint = bool(
        task_state.get("extra_constraints")
        or visible_state.get("output_constraints")
        or visible_state.get("format_constraints")
        or verifier_state.get("constraint_checks")
    )
    if not has_constraint:
        return ["C axis requires real executable constraints"]
    return []


def validate_hidden_tests(env: ExecutableEnvSpec) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    for index, test in enumerate(env.hidden_tests, start=1):
        if not isinstance(test, dict):
            issues.append("Hidden test must be a dict")
            continue
        test_id = str(test.get("test_id") or test.get("name") or f"hidden_test_{index}")
        if test_id in seen_ids:
            issues.append(f"duplicate_hidden_test_id: {test_id}")
        else:
            seen_ids.add(test_id)
        if str(test.get("source") or "") == "fallback":
            issues.append(f"fallback_hidden_test_present: {test_id}")
        if str(test.get("test_tier") or "") == "smoke":
            issues.append(f"smoke_hidden_test_present: {test_id}")
    return issues


def validate_semantic_delta(env: ExecutableEnvSpec, base_env: ExecutableEnvSpec | None) -> list[str]:
    difficulty = env.difficulty
    if not difficulty or difficulty.global_level == "M1" or base_env is None:
        return []
    current = _semantic_signature(env)
    base = _semantic_signature(base_env)
    if current == base:
        return ["semantic_equivalent_to_base_m1"]
    return []


def validate_operator_semantics(env: ExecutableEnvSpec) -> list[str]:
    difficulty = env.difficulty
    if not difficulty or difficulty.global_level == "M1":
        return []
    issues: list[str] = []
    for op in env.operator_instances or []:
        state_updates = op.get("state_updates") or {}
        if not any(
            [
                bool(state_updates.get("visible_state_patch")),
                bool(state_updates.get("gold_state_patch")),
                bool(state_updates.get("verifier_state_patch")),
                bool(state_updates.get("test_state_patch")),
                bool(state_updates.get("data_state_patch")),
            ]
        ):
            issues.append(f"operator_missing_semantic_delta: {op.get('operator_id', 'unknown_operator')}")
    return issues


def collect_blocking_quality_flags(env: ExecutableEnvSpec) -> list[str]:
    stage05_report = env.stage05_quality_report or {}
    if stage05_report:
        blocking: list[str] = []
        if not stage05_report.get("stage05_passed", False):
            blocking.extend(str(item) for item in stage05_report.get("rejection_reasons", []) or [])
        for gate_name, gate_result in (stage05_report.get("gate_results") or {}).items():
            severity = str(gate_result.get("severity") or "")
            if severity == "soft_fail":
                blocking.append(f"stage05_gate_soft_fail:{gate_name}")
            elif severity == "hard_fail":
                blocking.append(f"stage05_gate_hard_fail:{gate_name}")
        return _dedupe(blocking)

    blocking = []
    for flag in env.quality_flags:
        if any(pattern in flag for pattern in BLOCKING_FLAGS):
            blocking.append(flag)
    blocking.extend(validate_hidden_tests(env))
    blocking.extend(validate_operator_semantics(env))
    return _dedupe(blocking)


def split_clean_and_rejected(envs: list[ExecutableEnvSpec]) -> tuple[list[ExecutableEnvSpec], list[ExecutableEnvSpec]]:
    clean: list[ExecutableEnvSpec] = []
    rejected: list[ExecutableEnvSpec] = []
    base_by_task = {
        env.original_task_id: env
        for env in envs
        if env.difficulty and env.difficulty.global_level == "M1"
    }
    for env in envs:
        blocking = collect_blocking_quality_flags(env)
        if not env.stage05_quality_report:
            blocking.extend(validate_semantic_delta(env, base_by_task.get(env.original_task_id)))
        blocking = _dedupe(blocking)
        updated = env.model_copy(update={"blocking_quality_flags": blocking})
        explicit_decision = str((env.stage05_quality_report or {}).get("final_decision") or "")
        stage05_passed = bool((env.stage05_quality_report or {}).get("stage05_passed", env.stage05_passed))
        if explicit_decision == "clean" and stage05_passed and not blocking:
            clean.append(updated.model_copy(update={"export_status": "clean"}))
        elif blocking or explicit_decision == "rejected" or (explicit_decision and not stage05_passed):
            rejected.append(updated.model_copy(update={"export_status": "rejected"}))
        else:
            clean.append(updated.model_copy(update={"export_status": "clean"}))
    return clean, rejected


def build_quality_report(envs: list[ExecutableEnvSpec]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for env in envs:
        report.append(
            {
                "env_id": env.env_id,
                "original_task_id": env.original_task_id,
                "keep": env.export_status == "clean",
                "export_status": env.export_status,
                "quality_flags": list(env.quality_flags),
                "blocking_quality_flags": list(env.blocking_quality_flags),
                "stage05_quality_report": env.stage05_quality_report,
            }
        )
    return report


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _semantic_signature(env: ExecutableEnvSpec) -> dict[str, Any]:
    visible_state = env.visible_state or {}
    gold_state = env.gold_state or {}
    verifier_state = env.verifier_state or {}
    return {
        "problem": env.problem,
        "context": env.context,
        "gold_solution": env.gold_solution,
        "visible_state": {
            key: visible_state.get(key)
            for key in [
                "constraint_hints",
                "execution_requirements",
                "resource_complexity_notes",
                "input_description",
                "robustness_trap",
                "robustness_challenges",
                "output_constraints",
            ]
            if key in visible_state
        },
        "gold_state": gold_state,
        "verifier_state": verifier_state,
        "test_state": env.test_state or {},
        "semantic_hidden_tests": sorted(
            str(test.get("test_id") or test.get("name"))
            for test in env.hidden_tests
            if isinstance(test, dict) and is_semantic_hidden_test(test)
        ),
    }
