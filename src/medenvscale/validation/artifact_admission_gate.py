from __future__ import annotations

from typing import Any

from medenvscale.scaling.tool_config_validator import agent_tool_registry, validate_tool_config
from medenvscale.schemas import ExecutableEnvSpec, ToolConfig

from .report_schema import build_gate_result, summarize_operator_report

HARD_FAIL_CODES = {
    "FALLBACK_TOOL_CONFIG",
    "TOOL_NOT_IN_POOL",
    "TOOL_SCHEMA_MISSING",
    "TOOL_DESCRIPTION_MISSING",
    "ALLOWED_TOOLS_EMPTY",
    "MISSING_REQUIRED_FIELD",
    "EMPTY_OPERATOR_INSTANCES",
    "EMPTY_OPERATOR_REALIZATION_REPORT",
    "EMPTY_SCALED_ORACLE_CASES",
    "EMPTY_SCALED_GOLD",
    "SCALED_GOLD_SOLUTION_MISSING",
    "EMPTY_VALIDATED_ORACLE_CASES",
    "EMPTY_FINAL_PROMPT",
    "EMPTY_SCALING_PLAN",
    "OPERATOR_REALIZATION_REPORT_MISSING",
    "OPERATOR_REALIZATION_HARD_FAIL",
    "SCHEMA_NOT_LOADABLE",
    "EVALUATOR_CANNOT_USE_GOLD",
    "DOWNSTREAM_DRY_RUN_FAILED",
}


def run_pipeline_artifact_admission_gate(sample: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    seed_env = _load_env(sample, "seed_task")
    scaled_env = _load_env(sample, "scaled_task")
    operator_reports = sample.get("operator_realization_report") or list(scaled_env.operator_realization_report)
    prior_gates = sample.get("prior_gate_results", {})
    operator_summary = summarize_operator_report(operator_reports)
    tool_pool_cfg = (config or {}).get("tool_pool_cfg")
    budgets_cfg = (config or {}).get("budgets_cfg")

    tool_config_check = check_tool_config(scaled_env, tool_pool_cfg, budgets_cfg)
    completeness = check_result_completeness(scaled_env, operator_reports, prior_gates)
    operator_report_check = check_existing_operator_realization_report(operator_summary)
    policy_check = check_clean_rejected_policy(operator_summary, prior_gates)
    downstream = check_downstream_consumability(seed_env, scaled_env)

    checks = {
        "tool_config_passed": not tool_config_check["failure_reasons"],
        "result_completeness_passed": not completeness["failure_reasons"],
        "operator_realization_report_passed": not operator_report_check["failure_reasons"],
        "clean_rejected_policy_passed": not policy_check["failure_reasons"],
        "downstream_consumability_passed": not downstream["failure_reasons"],
    }
    failure_reasons = [
        *tool_config_check["failure_reasons"],
        *completeness["failure_reasons"],
        *operator_report_check["failure_reasons"],
        *policy_check["failure_reasons"],
        *downstream["failure_reasons"],
    ]
    warnings = [
        *tool_config_check["warnings"],
        *completeness["warnings"],
        *operator_report_check["warnings"],
        *policy_check["warnings"],
        *downstream["warnings"],
    ]

    return build_gate_result(
        gate_name="artifact_integrity_gate",
        checks=checks,
        failure_reasons=failure_reasons,
        warnings=warnings,
        evidence={
            "matched_operators": [str(item.get("operator_id") or "") for item in scaled_env.operator_instances or []],
            "matched_requirements": list((prior_gates.get("oracle_case_quality_gate", {}) or {}).get("evidence", {}).get("covered_requirements", [])),
        },
        hard_fail_codes=HARD_FAIL_CODES,
    )


def check_tool_config(scaled_env: ExecutableEnvSpec, tool_pool_cfg: dict | None, budgets_cfg: dict | None) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    tool_config_payload = scaled_env.tool_config or {}
    if not tool_config_payload:
        return {"failure_reasons": [], "warnings": ["TOOL_CONFIG_DISABLED"]}
    planning_source = str(tool_config_payload.get("planning_source") or "")
    validation_trace = [str(item) for item in tool_config_payload.get("validation_trace", []) or []]
    if planning_source == "fallback" or "fallback_tool_config" in validation_trace:
        failures.append("FALLBACK_TOOL_CONFIG")
    allowed_tools = tool_config_payload.get("allowed_tools", []) or []
    if not allowed_tools:
        failures.append("ALLOWED_TOOLS_EMPTY")
        return {"failure_reasons": failures, "warnings": warnings}

    registry = agent_tool_registry(tool_pool_cfg)
    for tool in allowed_tools:
        name = str(tool.get("tool_name") or "")
        if name not in registry:
            failures.append("TOOL_NOT_IN_POOL")
        if not tool.get("description"):
            failures.append("TOOL_DESCRIPTION_MISSING")
        if not tool.get("input_schema"):
            failures.append("TOOL_SCHEMA_MISSING")

    if budgets_cfg is not None:
        try:
            validate_tool_config(
                ToolConfig.model_validate(tool_config_payload),
                budgets_cfg=budgets_cfg,
                tool_pool_cfg=tool_pool_cfg,
                resource_manifest=scaled_env.resource_files,
                scaling_plan=scaled_env.scaling_plan or scaled_env.scaling or {},
                primary_domain=scaled_env.primary_domain,
                secondary_domain_names=[item.domain for item in scaled_env.secondary_domains],
            )
        except Exception as exc:
            failures.append(f"TOOL_BUDGET_MISMATCH: {exc}")
    return {"failure_reasons": failures, "warnings": warnings}


def check_result_completeness(
    scaled_env: ExecutableEnvSpec,
    operator_reports: list[dict[str, Any]],
    prior_gates: dict[str, Any],
) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    required_values = {
        "sample_id": scaled_env.env_id,
        "final_user_prompt": scaled_env.user_prompt,
        "operator_instances": scaled_env.operator_instances,
        "operator_realization_report": operator_reports,
        "scaled_oracle_cases": scaled_env.scaled_oracle_cases,
        "validated_oracle_cases": scaled_env.validated_oracle_cases,
        "oracle_case_validation_report": scaled_env.oracle_case_validation_report,
        "scaled_gold_case_execution_report": scaled_env.scaled_gold_case_execution_report,
        "quality_flags": scaled_env.quality_flags,
        "gate_results": prior_gates,
        "scaled_gold": scaled_env.gold_solution,
    }
    for name, value in required_values.items():
        if value is None or value == "":
            failures.append(f"MISSING_REQUIRED_FIELD: {name}")
    if not scaled_env.operator_instances:
        failures.append("EMPTY_OPERATOR_INSTANCES")
    if not operator_reports:
        failures.append("EMPTY_OPERATOR_REALIZATION_REPORT")
    answer_invariant = bool((scaled_env.gold_change_metadata or {}).get("answer_invariant", (scaled_env.gold_state or {}).get("answer_invariant", False)))
    if (scaled_env.difficulty and scaled_env.difficulty.global_level != "M1") and not answer_invariant and not scaled_env.scaled_oracle_cases:
        failures.append("EMPTY_SCALED_ORACLE_CASES")
    if (scaled_env.difficulty and scaled_env.difficulty.global_level != "M1") and not scaled_env.validated_oracle_cases:
        failures.append("EMPTY_VALIDATED_ORACLE_CASES")
    if not scaled_env.gold_solution:
        failures.append("EMPTY_SCALED_GOLD")
    if (scaled_env.difficulty and scaled_env.difficulty.global_level != "M1") and not scaled_env.scaled_gold_solution:
        failures.append("SCALED_GOLD_SOLUTION_MISSING")
    if not scaled_env.user_prompt:
        failures.append("EMPTY_FINAL_PROMPT")
    if not scaled_env.scaling_plan and not scaled_env.scaling:
        failures.append("EMPTY_SCALING_PLAN")
    return {"failure_reasons": failures, "warnings": warnings}


def check_existing_operator_realization_report(operator_summary: dict[str, Any]) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    severity = str(operator_summary.get("severity") or "")
    if not operator_summary.get("operator_ids"):
        failures.append("OPERATOR_REALIZATION_REPORT_MISSING")
    if severity == "hard_fail":
        failures.append("OPERATOR_REALIZATION_HARD_FAIL")
    elif severity == "soft_fail":
        failures.append("OPERATOR_REALIZATION_SOFT_FAIL")
    elif severity == "pass_with_warning":
        warnings.append("OPERATOR_REALIZATION_PASS_WITH_WARNING")
    return {"failure_reasons": failures, "warnings": warnings}


def check_clean_rejected_policy(operator_summary: dict[str, Any], prior_gates: dict[str, Any]) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if str(operator_summary.get("severity") or "") in {"soft_fail", "hard_fail"}:
        failures.append(f"OPERATOR_REALIZATION_{str(operator_summary.get('severity') or '').upper()}")
    for gate_name in ["oracle_case_quality_gate", "gold_case_execution_gate"]:
        gate = prior_gates.get(gate_name) or {}
        severity = str(gate.get("severity") or "")
        if severity in {"soft_fail", "hard_fail"}:
            failures.append(f"{gate_name.upper()}_{severity.upper()}")
    return {"failure_reasons": failures, "warnings": warnings}


def check_downstream_consumability(seed_env: ExecutableEnvSpec, scaled_env: ExecutableEnvSpec) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    try:
        ExecutableEnvSpec.model_validate(scaled_env.model_dump())
    except Exception:
        failures.append("SCHEMA_NOT_LOADABLE")
    if not scaled_env.user_prompt:
        failures.append("DOWNSTREAM_DRY_RUN_FAILED")
    if not scaled_env.gold_solution or not seed_env.gold_solution:
        failures.append("EVALUATOR_CANNOT_USE_GOLD")
    return {"failure_reasons": failures, "warnings": warnings}


def _load_env(sample: dict[str, Any], key: str) -> ExecutableEnvSpec:
    value = sample.get(key)
    if isinstance(value, ExecutableEnvSpec):
        return value
    if isinstance(value, dict):
        return ExecutableEnvSpec.model_validate(value)
    if key == "scaled_task":
        return ExecutableEnvSpec.model_validate(sample)
    raise ValueError(f"Missing environment payload: {key}")
