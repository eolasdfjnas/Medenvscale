from __future__ import annotations

from typing import Any

from medenvscale.schemas import ExecutableEnvSpec

from .report_schema import build_gate_result

HARD_FAIL_CODES = {
    "SCALED_EXECUTABLE_GOLD_CODE_MISSING",
    "SCALED_GOLD_CASE_EXECUTION_FAILED",
    "SCALED_GOLD_COMPILE_FAILED",
}


def run_gold_case_execution_gate(sample: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    scaled_env = _load_env(sample, "scaled_task")
    execution_report = list(scaled_env.scaled_gold_case_execution_report or [])
    checks = {
        "scaled_gold_exists": bool(str(scaled_env.scaled_executable_gold_code or "").strip()),
        "case_report_exists": bool(execution_report),
        "all_cases_passed": True,
    }
    failure_reasons: list[str] = []
    warnings: list[str] = []
    if not checks["scaled_gold_exists"]:
        failure_reasons.append("SCALED_EXECUTABLE_GOLD_CODE_MISSING")
    if execution_report:
        failed_rows = [row for row in execution_report if not bool(row.get("passed"))]
        if failed_rows:
            checks["all_cases_passed"] = False
            for row in failed_rows:
                case_id = str(row.get("case_id") or "unknown_case")
                reasons = ",".join(str(item) for item in row.get("failure_reasons", []) or [])
                failure_reasons.append(f"SCALED_GOLD_CASE_EXECUTION_FAILED:{case_id}:{reasons}")
    else:
        failure_reasons.append("SCALED_GOLD_CASE_EXECUTION_FAILED:NO_CASE_REPORT")

    return build_gate_result(
        gate_name="gold_case_execution_gate",
        checks=checks,
        failure_reasons=failure_reasons,
        warnings=warnings,
        evidence={
            "case_ids": [str(row.get("case_id") or "") for row in execution_report],
            "failed_case_ids": [str(row.get("case_id") or "") for row in execution_report if not bool(row.get("passed"))],
        },
        hard_fail_codes=HARD_FAIL_CODES,
    )


def _load_env(sample: dict[str, Any], key: str) -> ExecutableEnvSpec:
    value = sample.get(key)
    if isinstance(value, ExecutableEnvSpec):
        return value
    if isinstance(value, dict):
        return ExecutableEnvSpec.model_validate(value)
    if key == "scaled_task":
        return ExecutableEnvSpec.model_validate(sample)
    raise ValueError(f"Missing environment payload: {key}")
