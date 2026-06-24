from __future__ import annotations

from typing import Any

from medenvscale.schemas import ExecutableEnvSpec

from .artifact_admission_gate import run_pipeline_artifact_admission_gate
from .gold_case_execution_gate import run_gold_case_execution_gate
from .oracle_case_quality_gate import run_oracle_case_quality_gate
from .report_schema import PASS, PASS_WITH_WARNING, build_gate_result, collect_rejection_reasons, decide_final_decision, summarize_operator_report


def run_stage05_gates(sample: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    scaled_task = _load_env(sample, "scaled_task")
    if scaled_task.difficulty and scaled_task.difficulty.global_level == "M1":
        return _run_m1_baseline_stage05(scaled_task, config=config)
    operator_report = sample.get("operator_realization_report") or list(scaled_task.operator_realization_report)
    operator_summary = summarize_operator_report(operator_report)

    gate1 = run_oracle_case_quality_gate(sample, config=config)
    gate2 = run_gold_case_execution_gate(sample, config=config)
    gate3 = run_pipeline_artifact_admission_gate(
        {
            **sample,
            "prior_gate_results": {
                "oracle_case_quality_gate": gate1,
                "gold_case_execution_gate": gate2,
            },
        },
        config=config,
    )

    gate_results = {
        "oracle_case_quality_gate": gate1,
        "gold_case_execution_gate": gate2,
        "artifact_integrity_gate": gate3,
    }
    final_decision, stage05_passed = decide_final_decision(operator_summary, list(gate_results.values()))
    return {
        "sample_id": scaled_task.env_id,
        "stage05_passed": stage05_passed,
        "final_decision": final_decision,
        "existing_operator_realization": operator_summary,
        "gate_results": gate_results,
        "clean_eligible": stage05_passed,
        "rejection_reasons": collect_rejection_reasons(operator_summary, list(gate_results.values())),
    }


def _run_m1_baseline_stage05(scaled_task: ExecutableEnvSpec, config: dict | None = None) -> dict[str, Any]:
    validated_warning = "M1_BASELINE_USES_VALIDATED_SEED_CASE"
    operator_summary = {
        "passed": True,
        "severity": PASS_WITH_WARNING,
        "failure_reasons": [],
        "warnings": [validated_warning],
        "operator_ids": [],
    }
    gate1 = run_oracle_case_quality_gate({"scaled_task": scaled_task}, config=config)
    gate2 = run_gold_case_execution_gate({"scaled_task": scaled_task}, config=config)
    artifact_failures = []
    if not scaled_task.validated_oracle_cases:
        artifact_failures.append("EMPTY_VALIDATED_ORACLE_CASES")
    if not scaled_task.scaled_oracle_cases:
        artifact_failures.append("EMPTY_SCALED_ORACLE_CASES")
    if not scaled_task.scaled_gold_case_execution_report:
        artifact_failures.append("SCALED_GOLD_CASE_EXECUTION_REPORT_MISSING")
    if not str(scaled_task.gold_solution or "").strip():
        artifact_failures.append("EMPTY_SCALED_GOLD")
    gate_results = {
        "oracle_case_quality_gate": gate1,
        "gold_case_execution_gate": gate2,
        "artifact_integrity_gate": build_gate_result(
            gate_name="artifact_integrity_gate",
            checks={
                "tool_config_passed": True,
                "result_completeness_passed": not artifact_failures,
                "operator_realization_report_passed": True,
                "clean_rejected_policy_passed": str(gate1.get("severity") or "") not in {"soft_fail", "hard_fail"} and str(gate2.get("severity") or "") not in {"soft_fail", "hard_fail"},
                "downstream_consumability_passed": True,
            },
            failure_reasons=artifact_failures,
            warnings=[validated_warning],
            evidence={
                "matched_operators": [],
                "matched_requirements": list((gate1.get("evidence") or {}).get("covered_requirements", [])),
            },
            hard_fail_codes={"EMPTY_VALIDATED_ORACLE_CASES", "EMPTY_SCALED_ORACLE_CASES", "SCALED_GOLD_CASE_EXECUTION_REPORT_MISSING", "EMPTY_SCALED_GOLD"},
        ),
    }
    final_decision, stage05_passed = decide_final_decision(operator_summary, list(gate_results.values()))
    return {
        "sample_id": scaled_task.env_id,
        "stage05_passed": stage05_passed,
        "final_decision": final_decision,
        "existing_operator_realization": operator_summary,
        "gate_results": gate_results,
        "clean_eligible": stage05_passed,
        "rejection_reasons": collect_rejection_reasons(operator_summary, list(gate_results.values())),
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
