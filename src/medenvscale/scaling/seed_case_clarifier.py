from __future__ import annotations

import json
from typing import Any

from medenvscale.scaling.runtime_value_sanitizer import stabilize_expected_output_signature
from medenvscale.schemas import ExecutableEnvSpec


def add_seed_behavior_requirements_to_env(env: ExecutableEnvSpec) -> ExecutableEnvSpec:
    requirements = build_seed_behavior_requirements(env)
    if not requirements:
        return env
    visible_state = dict(env.visible_state or {})
    existing = _as_list(visible_state.get("execution_requirements"))
    visible_state["execution_requirements"] = _dedupe_strings(existing + requirements)
    visible_state["seed_behavior_requirements"] = requirements
    metadata = dict(env.metadata or {})
    metadata["seed_behavior_requirements"] = requirements
    output_requirements = _dedupe_strings([*(env.output_requirements or []), *requirements])
    return env.model_copy(
        update={
            "visible_state": visible_state,
            "metadata": metadata,
            "output_requirements": output_requirements,
        }
    )


def build_seed_behavior_requirements(env: ExecutableEnvSpec) -> list[str]:
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    expected = seed_case.get("expected_output_signature")
    if not isinstance(expected, dict) or not expected:
        expected = env.seed_ground_truth_output_signature if isinstance(env.seed_ground_truth_output_signature, dict) else {}
    if isinstance(expected, dict):
        expected = stabilize_expected_output_signature(dict(expected))
    call_code = str(seed_case.get("call_code") or "").strip()
    if not call_code or not isinstance(expected, dict) or not expected:
        return []

    target_name = _target_name_from_signature(env.signature)
    requirements: list[str] = [
        (
            f"Preserve the original seed behavior for {target_name}; newly added requirements are additive unless explicitly stated otherwise."
            if target_name
            else "Preserve the original seed task behavior; newly added requirements are additive unless explicitly stated otherwise."
        ),
        f"Original seed executable call: {_truncate_inline(call_code, 220)}",
    ]
    if "return_type" in expected:
        requirements.append(f"On the original seed call, return type must be {expected.get('return_type')!r}.")
    if "return_value" in expected:
        requirements.append(f"On the original seed call, return value must be {_format_value(expected.get('return_value'), 260)}.")
    if "return_value_contains" in expected:
        requirements.append(
            f"On the original seed call, return value must contain {_format_value(expected.get('return_value_contains'), 220)}."
        )
    if "return_keys" in expected:
        requirements.append(f"On the original seed call, returned mapping keys must include {_format_value(expected.get('return_keys'), 220)}.")
    if "stdout_contains" in expected:
        requirements.append(f"On the original seed call, stdout must contain {_format_value(expected.get('stdout_contains'), 220)}.")
    if "stdout_regex" in expected:
        requirements.append(f"On the original seed call, stdout must match {_format_value(expected.get('stdout_regex'), 220)}.")
    if "stderr_contains" in expected:
        requirements.append(f"On the original seed call, stderr must contain {_format_value(expected.get('stderr_contains'), 220)}.")
    if "file_artifacts" in expected:
        paths = [
            str(item.get("path") or "").strip()
            for item in expected.get("file_artifacts") or []
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        ]
        if paths:
            requirements.append(f"On the original seed call, create or update file artifact paths {_format_value(paths, 220)}.")
    return _dedupe_strings(requirements)


def build_seed_regression_oracle_case(env: ExecutableEnvSpec) -> dict[str, Any] | None:
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    expected = seed_case.get("expected_output_signature")
    if not isinstance(expected, dict) or not expected:
        expected = env.seed_ground_truth_output_signature if isinstance(env.seed_ground_truth_output_signature, dict) else {}
    if isinstance(expected, dict):
        expected = stabilize_expected_output_signature(dict(expected))
    call_code = str(seed_case.get("call_code") or "").strip()
    if not call_code or not isinstance(expected, dict) or not expected:
        return None

    base_case_id = str(seed_case.get("case_id") or "seed_case_main").strip() or "seed_case_main"
    requirements = build_seed_behavior_requirements(env)
    if not requirements:
        target_name = _target_name_from_signature(env.signature)
        requirements = [
            (
                f"Preserve original seed behavior for {target_name}."
                if target_name
                else "Preserve original seed task behavior."
            )
        ]
    requirement = requirements[0]
    return {
        "case_id": f"regression_{base_case_id}",
        "base_case_id": base_case_id,
        "description": str(seed_case.get("description") or "Seed regression oracle case derived from the original seed execution case."),
        "case_kind": "seed_regression",
        "targets_operator_id": "seed_regression",
        "axis": "REGRESSION",
        "semantic_intent": requirement,
        "target_constraint": requirement,
        "expected_failure_mode": "candidate satisfies scaled additions but breaks the original seed behavior",
        "setup_code": str(seed_case.get("setup_code") or ""),
        "call_code": call_code,
        "assertion_code": str(seed_case.get("assertion_code") or ""),
        "covered_requirements": requirements,
        "covers_requirements": requirements,
        "expected_output_signature": dict(expected),
    }


def merge_seed_regression_case(
    env: ExecutableEnvSpec,
    cases: list[dict[str, Any]],
    *,
    level: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_level = level or str((env.difficulty.global_level if env.difficulty else "") or "")
    if resolved_level not in {"M2", "M3", "M4"}:
        return list(cases), []
    regression_case = build_seed_regression_oracle_case(env)
    if not regression_case:
        return list(cases), [
            {
                "case_id": "regression_seed_case_main",
                "failure_code": "SEED_REGRESSION_CASE_MISSING",
                "failure_message": "seed_execution_case could not be wrapped as a seed_regression oracle case.",
            }
        ]
    regression_id = str(regression_case.get("case_id") or "")
    merged = [regression_case]
    for case in cases:
        if not isinstance(case, dict):
            continue
        if str(case.get("case_kind") or "") == "seed_regression":
            continue
        if regression_id and str(case.get("case_id") or "") == regression_id:
            continue
        merged.append(case)
    return merged, []


def seed_regression_validation_report_row(env: ExecutableEnvSpec, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_id": env.env_id,
        "level": str((env.difficulty.global_level if env.difficulty else "") or ""),
        "case_id": str(case.get("case_id") or "regression_seed_case_main"),
        "valid": True,
        "failure_reasons": [],
        "targets_operator_id": str(case.get("targets_operator_id") or "seed_regression"),
        "axis": str(case.get("axis") or "REGRESSION"),
        "covered_requirements": list(case.get("covered_requirements") or case.get("covers_requirements") or []),
        "matched_requirements": list(case.get("covered_requirements") or case.get("covers_requirements") or []),
    }


def _target_name_from_signature(signature: str | None) -> str:
    text = str(signature or "").strip()
    if text.startswith("def "):
        return text[4:].split("(", 1)[0].strip()
    if text.startswith("class "):
        return text[6:].split("(", 1)[0].split(":", 1)[0].strip()
    return ""


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _format_value(value: Any, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = repr(value)
    return _truncate_inline(text, limit)


def _truncate_inline(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
