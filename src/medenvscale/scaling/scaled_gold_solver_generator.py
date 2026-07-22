from __future__ import annotations

import json
import re
import ast
from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.case_execution import run_scaled_gold_on_validated_oracle_cases
from medenvscale.scaling.hidden_test_runner import insert_solution, looks_like_complete_program, run_hidden_test_execution_check
from medenvscale.scaling.oracle_case_validator import _is_generic_requirement_text, _specific_tokens, validate_scaled_oracle_cases
from medenvscale.scaling.output_constraints import (
    check_output_constraints,
    output_constraints_from_scaled_oracle_cases,
)
from medenvscale.scaling.output_signature import execute_candidate_solution
from medenvscale.scaling.path_safety import analyze_relative_path, extract_code_path_references
from medenvscale.scaling.runtime_value_sanitizer import (
    stabilize_expected_output_signature,
)
from medenvscale.scaling.requirement_registry import infer_covered_requirement_ids, requirements_match
from medenvscale.scaling.seed_case_clarifier import (
    build_seed_behavior_requirements,
    merge_seed_regression_case,
    seed_regression_validation_report_row,
)
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.validation.oracle_case_quality_gate import run_oracle_case_quality_gate


def detect_semantic_change(operator_instances: list[dict[str, Any]]) -> dict[str, Any]:
    changed_axes: list[str] = []
    semantic_operator_ids: list[str] = []
    operator_ids: list[str] = []
    for operator in operator_instances:
        axis = str(operator.get("axis") or "")
        operator_id = str(operator.get("operator_id") or "")
        if operator_id:
            operator_ids.append(operator_id)
        semantic_change = bool(operator.get("semantic_change"))
        state_updates = operator.get("state_updates") or {}
        semantic_patch = any(
            bool(state_updates.get(field))
            for field in [
                "visible_state_patch",
                "task_state_patch",
                "gold_state_patch",
                "data_state_patch",
                "test_state_patch",
                "verifier_state_patch",
            ]
        )
        if axis == "V" and not semantic_change and not semantic_patch:
            continue
        if axis == "V" and semantic_change and semantic_patch:
            changed_axes.append(axis)
            semantic_operator_ids.append(str(operator.get("operator_id") or ""))
            continue
        if axis in {"D", "C", "A"} and (semantic_change or semantic_patch):
            changed_axes.append(axis)
            semantic_operator_ids.append(str(operator.get("operator_id") or ""))
    return {
        "semantic_change": bool(changed_axes),
        "changed_axes": changed_axes,
        "semantic_operator_ids": semantic_operator_ids,
        "operator_ids": operator_ids,
        "v_only": bool(operator_instances) and set(changed_axes).issubset({"V"}) and changed_axes != [],
    }


def _oracle_case_operator_ids(operator_instances: list[dict[str, Any]], semantic_info: dict[str, Any]) -> list[str]:
    semantic_ids = [str(item).strip() for item in semantic_info.get("semantic_operator_ids", []) if str(item).strip()]
    if semantic_ids:
        return semantic_ids
    return [str(operator.get("operator_id") or "").strip() for operator in operator_instances if str(operator.get("operator_id") or "").strip()]


def _seed_executable_code(env: ExecutableEnvSpec) -> str:
    return str(env.code or env.scaled_executable_gold_code or env.seed_gold_solution or env.gold_solution or "")


def check_seed_case_admission(env: ExecutableEnvSpec) -> dict[str, Any]:
    failure_reasons: list[str] = []
    audit = env.seed_case_audit or {}
    seed_case = env.seed_execution_case or {}
    ground_truth = env.seed_ground_truth_output_signature or {}
    if str(audit.get("status") or "").strip() != "pass":
        failure_reasons.append("SEED_CASE_AUDIT_NOT_PASS")
    if not isinstance(seed_case, dict) or not seed_case:
        failure_reasons.append("SEED_EXECUTION_CASE_MISSING")
    if not isinstance(ground_truth, dict) or not ground_truth:
        failure_reasons.append("SEED_GROUND_TRUTH_OUTPUT_SIGNATURE_MISSING")
    return {
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "seed_case_id": str(seed_case.get("case_id") or "seed_case_main"),
        "seed_case_status": str(audit.get("status") or ""),
    }


def build_scaled_case_plan(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic_info = detect_semantic_change(operator_instances)
    oracle_operator_ids = _oracle_case_operator_ids(operator_instances, semantic_info)
    admission = check_seed_case_admission(env)
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    case_design_targets = _collect_case_design_targets(operator_instances)
    covered_requirements = _dedupe_strings(
        [str(item).strip() for item in (env.output_requirements or []) if str(item).strip()]
    )
    main_case_blueprint = {
        "case_id": "scaled_seed_case_main",
        "base_case_id": str(seed_case.get("case_id") or "seed_case_main"),
        "case_kind": "scaled_seed_case_main",
        "setup_code": str(seed_case.get("setup_code") or "").strip(),
        "call_code": str(seed_case.get("call_code") or "").strip(),
        "expected_output_signature": _seed_expected_output_signature(env),
        "targets_operator_id": ",".join(oracle_operator_ids),
        "covered_requirements": covered_requirements,
    }
    operator_actions: list[dict[str, Any]] = []
    for target in case_design_targets:
        operator_actions.append(
            {
                "operator_id": str(target.get("operator_id") or ""),
                "axis": str(target.get("axis") or ""),
                "transformation_goal": str(target.get("transformation_goal") or ""),
                "covered_requirements": list(target.get("output_requirements") or target.get("visible_requirements") or target.get("semantic_targets") or []),
                "case_strategy": "update_main_seed_case",
            }
        )
    return {
        "plan_id": f"{env.env_id}_scaled_case_plan",
        "strategy": "seed_case_first",
        "status": "ready" if admission["passed"] else "blocked",
        "seed_case_admission": admission,
        "seed_case_id": str(seed_case.get("case_id") or "seed_case_main"),
        "seed_ground_truth_strategy": "inherit_seed_ground_truth_output_signature",
        "expected_output_strategy": "operator_may_rewrite",
        "required_validated_case_count": _scaled_oracle_case_target_count(env, semantic_info),
        "recommended_additional_case_count": max(0, len(oracle_operator_ids) - 1),
        "main_case_blueprint": main_case_blueprint,
        "operator_actions": operator_actions,
        "semantic_test_spec_ids": [str(spec.get("spec_id") or "") for spec in semantic_test_specs if isinstance(spec, dict)],
        "notes": [
            "Preserve the original seed execution path when possible.",
            "Rewrite expected_output_signature to reflect scaled constraints.",
            "Add extra cases only when the main scaled seed case does not cover all operator requirements.",
        ],
    }


def _scaled_oracle_case_target_count(env: ExecutableEnvSpec, semantic_info: dict[str, Any]) -> int:
    level = str((env.difficulty.global_level if env.difficulty else "") or "")
    if level == "M1":
        return 0
    if semantic_info.get("operator_ids"):
        return 1
    if not semantic_info.get("semantic_change"):
        return 0
    return 1


def _normalize_scaled_oracle_cases(
    cases: list[dict[str, Any]] | None,
    *,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases or [], start=1):
        if not isinstance(case, dict):
            failures.append(
                {
                    "case_id": f"scaled_oracle_case_{index}",
                    "failure_code": "SCALED_ORACLE_CASE_INVALID",
                    "failure_message": "Scaled oracle case must be a dict.",
                }
            )
            continue
        case_id = str(case.get("case_id") or case.get("example_id") or case.get("test_id") or f"scaled_oracle_case_{index}")
        expected_output_signature = (
            case.get("expected_output_signature")
            if isinstance(case.get("expected_output_signature"), dict)
            else {}
        )
        row = {
            "case_id": case_id,
            "description": str(case.get("description") or "").strip(),
            "case_kind": str(case.get("case_kind") or "coverage_extension").strip(),
            "targets_operator_id": str(case.get("targets_operator_id") or ""),
            "axis": str(case.get("axis") or ""),
            "semantic_intent": str(case.get("semantic_intent") or case.get("description") or "").strip(),
            "target_constraint": str(case.get("target_constraint") or "").strip(),
            "expected_failure_mode": str(case.get("expected_failure_mode") or "").strip(),
            "setup_code": str(case.get("setup_code") or "").strip(),
            "call_code": str(case.get("call_code") or "").strip(),
            "assertion_code": str(case.get("assertion_code") or case.get("test_code") or "").strip(),
            "expected_output_signature": expected_output_signature,
            "covered_requirements": [
                str(item).strip()
                for item in ((case.get("covered_requirements") or case.get("covers_requirements")) or [])
                if str(item).strip()
            ],
            "covered_requirement_ids": [
                str(item).strip()
                for item in (case.get("covered_requirement_ids") or [])
                if str(item).strip()
            ],
        }
        if not row["call_code"]:
            failures.append(
                {
                    "case_id": case_id,
                    "failure_code": "SCALED_ORACLE_CASE_INVALID",
                    "failure_message": "Scaled oracle case must provide call_code.",
                }
            )
            continue
        if not row["expected_output_signature"]:
            failures.append(
                {
                    "case_id": case_id,
                    "failure_code": "SCALED_ORACLE_CASE_INVALID",
                    "failure_message": "Scaled oracle case must provide expected_output_signature.",
                }
            )
            continue
        row["covers_requirements"] = list(row["covered_requirements"])
        normalized.append(row)
    if target_count and len(normalized) < target_count:
        failures.append(
            {
                "case_id": "scaled_oracle_cases",
                "failure_code": "SCALED_ORACLE_CASES_TOO_FEW",
                "failure_message": f"Expected at least {target_count} scaled oracle cases, got {len(normalized)}.",
            }
        )
    return normalized, failures


def _scaled_oracle_coverage_summary(scaled_oracle_cases: list[dict[str, Any]], semantic_operator_ids: list[str]) -> dict[str, Any]:
    covered_operator_ids = {
        str(case.get("targets_operator_id") or "")
        for case in scaled_oracle_cases
        if isinstance(case, dict) and str(case.get("targets_operator_id") or "")
    }
    combined_case_ids = [
        str(case.get("case_id") or "")
        for case in scaled_oracle_cases
        if isinstance(case, dict) and "," in str(case.get("axis") or "")
    ]
    return {
        "scaled_oracle_case_count": len(scaled_oracle_cases),
        "covered_operator_ids": sorted(covered_operator_ids),
        "missing_operator_ids": [operator_id for operator_id in semantic_operator_ids if operator_id not in covered_operator_ids],
        "combined_case_ids": combined_case_ids,
    }


def _resolve_oracle_case_target_count(
    env: ExecutableEnvSpec,
    semantic_info: dict[str, Any],
    config: dict[str, Any] | None,
) -> int:
    default_count = _scaled_oracle_case_target_count(env, semantic_info)
    level = str((env.difficulty.global_level if env.difficulty else "") or "")
    stage05_cfg = (config or {}).get("stage05_cfg") or {}
    configured_counts = stage05_cfg.get("min_validated_oracle_cases", {}) if isinstance(stage05_cfg, dict) else {}
    if not isinstance(configured_counts, dict) or level not in configured_counts:
        return default_count
    try:
        configured_count = int(configured_counts[level])
    except (TypeError, ValueError):
        return default_count
    return max(default_count, configured_count)


def _rule_repair_scaled_oracle_cases(
    env: ExecutableEnvSpec,
    cases: list[dict[str, Any]] | None,
    operator_instances: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired_cases: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    for index, case in enumerate(cases or [], start=1):
        if not isinstance(case, dict):
            repaired_cases.append(case)
            report_rows.append(
                {
                    "case_id": f"scaled_oracle_case_{index}",
                    "changed": False,
                    "actions": [],
                    "warnings": ["CASE_NOT_DICT"],
                }
            )
            continue
        original_case = json.loads(json.dumps(case, ensure_ascii=False))
        repaired = dict(case)
        actions: list[str] = []
        warnings: list[str] = []
        case_id = str(repaired.get("case_id") or f"scaled_oracle_case_{index}")

        expected = repaired.get("expected_output_signature")
        if not isinstance(expected, dict):
            expected = {}
            repaired["expected_output_signature"] = expected
            actions.append("NORMALIZED_EXPECTED_OUTPUT_SIGNATURE_DICT")
        else:
            expected = dict(expected)
            repaired["expected_output_signature"] = expected
        stabilized_expected = stabilize_expected_output_signature(expected)
        if stabilized_expected != expected:
            expected = stabilized_expected
            repaired["expected_output_signature"] = expected
            actions.append("STABILIZED_UNSTABLE_OBJECT_REPRS")

        if not repaired.get("case_kind"):
            inferred_kind = "scaled_seed_case_main" if "seed_case_main" in case_id else "coverage_extension"
            repaired["case_kind"] = inferred_kind
            actions.append(f"SET_CASE_KIND:{inferred_kind}")

        covered_requirements = repaired.get("covered_requirements")
        legacy_requirements = repaired.get("covers_requirements")
        if (not isinstance(covered_requirements, list) or not covered_requirements) and isinstance(legacy_requirements, list):
            repaired["covered_requirements"] = list(legacy_requirements)
            actions.append("MIGRATED_COVERS_REQUIREMENTS")
        if isinstance(repaired.get("covered_requirements"), list):
            repaired["covers_requirements"] = list(repaired.get("covered_requirements") or [])

        call_code = str(repaired.get("call_code") or "").strip()
        if call_code and not _has_result_assignment(call_code):
            repaired["call_code"] = f"{call_code}\nresult = None"
            call_code = str(repaired["call_code"])
            actions.append("APPENDED_RESULT_ASSIGNMENT")

        setup_code = str(repaired.get("setup_code") or "")
        call_print_tokens = _extract_print_tokens(call_code)
        setup_print_tokens = _extract_setup_print_tokens(setup_code)
        stdout_contains = expected.get("stdout_contains")
        if isinstance(stdout_contains, list) and stdout_contains:
            filtered_stdout_contains = []
            removed_tokens: list[str] = []
            for item in stdout_contains:
                token = str(item)
                if token in setup_print_tokens:
                    removed_tokens.append(token)
                    continue
                if _should_remove_stdout_token(token, call_code=call_code, call_print_tokens=call_print_tokens, expected_output_signature=expected):
                    removed_tokens.append(token)
                    continue
                filtered_stdout_contains.append(item)
            if len(filtered_stdout_contains) != len(stdout_contains):
                expected["stdout_contains"] = filtered_stdout_contains
                if any(token in setup_print_tokens for token in removed_tokens):
                    actions.append("REMOVED_SETUP_STDOUT_TOKENS")
                if any(token not in setup_print_tokens for token in removed_tokens):
                    actions.append("REMOVED_CALL_STDOUT_SCaffold_TOKENS")

        setup_artifacts = _normalize_path_set(_extract_setup_created_files(setup_code))
        scaffold_created_files = _normalize_path_set(
            _extract_setup_created_files(str(env.context or ""))
            | _extract_setup_created_files(str(env.code or ""))
            | _extract_setup_created_files(str(env.seed_gold_solution or ""))
            | _extract_setup_created_files(str(env.gold_solution or ""))
        )
        setup_artifacts.update(scaffold_created_files)
        case_requirement_text = _case_requirement_text(repaired)
        file_artifacts = expected.get("file_artifacts")
        if isinstance(file_artifacts, list) and file_artifacts:
            filtered_artifacts = []
            removed_paths: list[str] = []
            removed_log_paths: list[str] = []
            removed_unsafe_paths: list[str] = []
            normalized_paths: list[str] = []
            for artifact in file_artifacts:
                if isinstance(artifact, dict):
                    path = str(artifact.get("path") or "").strip()
                else:
                    path = str(artifact).strip()
                path_result = analyze_relative_path(path, artifact=True)
                if not path_result.safe:
                    removed_unsafe_paths.append(f"{path}:{path_result.reason}")
                    continue
                normalized_path = path_result.normalized_path
                if normalized_path != path:
                    normalized_paths.append(f"{path}->{normalized_path}")
                    if isinstance(artifact, dict):
                        artifact = {**artifact, "path": normalized_path}
                    else:
                        artifact = normalized_path
                if normalized_path.lower().endswith(".log") and not _case_requires_candidate_file_mutation(normalized_path, case_requirement_text):
                    removed_log_paths.append(normalized_path)
                    continue
                if normalized_path and normalized_path in setup_artifacts and not _case_requires_candidate_file_mutation(normalized_path, case_requirement_text):
                    removed_paths.append(path)
                    continue
                filtered_artifacts.append(artifact)
            if removed_paths or removed_log_paths or removed_unsafe_paths or normalized_paths:
                expected["file_artifacts"] = filtered_artifacts
            if normalized_paths:
                actions.append(f"NORMALIZED_FILE_ARTIFACT_PATHS:{','.join(sorted(normalized_paths))}")
            if removed_unsafe_paths:
                actions.append(f"REMOVED_UNSAFE_FILE_ARTIFACT_PATHS:{','.join(sorted(removed_unsafe_paths))}")
            if removed_log_paths:
                actions.append(f"REMOVED_LOG_ARTIFACT_EXPECTATIONS:{','.join(sorted(removed_log_paths))}")
            if removed_paths:
                actions.append(f"REMOVED_SETUP_ARTIFACT_EXPECTATIONS:{','.join(sorted(removed_paths))}")

        inferred_return_type = _infer_expected_return_type(expected, str(repaired.get("call_code") or ""), seed_case)
        current_return_type = str(expected.get("return_type") or "").strip()
        if inferred_return_type and current_return_type != inferred_return_type:
            expected["return_type"] = inferred_return_type
            actions.append(f"ALIGNED_RETURN_TYPE:{inferred_return_type}")

        if not str(repaired.get("description") or "").strip() and seed_case.get("description"):
            repaired["description"] = str(seed_case.get("description") or "").strip()
            actions.append("BACKFILLED_DESCRIPTION_FROM_SEED_CASE")

        generic_repairs = _repair_generic_oracle_case_requirements(repaired, operator_instances or env.operator_instances or [])
        if generic_repairs:
            actions.extend(generic_repairs)

        covered_requirement_ids = [
            str(item).strip()
            for item in (repaired.get("covered_requirement_ids") or [])
            if str(item).strip()
        ]
        inferred_requirement_ids = infer_covered_requirement_ids(env, repaired)
        if not covered_requirement_ids and inferred_requirement_ids:
            repaired["covered_requirement_ids"] = inferred_requirement_ids
            actions.append("INFERRED_COVERED_REQUIREMENT_IDS")
        elif covered_requirement_ids:
            merged_requirement_ids = list(dict.fromkeys([*covered_requirement_ids, *inferred_requirement_ids]))
            if merged_requirement_ids != covered_requirement_ids:
                repaired["covered_requirement_ids"] = merged_requirement_ids
                actions.append("AUGMENTED_COVERED_REQUIREMENT_IDS")

        changed = repaired != original_case
        if changed:
            repaired.setdefault("original_scaled_oracle_case", original_case)
            trace = list(repaired.get("case_repair_trace") or [])
            trace.append(
                {
                    "repair_stage": "rule_repair",
                    "actions": actions,
                }
            )
            repaired["case_repair_trace"] = trace
        repaired_cases.append(repaired)
        report_rows.append(
            {
                "case_id": case_id,
                "changed": changed,
                "actions": actions,
                "warnings": warnings,
            }
        )
    return repaired_cases, report_rows


def _repair_generic_oracle_case_requirements(case: dict[str, Any], operator_instances: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    replacement = _best_specific_case_requirement(case, operator_instances)
    if not replacement:
        return actions
    for field in ["target_constraint", "semantic_intent"]:
        if _is_generic_requirement_text(case.get(field)):
            case[field] = replacement
            actions.append(f"REPLACED_GENERIC_{field.upper()}")
    return actions


def _best_specific_case_requirement(case: dict[str, Any], operator_instances: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for key in ["description", "expected_failure_mode"]:
        text = str(case.get(key) or "").strip()
        if text:
            candidates.append(text)
    for item in (case.get("covered_requirements") or case.get("covers_requirements") or []):
        text = str(item or "").strip()
        if text:
            candidates.append(text)
    target_ids = {item.strip() for item in str(case.get("targets_operator_id") or "").split(",") if item.strip()}
    for operator in operator_instances or []:
        if not isinstance(operator, dict):
            continue
        if target_ids and str(operator.get("operator_id") or "") not in target_ids:
            continue
        for key in ["transformation_goal"]:
            text = str(operator.get(key) or "").strip()
            if text:
                candidates.append(text)
        for item in operator.get("output_requirements") or []:
            text = str(item or "").strip()
            if text:
                candidates.append(text)
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        if _is_generic_requirement_text(candidate):
            continue
        tokens = _specific_tokens(candidate)
        if len(tokens) < 2:
            continue
        scored.append((len(tokens), candidate))
    if not scored:
        return ""
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _extract_setup_print_tokens(setup_code: str) -> set[str]:
    return _extract_print_tokens(setup_code)


def _extract_print_tokens(code: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(r"print\(\s*(['\"])(.*?)\1\s*\)", str(code or ""), flags=re.DOTALL):
        value = match.group(2)
        if value is not None:
            tokens.add(value)
    return tokens


def _extract_setup_created_files(setup_code: str) -> set[str]:
    paths: set[str] = set()
    paths.update(_extract_setup_created_files_ast(setup_code))
    for ref in extract_code_path_references(setup_code):
        if ref.operation in {"path_write_text", "path_write_bytes"}:
            paths.add(ref.path)
    bindings = _extract_string_bindings(setup_code)
    literal_pattern = re.compile(r"open\(\s*(['\"])(?P<path>.+?)\1\s*,\s*(['\"])(?P<mode>[wax][^'\"]*)\3")
    variable_pattern = re.compile(r"open\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*(['\"])(?P<mode>[wax][^'\"]*)\2")
    for match in literal_pattern.finditer(str(setup_code or "")):
        path = str(match.group("path") or "").strip()
        if path:
            paths.add(path)
    for match in variable_pattern.finditer(str(setup_code or "")):
        var_name = str(match.group("var") or "").strip()
        path = bindings.get(var_name, "").strip()
        if path:
            paths.add(path)
    return paths


def _normalize_path_set(paths: set[str]) -> set[str]:
    normalized: set[str] = set()
    for path in paths:
        result = analyze_relative_path(path)
        if result.safe and not result.normalizes_to_workdir:
            normalized.add(result.normalized_path)
    return normalized


def _extract_setup_created_files_ast(setup_code: str) -> set[str]:
    try:
        tree = ast.parse(str(setup_code or ""))
    except SyntaxError:
        return set()
    visitor = _SetupCreatedFileVisitor()
    visitor.visit(tree)
    return visitor.paths


class _SetupCreatedFileVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bindings: dict[str, set[str]] = {}
        self.paths: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        values = _literal_string_values(node.value)
        if values:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.bindings[target.id] = set(values)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        previous: set[str] | None = None
        target_name = node.target.id if isinstance(node.target, ast.Name) else ""
        values = _literal_string_values(node.iter)
        if target_name and values:
            previous = set(self.bindings.get(target_name, set()))
            self.bindings[target_name] = set(values)
        for child in node.body:
            self.visit(child)
        if target_name and values:
            if previous:
                self.bindings[target_name] = previous
            else:
                self.bindings.pop(target_name, None)
        for child in node.orelse:
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_open_call(node) and _open_mode_writes(node):
            self.paths.update(self._paths_from_arg(node.args[0] if node.args else None))
        self.generic_visit(node)

    def _paths_from_arg(self, arg: ast.AST | None) -> set[str]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return {arg.value}
        if isinstance(arg, ast.Name):
            return set(self.bindings.get(arg.id, set()))
        return set()


def _literal_string_values(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: set[str] = set()
        for item in node.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.add(item.value)
            else:
                return set()
        return values
    return set()


def _is_open_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "open"


def _open_mode_writes(node: ast.Call) -> bool:
    mode_node = node.args[1] if len(node.args) >= 2 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
            break
    if mode_node is None:
        return False
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return bool(mode_node.value) and mode_node.value[0] in {"w", "a", "x"}
    return False


def _case_requirement_text(case: dict[str, Any]) -> str:
    return _flatten_text(
        [
            case.get("description"),
            case.get("semantic_intent"),
            case.get("target_constraint"),
            case.get("expected_failure_mode"),
            case.get("covered_requirements") or case.get("covers_requirements") or [],
        ]
    )


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


def _case_requires_candidate_file_mutation(path: str, case_text: str) -> bool:
    text = str(case_text or "").lower()
    filename = str(path or "").lower()
    if not text or not filename:
        return False
    mutation_words = [
        "append",
        "artifact",
        "create",
        "created",
        "emit",
        "export",
        "modify",
        "modified",
        "mutate",
        "output file",
        "overwrite",
        "produce",
        "save",
        "update",
        "write",
        "written",
    ]
    return filename in text and any(word in text for word in mutation_words)


def _extract_string_bindings(code: str) -> dict[str, str]:
    bindings: dict[str, str] = {}
    pattern = re.compile(r"^\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(?P<value>.*?)\2\s*$")
    for line in str(code or "").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        bindings[str(match.group("var") or "").strip()] = str(match.group("value") or "")
    return bindings


def _should_remove_stdout_token(
    token: str,
    *,
    call_code: str,
    call_print_tokens: set[str],
    expected_output_signature: dict[str, Any],
) -> bool:
    stripped = str(token).strip()
    if not stripped:
        return False
    if stripped in call_print_tokens:
        return False

    lowered = stripped.lower()
    if lowered.startswith("stdout:") or lowered.startswith("stderr:") or lowered.startswith("return code:"):
        return True

    has_call_print = "print(" in str(call_code or "")
    expected_return = expected_output_signature.get("return_value")
    expected_stdout_payload = ""
    if isinstance(expected_return, dict):
        expected_stdout_payload = str(expected_return.get("stdout") or "")
    if not has_call_print and expected_stdout_payload and stripped in expected_stdout_payload:
        return True
    return False


def _infer_expected_return_type(
    expected_output_signature: dict[str, Any],
    call_code: str,
    seed_case: dict[str, Any],
) -> str:
    if "return_value" in expected_output_signature:
        value = expected_output_signature.get("return_value")
        if value is not None:
            return type(value).__name__
    inferred_from_assignment = _infer_result_assignment_type(call_code)
    if inferred_from_assignment:
        return inferred_from_assignment
    seed_expected = seed_case.get("expected_output_signature")
    if isinstance(seed_expected, dict):
        seed_return_type = str(seed_expected.get("return_type") or "").strip()
        if seed_return_type:
            return seed_return_type
        if "return_value" in seed_expected and seed_expected.get("return_value") is not None:
            return type(seed_expected.get("return_value")).__name__
    return ""


def _infer_result_assignment_type(call_code: str) -> str:
    result_matches = re.findall(r"(^|\n)\s*result\s*=\s*(.+)", str(call_code or ""))
    if not result_matches:
        return ""
    rhs = str(result_matches[-1][1] or "").strip()
    if rhs.startswith("{"):
        return "dict"
    if rhs.startswith("["):
        return "list"
    if rhs.startswith("("):
        return "tuple"
    if rhs.startswith(("'", '"', "f'", 'f"')):
        return "str"
    if rhs in {"None"}:
        return "NoneType"
    if rhs in {"True", "False"}:
        return "bool"
    if re.fullmatch(r"-?\d+", rhs):
        return "int"
    if re.fullmatch(r"-?\d+\.\d*", rhs):
        return "float"
    return ""


def _has_result_assignment(call_code: str) -> bool:
    return bool(re.search(r"(^|\n)\s*result\s*=", str(call_code or "")))


def _aligned_output_constraint_spec(
    base_spec: dict[str, Any] | None,
    validated_oracle_cases: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    oracle_case_spec = output_constraints_from_scaled_oracle_cases(validated_oracle_cases or [])
    if oracle_case_spec.get("checks"):
        return oracle_case_spec
    return base_spec or {"checks": [], "require_full_coverage": True}


def validate_and_repair_oracle_cases(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    oracle_case_candidates: list[dict[str, Any]],
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    stage05_cfg = config.get("stage05_cfg") or {}
    repair_cfg = stage05_cfg.get("oracle_case_repair", {}) if isinstance(stage05_cfg, dict) else {}
    repair_enabled = bool(repair_cfg.get("enabled", True))
    max_rounds = int(repair_cfg.get("max_rounds", 3))

    current_cases = list(oracle_case_candidates or [])
    repair_trace: list[dict[str, Any]] = []
    for attempt in range(max_rounds + 1):
        current_cases, rule_repair_report = _rule_repair_scaled_oracle_cases(env, current_cases, operator_instances)
        validated_cases, invalid_cases, report_rows, summary = validate_scaled_oracle_cases(
            env=env,
            operator_instances=operator_instances,
            cases=current_cases,
        )
        coverage_gaps = _oracle_case_requirement_coverage_gaps(env, validated_cases)
        if (not invalid_cases and not coverage_gaps) or not repair_enabled or attempt >= max_rounds:
            return {
                "scaled_oracle_cases": current_cases,
                "validated_oracle_cases": validated_cases,
                "scaled_oracle_case_failures": [
                    {
                        "case_id": str(row.get("case_id") or ""),
                        "failure_code": "SCALED_ORACLE_CASE_INVALID",
                        "failure_message": ",".join(str(item) for item in row.get("failure_reasons", []) or []),
                    }
                    for row in report_rows
                    if not bool(row.get("valid"))
                ],
                "oracle_case_validation_report": report_rows,
                "oracle_case_validation_summary": summary,
                "oracle_case_rule_repair_report": rule_repair_report,
                "oracle_case_repair_trace": repair_trace,
            }
        failure_summary = [
            {
                "case_id": str(row.get("case_id") or ""),
                "failure_reasons": list(row.get("failure_reasons", []) or []),
            }
            for row in report_rows
            if not bool(row.get("valid"))
        ]
        failure_summary.extend(coverage_gaps)
        repaired_cases = _repair_scaled_oracle_cases(
            env=env,
            operator_instances=operator_instances,
            current_cases=current_cases,
            validation_report=report_rows,
            failure_summary=failure_summary,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
        )
        repair_trace.append(
            {
                "attempt": attempt + 1,
                "rule_repair_report": rule_repair_report,
                "failure_summary": failure_summary,
                "coverage_gap_count": len(coverage_gaps),
                "repaired_case_count": len(repaired_cases),
            }
        )
        if repaired_cases:
            current_cases = repaired_cases
    return {
        "scaled_oracle_cases": current_cases,
        "validated_oracle_cases": [],
        "scaled_oracle_case_failures": [{"case_id": "scaled_oracle_cases", "failure_code": "ORACLE_CASE_REPAIR_EXHAUSTED", "failure_message": "Oracle case repair exhausted."}],
        "oracle_case_validation_report": [],
        "oracle_case_validation_summary": {},
        "oracle_case_rule_repair_report": [],
        "oracle_case_repair_trace": repair_trace,
    }


def _oracle_case_requirement_coverage_gaps(
    env: ExecutableEnvSpec,
    validated_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = [row for row in (env.output_requirement_metadata or []) if isinstance(row, dict)]
    if not metadata:
        return []

    covered_ids: set[str] = set()
    weak_ids: set[str] = set()
    for case in validated_cases:
        if not isinstance(case, dict):
            continue
        for check in case.get("bound_requirement_checks") or []:
            if not isinstance(check, dict):
                continue
            req_id = str(check.get("requirement_id") or "")
            if not req_id:
                continue
            if check.get("passed"):
                covered_ids.add(req_id)
            else:
                weak_ids.add(req_id)
        for req_id in case.get("covered_requirement_ids") or []:
            req_text = str(req_id).strip()
            if req_text and req_text not in covered_ids:
                weak_ids.add(req_text)

    gaps: list[dict[str, Any]] = []
    for row in metadata:
        req_id = str(row.get("requirement_id") or "")
        requirement = str(row.get("text") or "").strip()
        if not req_id or not requirement or not bool(row.get("required_coverage", True)):
            continue
        if _is_generic_requirement_text(requirement):
            continue
        if req_id in covered_ids:
            continue
        failure_code = "REQUIREMENT_ID_HAS_WEAK_CASE_ONLY" if req_id in weak_ids else "MISSING_REQUIREMENT_COVERAGE"
        gaps.append(
            {
                "case_id": f"missing_coverage:{req_id}",
                "failure_reasons": [failure_code],
                "missing_requirement_id": req_id,
                "missing_requirement": requirement,
                "operator_id": row.get("operator_id"),
                "axis": row.get("axis"),
                "repair_instruction": (
                    "Add or repair an oracle case whose covered_requirement_ids contains this exact requirement_id. "
                    "The case setup/call/expected_output_signature must make the requirement observable."
                ),
            }
        )
    return gaps


def _repair_oracle_cases_for_quality_gate(
    env: ExecutableEnvSpec,
    generation: dict[str, Any],
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    output_constraint_spec: dict[str, Any] | None,
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    stage05_cfg = config.get("stage05_cfg") or {}
    repair_cfg = stage05_cfg.get("oracle_case_quality_repair") or stage05_cfg.get("oracle_case_repair", {})
    if isinstance(repair_cfg, dict) and not bool(repair_cfg.get("enabled", True)):
        return {"repaired": False}
    max_rounds = int(repair_cfg.get("max_rounds", 1)) if isinstance(repair_cfg, dict) else 1
    max_rounds = max(0, max_rounds)
    current_generation = dict(generation)
    current_cases = list(current_generation.get("validated_oracle_cases") or current_generation.get("scaled_oracle_cases") or [])
    if not current_generation.get("scaled_executable_gold_code"):
        return {"repaired": False}

    repaired = False
    latest_evaluation: dict[str, Any] | None = None
    for attempt in range(max_rounds):
        gate_env = env.model_copy(
            update={
                "scaled_oracle_cases": current_cases,
                "validated_oracle_cases": current_cases,
                "oracle_case_validation_report": current_generation.get("oracle_case_validation_report", []),
                "scaled_gold_case_execution_report": current_generation.get("scaled_gold_case_execution_report", []),
            }
        )
        gate = run_oracle_case_quality_gate({"scaled_task": gate_env}, config=config)
        coverage_gaps = _quality_gate_requirement_coverage_gaps(env, gate)
        if not coverage_gaps:
            break

        repaired_cases = _repair_scaled_oracle_cases(
            env=env,
            operator_instances=operator_instances,
            current_cases=current_cases,
            validation_report=list(current_generation.get("oracle_case_validation_report") or []),
            failure_summary=coverage_gaps,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
        )
        trace_row = {
            "attempt": f"quality_gate_coverage_repair_{attempt + 1}",
            "gate_failure_reasons": list(gate.get("failure_reasons") or []),
            "missing_requirement_ids": [str(item.get("missing_requirement_id") or "") for item in coverage_gaps],
            "repaired_case_count": len(repaired_cases),
        }
        current_generation["oracle_case_repair_trace"] = list(current_generation.get("oracle_case_repair_trace") or [])
        current_generation["oracle_case_repair_trace"].append(trace_row)
        if not repaired_cases:
            break

        oracle_case_result = validate_and_repair_oracle_cases(
            env=env,
            operator_instances=operator_instances,
            oracle_case_candidates=repaired_cases,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            config=config,
        )
        validated_cases = list(oracle_case_result.get("validated_oracle_cases") or [])
        oracle_case_result, validated_cases, merge_failures = _merge_seed_regression_case_into_oracle_result(
            env=env,
            oracle_case_result=oracle_case_result,
            generation_failures=[],
        )
        if not validated_cases:
            current_generation["scaled_oracle_case_failures"] = list(current_generation.get("scaled_oracle_case_failures") or [])
            current_generation["scaled_oracle_case_failures"].extend(list(oracle_case_result.get("scaled_oracle_case_failures") or []))
            break

        aligned_spec = _aligned_output_constraint_spec(output_constraint_spec, validated_cases)
        latest_evaluation = evaluate_scaled_gold_candidate(
            env=env,
            candidate_code=str(current_generation.get("scaled_executable_gold_code") or ""),
            scaled_oracle_cases=validated_cases,
            hidden_tests=[],
            semantic_test_specs=semantic_test_specs,
            output_constraint_spec=aligned_spec,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            config=config,
        )
        current_cases = validated_cases
        current_generation.update(
            {
                "scaled_oracle_cases": validated_cases,
                "validated_oracle_cases": validated_cases,
                "oracle_case_validation_report": list(oracle_case_result.get("oracle_case_validation_report") or []),
                "oracle_case_rule_repair_report": list(current_generation.get("oracle_case_rule_repair_report") or [])
                + list(oracle_case_result.get("oracle_case_rule_repair_report") or []),
                "scaled_oracle_case_failures": list(oracle_case_result.get("scaled_oracle_case_failures") or []) + list(merge_failures or []),
                "scaled_oracle_coverage_summary": _scaled_oracle_coverage_summary(
                    validated_cases,
                    _semantic_operator_ids_from_cases_or_operators(validated_cases, operator_instances),
                ),
                "output_constraint_spec_aligned": aligned_spec,
                "scaled_gold_case_execution_report": latest_evaluation.get("scaled_gold_case_execution_report", []),
            }
        )
        repaired = True

    if not repaired or latest_evaluation is None:
        return {"repaired": False}
    return {"repaired": True, "generation": current_generation, "evaluation": latest_evaluation}


def _quality_gate_requirement_coverage_gaps(env: ExecutableEnvSpec, gate: dict[str, Any]) -> list[dict[str, Any]]:
    failure_reasons = {str(item) for item in (gate.get("failure_reasons") or [])}
    warnings = {str(item) for item in (gate.get("warnings") or [])}
    if (
        "NO_CASE_COVERS_NEW_REQUIREMENT" not in failure_reasons
        and "PARTIAL_CASE_REQUIREMENT_COVERAGE" not in failure_reasons
        and "PARTIAL_CASE_REQUIREMENT_COVERAGE" not in warnings
    ):
        return []
    evidence = gate.get("evidence") or {}
    missing_ids = [str(item).strip() for item in (evidence.get("missing_requirement_ids") or []) if str(item).strip()]
    if not missing_ids and "NO_CASE_COVERS_NEW_REQUIREMENT" in failure_reasons:
        missing_ids = [str(item).strip() for item in (evidence.get("coverage_target_ids") or []) if str(item).strip()]
    metadata_by_id = {
        str(row.get("requirement_id") or ""): row
        for row in (env.output_requirement_metadata or [])
        if isinstance(row, dict) and str(row.get("requirement_id") or "")
    }
    gaps: list[dict[str, Any]] = []
    for req_id in missing_ids:
        row = metadata_by_id.get(req_id, {})
        requirement = str(row.get("text") or "").strip()
        gaps.append(
            {
                "case_id": f"missing_quality_coverage:{req_id}",
                "failure_reasons": ["QUALITY_GATE_MISSING_REQUIREMENT_COVERAGE"],
                "missing_requirement_id": req_id,
                "missing_requirement": requirement,
                "operator_id": row.get("operator_id"),
                "axis": row.get("axis"),
                "repair_instruction": (
                    "Add or repair an oracle case whose covered_requirement_ids contains this exact requirement_id. "
                    "The case description, target_constraint, setup_code, call_code, expected_failure_mode, and "
                    "expected_output_signature must make the requirement observable. Do not satisfy this by ID only."
                ),
            }
        )
    return gaps


def _semantic_operator_ids_from_cases_or_operators(
    cases: list[dict[str, Any]],
    operator_instances: list[dict[str, Any]],
) -> list[str]:
    ids = [
        str(item).strip()
        for case in cases
        for item in str(case.get("targets_operator_id") or "").split(",")
        if str(item).strip()
    ]
    if ids:
        return _dedupe_strings(ids)
    return _dedupe_strings([str(operator.get("operator_id") or "") for operator in operator_instances if str(operator.get("operator_id") or "")])


def _should_add_fallback_oracle_case(env: ExecutableEnvSpec, operator_instances: list[dict[str, Any]]) -> bool:
    level = str((env.difficulty.global_level if env.difficulty else "") or "")
    if level == "M1":
        return False
    return bool(operator_instances)


def _merge_seed_regression_case_into_oracle_result(
    env: ExecutableEnvSpec,
    oracle_case_result: dict[str, Any],
    generation_failures: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    level = str((env.difficulty.global_level if env.difficulty else "") or "")
    validated_cases = list(oracle_case_result.get("validated_oracle_cases") or [])
    merged_cases, regression_failures = merge_seed_regression_case(env, validated_cases, level=level)
    if regression_failures:
        return oracle_case_result, validated_cases, generation_failures + regression_failures
    merged_cases, regression_rule_repair_report = _rule_repair_scaled_oracle_cases(env, merged_cases, [])
    if merged_cases == validated_cases:
        return oracle_case_result, validated_cases, generation_failures

    updated_result = dict(oracle_case_result)
    updated_result["validated_oracle_cases"] = merged_cases
    updated_result["scaled_oracle_cases"] = merged_cases
    updated_result["oracle_case_rule_repair_report"] = list(updated_result.get("oracle_case_rule_repair_report") or [])
    updated_result["oracle_case_rule_repair_report"].extend(regression_rule_repair_report)
    report = list(updated_result.get("oracle_case_validation_report") or [])
    regression_case = next(
        (case for case in merged_cases if isinstance(case, dict) and str(case.get("case_kind") or "") == "seed_regression"),
        None,
    )
    if regression_case and not any(str(row.get("case_id") or "") == str(regression_case.get("case_id") or "") for row in report):
        report.append(seed_regression_validation_report_row(env, regression_case))
    updated_result["oracle_case_validation_report"] = report
    return updated_result, merged_cases, generation_failures


def _build_m1_seed_baseline_oracle_case(env: ExecutableEnvSpec) -> dict[str, Any] | None:
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    if not seed_case:
        return None
    call_code = str(seed_case.get("call_code") or "").strip()
    expected = seed_case.get("expected_output_signature")
    if not call_code or not isinstance(expected, dict) or not expected:
        return None
    target_name = _target_name_from_signature(env.signature)
    requirements = build_seed_behavior_requirements(env)
    if not requirements:
        requirements = [
            (
                f"Validate original seed behavior for {target_name}."
                if target_name
                else "Validate original seed task behavior."
            )
        ]
    requirement = requirements[0]
    return {
        "case_id": str(seed_case.get("case_id") or "seed_case_main"),
        "description": str(seed_case.get("description") or "M1 baseline oracle case derived from the seed execution case."),
        "case_kind": "seed_baseline",
        "targets_operator_id": "",
        "axis": "M1",
        "semantic_intent": requirement,
        "target_constraint": requirement,
        "expected_failure_mode": "candidate does not satisfy the original seed task behavior",
        "setup_code": str(seed_case.get("setup_code") or ""),
        "call_code": call_code,
        "assertion_code": str(seed_case.get("assertion_code") or ""),
        "covered_requirements": requirements,
        "covers_requirements": requirements,
        "expected_output_signature": stabilize_expected_output_signature(dict(expected)),
    }


def _target_name_from_signature(signature: str | None) -> str:
    text = str(signature or "").strip()
    if text.startswith("def "):
        return text[4:].split("(", 1)[0].strip()
    if text.startswith("class "):
        return text[6:].split("(", 1)[0].split(":", 1)[0].strip()
    return ""


def _build_fallback_oracle_case(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    blueprint = (env.scaled_case_plan or {}).get("main_case_blueprint") if isinstance(env.scaled_case_plan, dict) else None
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    source = blueprint if isinstance(blueprint, dict) and str(blueprint.get("call_code") or "").strip() else seed_case
    if not isinstance(source, dict) or not str(source.get("call_code") or "").strip():
        return None
    expected = source.get("expected_output_signature")
    if not isinstance(expected, dict) or not expected:
        expected = seed_case.get("expected_output_signature") if isinstance(seed_case.get("expected_output_signature"), dict) else {}
    if not expected:
        return None
    operator_ids = [str(op.get("operator_id") or "").strip() for op in operator_instances if str(op.get("operator_id") or "").strip()]
    axes = [str(op.get("axis") or "").strip() for op in operator_instances if str(op.get("axis") or "").strip()]
    requirement = _fallback_case_requirement(env, operator_instances, semantic_test_specs, source)
    return {
        "case_id": str(source.get("case_id") or "fallback_seed_oracle_case"),
        "description": str(source.get("description") or "Fallback executable oracle case derived from the seed case."),
        "case_kind": str(source.get("case_kind") or "fallback_seed_case"),
        "targets_operator_id": ",".join(operator_ids),
        "axis": ",".join(dict.fromkeys(axes)),
        "semantic_intent": requirement,
        "target_constraint": requirement,
        "expected_failure_mode": "scaled requirement is not exercised by an executable oracle case",
        "setup_code": str(source.get("setup_code") or ""),
        "call_code": str(source.get("call_code") or ""),
        "assertion_code": str(source.get("assertion_code") or ""),
        "covered_requirements": [requirement],
        "covers_requirements": [requirement],
        "expected_output_signature": stabilize_expected_output_signature(dict(expected)),
    }


def _fallback_case_requirement(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    source_case: dict[str, Any],
) -> str:
    candidates: list[str] = []
    for spec in semantic_test_specs or []:
        candidates.extend(
            str(spec.get(key) or "").strip()
            for key in ["target_constraint", "semantic_intent", "test_case_description"]
            if str(spec.get(key) or "").strip()
        )
    for operator in operator_instances or []:
        candidates.extend(str(item).strip() for item in (operator.get("output_requirements") or []) if str(item).strip())
        for key in ["transformation_goal", "rationale", "operator_type"]:
            text = str(operator.get(key) or "").strip()
            if text:
                candidates.append(text)
    for item in (source_case.get("covered_requirements") or source_case.get("covers_requirements") or []):
        if str(item).strip():
            candidates.append(str(item).strip())
    for item in _additional_requirement_lines(str(env.user_prompt or "")):
        candidates.append(item)
    for candidate in candidates:
        if candidate and not _is_generic_requirement_text(candidate):
            return candidate
    return "The scaled task must keep the seed executable behavior covered by a validated oracle case."


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


def _build_hidden_tests_from_scaled_oracle_cases(
    scaled_oracle_cases: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hidden_tests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(scaled_oracle_cases or [], start=1):
        if not isinstance(case, dict):
            failures.append(
                {
                    "case_id": f"scaled_oracle_case_{index}",
                    "failure_code": "SCALED_ORACLE_CASE_INVALID",
                    "failure_message": "Scaled oracle case must be a dict.",
                }
            )
            continue
        case_id = str(case.get("case_id") or case.get("example_id") or case.get("test_id") or f"scaled_oracle_case_{index}")
        code = _compile_hidden_test_code_from_scaled_oracle_case(case)
        if not code:
            failures.append(
                {
                    "case_id": case_id,
                    "failure_code": "SCALED_ORACLE_CASE_INVALID",
                    "failure_message": "Scaled oracle case could not be compiled into a hidden test.",
                }
            )
            continue
        try:
            compile(code, f"{case_id}.py", "exec")
        except SyntaxError as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "failure_code": "SCALED_ORACLE_CASE_INVALID",
                    "failure_message": f"compile failed: {exc.msg}",
                }
            )
            continue
        hidden_tests.append(
            {
                "test_id": case_id,
                "name": case_id,
                "targets_operator_id": str(case.get("targets_operator_id") or ""),
                "axis": str(case.get("axis") or ""),
                "semantic_intent": str(case.get("semantic_intent") or case.get("description") or ""),
                "target_constraint": str(case.get("target_constraint") or ""),
                "expected_failure_mode": str(case.get("expected_failure_mode") or ""),
                "code": code,
                "assertion_code": code,
                "source": "scaled_oracle_case",
                "is_semantic": True,
                "is_placeholder": False,
                "test_tier": "semantic",
                "counts_as_hidden_test": True,
                "eligible_for_clean_export": True,
            }
        )
    return hidden_tests, failures


def _compile_hidden_test_code_from_scaled_oracle_case(case: dict[str, Any]) -> str:
    setup_code = str(case.get("setup_code") or "").strip()
    call_code = str(case.get("call_code") or "").strip()
    assertion_code = str(case.get("assertion_code") or "").strip()
    expected_output_signature = (
        case.get("expected_output_signature")
        if isinstance(case.get("expected_output_signature"), dict)
        else {}
    )
    if not call_code or not expected_output_signature:
        return ""

    code_parts = []
    if setup_code:
        code_parts.append(setup_code)
    code_parts.append(_instrument_call_code(call_code))
    compiled_assertions = _compile_assertions_from_expected_output_signature(expected_output_signature)
    if compiled_assertions:
        code_parts.append(compiled_assertions)
    if assertion_code:
        code_parts.append(assertion_code)
    return "\n\n".join(part for part in code_parts if part).strip()


def _instrument_call_code(call_code: str) -> str:
    return "\n".join(
        [
            "import contextlib",
            "import io",
            "_scaled_oracle_stdout = io.StringIO()",
            "_scaled_oracle_stderr = io.StringIO()",
            "with contextlib.redirect_stdout(_scaled_oracle_stdout), contextlib.redirect_stderr(_scaled_oracle_stderr):",
            _indent_block(call_code, "    "),
            "stdout = _scaled_oracle_stdout.getvalue()",
            "stderr = _scaled_oracle_stderr.getvalue()",
            "rc = 0",
        ]
    )


def _compile_assertions_from_expected_output_signature(expected: dict[str, Any]) -> str:
    lines: list[str] = []
    return_type = str(expected.get("return_type") or "").strip()
    if return_type:
        lines.append(f"assert type(result).__name__ == {return_type!r}, f'unexpected return type: {{type(result).__name__}}'")

    return_keys = expected.get("return_keys")
    if isinstance(return_keys, list) and return_keys:
        keys = [str(item) for item in return_keys if str(item).strip()]
        lines.append("assert isinstance(result, dict), 'result must be a dict'")
        lines.append(f"_expected_keys = {keys!r}")
        lines.append("assert all(key in result for key in _expected_keys), f'missing dict keys: {[key for key in _expected_keys if key not in result]}'")

    if "return_value" in expected:
        lines.append(f"assert result == {repr(expected.get('return_value'))}, f'unexpected return value: {{result!r}}'")

    stdout_contains = expected.get("stdout_contains")
    if isinstance(stdout_contains, list):
        for token in stdout_contains:
            value = str(token).strip()
            if value:
                lines.append(f"assert {value!r} in stdout, 'missing expected stdout token'")

    stdout_regex = expected.get("stdout_regex")
    if isinstance(stdout_regex, list):
        lines.append("import re")
        for pattern in stdout_regex:
            value = str(pattern).strip()
            if value:
                lines.append(f"assert re.search(r{value!r}, stdout), 'stdout regex did not match'")

    file_artifacts = expected.get("file_artifacts")
    if isinstance(file_artifacts, list):
        path_lines = []
        for artifact in file_artifacts:
            if isinstance(artifact, dict):
                path = str(artifact.get("path") or "").strip()
            else:
                path = str(artifact).strip()
            if path:
                path_lines.append(path)
        if path_lines:
            lines.append("from pathlib import Path")
            for path in path_lines:
                lines.append(f"assert Path({path!r}).exists(), 'missing expected file artifact'")

    return "\n".join(lines).strip()


def _indent_block(block: str, indent: str) -> str:
    return "\n".join(f"{indent}{line}" if line.strip() else "" for line in str(block).splitlines())


def generate_scaled_gold_solution_if_needed(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    output_constraint_spec: dict[str, Any] | None,
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    config: dict | None = None,
) -> dict[str, Any]:
    config = config or {}
    seed_gold_solution = env.seed_gold_solution or env.gold_solution
    semantic_info = detect_semantic_change(operator_instances)
    semantic_operator_ids = _oracle_case_operator_ids(operator_instances, semantic_info)
    scaled_case_plan = dict(env.scaled_case_plan or build_scaled_case_plan(env, operator_instances, semantic_test_specs))
    seed_case_admission = check_seed_case_admission(env)
    target_oracle_count = _resolve_oracle_case_target_count(env, semantic_info, config)
    scaled_case_plan["required_validated_case_count"] = target_oracle_count
    level = str((env.difficulty.global_level if env.difficulty else "") or "")
    if level != "M1" and level not in {"M2", "M3", "M4"} and not semantic_info["semantic_change"] and target_oracle_count <= 0:
        seed_executable_code = _seed_executable_code(env)
        return {
            "seed_gold_solution": seed_gold_solution,
            "scaled_gold_solution": seed_executable_code,
            "scaled_executable_gold_code": seed_executable_code,
            "scaled_oracle_cases": [],
            "validated_oracle_cases": [],
            "scaled_oracle_case_failures": [],
            "scaled_oracle_coverage_summary": _scaled_oracle_coverage_summary([], semantic_operator_ids),
            "gold_changed": False,
            "answer_invariant": True,
            "gold_generation_method": "reuse_seed_gold_solution",
            "gold_change_reason": "V-axis strengthens verifier coverage only; task semantics unchanged.",
            "seed_gold_compatible_with_scaled_task": True,
            "covered_operator_ids": semantic_operator_ids,
            "covered_requirements": [],
            "compile_passed": True,
            "visible_tests_passed": True,
            "hidden_tests_passed": True,
            "scaled_ground_truth_output_signature": dict(env.seed_ground_truth_output_signature or {}),
            "output_constraint_result": {"passed": True, "passed_checks": [], "failed_checks": []},
            "hidden_tests": [],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "oracle_case_validation_report": [],
            "scaled_gold_case_execution_report": [],
            "scaled_case_plan": scaled_case_plan,
            "repair_trace": [],
            "repair_attempts": 0,
            "failure_reasons": [],
        }

    gold_policies = [operator.get("gold_update_policy") or {} for operator in operator_instances]
    if (
        level != "M1"
        and level not in {"M2", "M3", "M4"}
        and target_oracle_count <= 0
        and gold_policies
        and all(bool(policy.get("answer_invariant")) for policy in gold_policies if policy)
    ):
        reason = "; ".join(str(policy.get("gold_change_reason") or "") for policy in gold_policies if policy)
        seed_executable_code = _seed_executable_code(env)
        return {
            "seed_gold_solution": seed_gold_solution,
            "scaled_gold_solution": seed_executable_code,
            "scaled_executable_gold_code": seed_executable_code,
            "scaled_oracle_cases": [],
            "validated_oracle_cases": [],
            "scaled_oracle_case_failures": [],
            "scaled_oracle_coverage_summary": _scaled_oracle_coverage_summary([], semantic_operator_ids),
            "gold_changed": False,
            "answer_invariant": True,
            "gold_generation_method": "reuse_seed_gold_solution",
            "gold_change_reason": reason or "Original gold solution already satisfies all newly added semantic requirements.",
            "seed_gold_compatible_with_scaled_task": True,
            "covered_operator_ids": semantic_operator_ids,
            "covered_requirements": [str(spec.get("spec_id") or "") for spec in semantic_test_specs],
            "compile_passed": True,
            "visible_tests_passed": True,
            "hidden_tests_passed": True,
            "scaled_ground_truth_output_signature": dict(env.seed_ground_truth_output_signature or {}),
            "output_constraint_result": {"passed": True, "passed_checks": [], "failed_checks": []},
            "hidden_tests": [],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "oracle_case_validation_report": [],
            "scaled_gold_case_execution_report": [],
            "scaled_case_plan": scaled_case_plan,
            "repair_trace": [],
            "repair_attempts": 0,
            "failure_reasons": [],
        }

    if not seed_case_admission["passed"]:
        return {
            "seed_gold_solution": seed_gold_solution,
            "scaled_gold_solution": "",
            "scaled_executable_gold_code": "",
            "scaled_oracle_cases": [],
            "validated_oracle_cases": [],
            "scaled_oracle_case_failures": [],
            "scaled_oracle_coverage_summary": _scaled_oracle_coverage_summary([], semantic_operator_ids),
            "gold_changed": True,
            "answer_invariant": False,
            "gold_generation_method": "blocked_by_seed_case_admission",
            "gold_change_reason": "Seed execution case admission failed; Stage05 case-first flow aborted before oracle generation.",
            "seed_gold_compatible_with_scaled_task": False,
            "covered_operator_ids": [],
            "covered_requirements": [],
            "compile_passed": False,
            "visible_tests_passed": False,
            "hidden_tests_passed": True,
            "scaled_ground_truth_output_signature": dict(env.seed_ground_truth_output_signature or {}),
            "output_constraint_result": {"passed": False, "passed_checks": [], "failed_checks": []},
            "hidden_tests": [],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "oracle_case_validation_report": [],
            "scaled_gold_case_execution_report": [],
            "scaled_case_plan": scaled_case_plan,
            "repair_trace": [],
            "repair_attempts": 0,
            "failure_reasons": list(seed_case_admission["failure_reasons"]),
        }

    if level == "M1":
        seed_baseline_case = _build_m1_seed_baseline_oracle_case(env)
        scaled_oracle_cases = [seed_baseline_case] if seed_baseline_case else []
        generation_failures = [] if seed_baseline_case else [
            {
                "case_id": "seed_case_main",
                "failure_code": "M1_SEED_BASELINE_CASE_MISSING",
                "failure_message": "M1 seed_execution_case could not be wrapped as a validated oracle case.",
            }
        ]
    else:
        scaled_oracle_cases, generation_failures = _generate_scaled_oracle_cases(
            env=env,
            operator_instances=operator_instances,
            semantic_test_specs=semantic_test_specs,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            target_count=target_oracle_count,
        )
    oracle_case_result = validate_and_repair_oracle_cases(
        env=env,
        operator_instances=operator_instances,
        oracle_case_candidates=scaled_oracle_cases,
        llm_client=llm_client,
        prompt_runner=prompt_runner,
        config=config,
    )
    validated_oracle_cases = list(oracle_case_result.get("validated_oracle_cases") or [])
    if not validated_oracle_cases and _should_add_fallback_oracle_case(env, operator_instances):
        fallback_case = _build_fallback_oracle_case(env, operator_instances, semantic_test_specs)
        if fallback_case:
            fallback_result = validate_and_repair_oracle_cases(
                env=env,
                operator_instances=operator_instances,
                oracle_case_candidates=[fallback_case],
                llm_client=llm_client,
                prompt_runner=prompt_runner,
                config=config,
            )
            fallback_validated = list(fallback_result.get("validated_oracle_cases") or [])
            fallback_trace = list(oracle_case_result.get("oracle_case_repair_trace") or [])
            fallback_trace.append(
                {
                    "attempt": "fallback_seed_oracle_case",
                    "actions": [
                        "FALLBACK_SEED_ORACLE_CASE_ADDED",
                        "FALLBACK_SEED_ORACLE_CASE_VALIDATED" if fallback_validated else "FALLBACK_SEED_ORACLE_CASE_REJECTED",
                    ],
                    "validation_report": list(fallback_result.get("oracle_case_validation_report") or []),
                }
            )
            if fallback_validated:
                oracle_case_result = dict(fallback_result)
                oracle_case_result["oracle_case_repair_trace"] = fallback_trace
                validated_oracle_cases = fallback_validated
            else:
                oracle_case_result = dict(oracle_case_result)
                oracle_case_result["oracle_case_repair_trace"] = fallback_trace
                oracle_case_result["scaled_oracle_case_failures"] = list(oracle_case_result.get("scaled_oracle_case_failures") or [])
                oracle_case_result["scaled_oracle_case_failures"].extend(list(fallback_result.get("scaled_oracle_case_failures") or []))

    oracle_case_result, validated_oracle_cases, generation_failures = _merge_seed_regression_case_into_oracle_result(
        env=env,
        oracle_case_result=oracle_case_result,
        generation_failures=generation_failures,
    )

    answer_invariant = (
        not semantic_info["semantic_change"]
        or (
            bool(gold_policies)
            and all(bool(policy.get("answer_invariant")) for policy in gold_policies if policy)
        )
    )
    if answer_invariant:
        seed_executable_code = _seed_executable_code(env)
        reason = "; ".join(str(policy.get("gold_change_reason") or "") for policy in gold_policies if policy)
        generation = {
            "seed_gold_solution": seed_gold_solution,
            "scaled_gold_solution": seed_executable_code,
            "scaled_executable_gold_code": seed_executable_code,
            "gold_changed": False,
            "answer_invariant": True,
            "gold_generation_method": "reuse_seed_gold_solution",
            "gold_change_reason": reason or "Task answer is invariant; Stage05 still validates the required oracle cases.",
            "seed_gold_compatible_with_scaled_task": True,
            "covered_operator_ids": list(semantic_operator_ids),
            "covered_requirements": _dedupe_strings(
                [
                    str(item).strip()
                    for case in validated_oracle_cases
                    for item in ((case.get("covered_requirements") or case.get("covers_requirements")) or [])
                    if str(item).strip()
                ]
            ),
            "compile_passed": True,
            "visible_tests_passed": True,
            "hidden_tests_passed": True,
            "scaled_ground_truth_output_signature": dict(env.seed_ground_truth_output_signature or {}),
            "output_constraint_result": {"passed": True, "passed_checks": [], "failed_checks": []},
            "hidden_tests": [],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "repair_trace": [],
            "repair_attempts": 0,
            "failure_reasons": [],
        }
    else:
        generation = _generate_scaled_gold_solution(
            env=env,
            operator_instances=operator_instances,
            semantic_test_specs=semantic_test_specs,
            scaled_oracle_cases=validated_oracle_cases,
            hidden_tests=[],
            llm_client=llm_client,
            prompt_runner=prompt_runner,
        )
    generation["scaled_oracle_cases"] = list(validated_oracle_cases)
    generation["validated_oracle_cases"] = validated_oracle_cases
    generation["oracle_case_validation_report"] = list(oracle_case_result.get("oracle_case_validation_report") or [])
    generation["oracle_case_rule_repair_report"] = list(oracle_case_result.get("oracle_case_rule_repair_report") or [])
    generation["oracle_case_repair_trace"] = list(oracle_case_result.get("oracle_case_repair_trace") or [])
    generation["scaled_oracle_case_failures"] = list(generation.get("scaled_oracle_case_failures") or [])
    generation["scaled_oracle_case_failures"].extend(generation_failures)
    generation["scaled_oracle_case_failures"].extend(list(oracle_case_result.get("scaled_oracle_case_failures") or []))
    generation["scaled_oracle_coverage_summary"] = _scaled_oracle_coverage_summary(validated_oracle_cases, semantic_operator_ids)
    generation["scaled_case_plan"] = scaled_case_plan
    generation["output_constraint_spec_aligned"] = _aligned_output_constraint_spec(output_constraint_spec or {}, validated_oracle_cases)
    if not generation["scaled_executable_gold_code"]:
        generation["failure_reasons"].append("SCALED_GOLD_SOLUTION_MISSING")
        return generation
    evaluation = evaluate_scaled_gold_candidate(
        env=env,
        candidate_code=generation["scaled_executable_gold_code"],
        scaled_oracle_cases=validated_oracle_cases,
        hidden_tests=[],
        semantic_test_specs=semantic_test_specs,
        output_constraint_spec=generation["output_constraint_spec_aligned"],
        llm_client=llm_client,
        prompt_runner=prompt_runner,
        config=config,
    )
    generation.update(evaluation)
    generation["compile_passed"] = evaluation["compile_passed"]
    generation["visible_tests_passed"] = evaluation["execution_passed"]
    generation["hidden_tests_passed"] = evaluation["hidden_tests_passed"]
    generation["scaled_executable_gold_code"] = evaluation["scaled_executable_gold_code"]
    generation["scaled_ground_truth_output_signature"] = evaluation["scaled_ground_truth_output_signature"]
    generation["output_constraint_result"] = evaluation["output_constraint_result"]
    generation["hidden_tests"] = evaluation["hidden_tests"]
    generation["hidden_tests_mode"] = "disabled_in_case_first_stage05"
    generation["scaled_oracle_cases"] = generation.get("scaled_oracle_cases", [])
    generation["validated_oracle_cases"] = evaluation.get("validated_oracle_cases", validated_oracle_cases)
    generation["scaled_gold_case_execution_report"] = evaluation.get("scaled_gold_case_execution_report", [])
    quality_repair = _repair_oracle_cases_for_quality_gate(
        env=env,
        generation=generation,
        operator_instances=operator_instances,
        semantic_test_specs=semantic_test_specs,
        output_constraint_spec=output_constraint_spec,
        llm_client=llm_client,
        prompt_runner=prompt_runner,
        config=config,
    )
    if quality_repair.get("repaired"):
        generation.update(quality_repair["generation"])
        evaluation = quality_repair["evaluation"]
        validated_oracle_cases = list(generation.get("validated_oracle_cases") or [])
        generation["compile_passed"] = evaluation["compile_passed"]
        generation["visible_tests_passed"] = evaluation["execution_passed"]
        generation["hidden_tests_passed"] = evaluation["hidden_tests_passed"]
        generation["scaled_executable_gold_code"] = evaluation["scaled_executable_gold_code"]
        generation["scaled_ground_truth_output_signature"] = evaluation["scaled_ground_truth_output_signature"]
        generation["output_constraint_result"] = evaluation["output_constraint_result"]
        generation["hidden_tests"] = evaluation["hidden_tests"]
        generation["hidden_tests_mode"] = "disabled_in_case_first_stage05"
        generation["scaled_gold_case_execution_report"] = evaluation.get("scaled_gold_case_execution_report", [])
    if not evaluation["compile_passed"]:
        generation["failure_reasons"].append("SCALED_GOLD_COMPILE_FAILED")
    if not evaluation["execution_passed"]:
        generation["failure_reasons"].append("SCALED_GOLD_EXECUTION_FAILED")
    if not evaluation.get("case_first_mode") and not evaluation["output_constraint_result"]["passed"]:
        generation["failure_reasons"].append("SCALED_OUTPUT_CONSTRAINTS_FAILED")
    if target_oracle_count and not validated_oracle_cases:
        generation["failure_reasons"].append("NO_VALIDATED_ORACLE_CASES")
    if generation["scaled_oracle_case_failures"]:
        generation["failure_reasons"].extend(
            f"{item.get('failure_code')}:{item.get('case_id')}" for item in generation["scaled_oracle_case_failures"]
        )
    if generation["scaled_executable_gold_code"].strip() == _seed_executable_code(env).strip() and not bool(generation.get("answer_invariant")):
        generation["failure_reasons"].append("SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION")
    missing_operator_coverage = [
        operator_id for operator_id in semantic_operator_ids if operator_id not in set(generation.get("covered_operator_ids", []))
    ]
    if missing_operator_coverage:
        generation["failure_reasons"].append("SCALED_GOLD_DOES_NOT_COVER_OPERATOR")
    generation["scaled_oracle_case_constraint_spec"] = output_constraints_from_scaled_oracle_cases(validated_oracle_cases)
    generation["scaled_case_plan"] = scaled_case_plan
    return generation


def repair_scaled_gold_solution(
    env: ExecutableEnvSpec,
    gold_result: dict[str, Any],
    semantic_test_specs: list[dict[str, Any]],
    output_constraint_spec: dict[str, Any],
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    config: dict | None = None,
) -> dict[str, Any]:
    config = config or {}
    attempts = int(gold_result.get("repair_attempts", 0))
    stage05_cfg = config.get("stage05_cfg") or {}
    repair_cfg = stage05_cfg.get("scaled_gold_repair", {}) if isinstance(stage05_cfg, dict) else {}
    max_attempts = int(repair_cfg.get("max_rounds", config.get("max_gold_repair_attempts", 3)))
    repair_enabled = bool(repair_cfg.get("enabled", True))
    current = dict(gold_result)
    repair_trace = list(current.get("repair_trace", []))
    while attempts < max_attempts:
        candidate = str(current.get("scaled_executable_gold_code") or current.get("scaled_gold_solution") or env.code or env.gold_solution or "")
        scaled_oracle_cases = list(current.get("validated_oracle_cases") or current.get("scaled_oracle_cases", []))
        evaluation = evaluate_scaled_gold_candidate(
            env,
            candidate_code=candidate,
            scaled_oracle_cases=scaled_oracle_cases,
            hidden_tests=[],
            semantic_test_specs=semantic_test_specs,
            output_constraint_spec=current.get("output_constraint_spec_aligned") or _aligned_output_constraint_spec(output_constraint_spec, scaled_oracle_cases),
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            config=config,
        )
        current.update(evaluation)
        current["scaled_executable_gold_code"] = evaluation["scaled_executable_gold_code"]
        current["scaled_ground_truth_output_signature"] = evaluation["scaled_ground_truth_output_signature"]
        current["output_constraint_result"] = evaluation["output_constraint_result"]
        current["hidden_tests"] = []
        current["hidden_tests_mode"] = "disabled_in_case_first_stage05"
        current["hidden_tests_passed"] = evaluation["hidden_tests_passed"]
        current["validated_oracle_cases"] = evaluation.get("validated_oracle_cases", scaled_oracle_cases)
        current["scaled_gold_case_execution_report"] = evaluation.get("scaled_gold_case_execution_report", [])
        if _evaluation_passed(evaluation):
            current["failure_reasons"] = []
            current["repair_trace"] = repair_trace
            return current
        if not repair_enabled:
            break

        attempts += 1
        failure_summary = _collect_failure_summary(current, evaluation)
        failed_case_diffs = _collect_failed_case_diffs(
            scaled_oracle_cases,
            evaluation.get("scaled_gold_case_execution_report", []),
        )
        repair_rules = _derive_scaled_gold_repair_rules(failed_case_diffs)
        repair_trace.append(
            {
                "attempt": attempts,
                "failure_summary": failure_summary,
                "failed_case_diffs": failed_case_diffs,
                "repair_rules": repair_rules,
                "output_constraint_result": evaluation["output_constraint_result"],
                "case_execution_failures": evaluation.get("scaled_gold_case_execution_report", []),
                "scaled_oracle_case_failures": list(current.get("scaled_oracle_case_failures", [])),
            }
        )
        regenerated = _generate_scaled_gold_solution(
            env=env,
            operator_instances=env.operator_instances,
            semantic_test_specs=semantic_test_specs,
            scaled_oracle_cases=scaled_oracle_cases,
            hidden_tests=[],
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            previous_solution=candidate,
            previous_errors=failure_summary,
            repair_context={
                "compile_result": {
                    "compile_passed": bool(evaluation.get("compile_passed")),
                },
                "execution_result": evaluation.get("execution_result", {}),
                "visible_test_result": {
                    "status": "not_separately_executed",
                    "notes": "The current pipeline does not run a separate visible-test harness for scaled_gold before repair.",
                },
                "hidden_test_result": {
                    "status": "disabled_in_case_first_stage05",
                    "errors": [],
                },
                "output_constraint_result": evaluation.get("output_constraint_result", {}),
                "observed_output_signature": evaluation.get("scaled_ground_truth_output_signature", {}),
                "failure_summary": failure_summary,
                "failed_case_diffs": failed_case_diffs,
                "repair_rules": repair_rules,
                "fixed_scaled_oracle_cases": scaled_oracle_cases,
                "compiled_hidden_tests": [],
                "scaled_gold_case_execution_report": evaluation.get("scaled_gold_case_execution_report", []),
                "repair_mode": True,
            },
        )
        current.update(regenerated)
        current["scaled_oracle_cases"] = list(scaled_oracle_cases)
        current["validated_oracle_cases"] = list(scaled_oracle_cases)
        if not str(current.get("scaled_executable_gold_code") or "").strip():
            current["scaled_executable_gold_code"] = candidate
        current["scaled_gold_solution"] = str(current.get("scaled_executable_gold_code") or current.get("scaled_gold_solution") or "")
        current["repair_attempts"] = attempts
        current["output_constraint_spec_aligned"] = current.get("output_constraint_spec_aligned") or _aligned_output_constraint_spec(output_constraint_spec, scaled_oracle_cases)
        current["failure_reasons"] = failure_summary
    current["repair_attempts"] = attempts
    current["repair_trace"] = repair_trace
    if not current.get("failure_reasons"):
        current["failure_reasons"] = ["SCALED_GOLD_REGENERATION_FAILED"]
    return current


def run_hidden_tests_against_scaled_gold(env: ExecutableEnvSpec, scaled_executable_gold_code: str, hidden_tests: list[dict[str, Any]]) -> dict[str, Any]:
    test_env = env.model_copy(
        update={
            "gold_solution": scaled_executable_gold_code,
            "scaled_gold_solution": scaled_executable_gold_code,
            "scaled_executable_gold_code": scaled_executable_gold_code,
            "hidden_tests": hidden_tests,
        }
    )
    result = run_hidden_test_execution_check(test_env)
    return {"passed": result.status == "pass", "errors": result.errors}


def _collect_failure_summary(current: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    case_reports = [row for row in (evaluation.get("scaled_gold_case_execution_report", []) or []) if isinstance(row, dict)]
    passed_cases = [str(row.get("case_id") or "unknown_case") for row in case_reports if bool(row.get("passed"))]
    failed_case_ids = {str(row.get("case_id") or "unknown_case") for row in case_reports if not bool(row.get("passed"))}
    failures: list[str] = [f"passed_cases={passed_cases}"]
    execution_result = evaluation.get("execution_result", {}) if isinstance(evaluation.get("execution_result"), dict) else {}

    if not evaluation.get("compile_passed"):
        compile_error = str(execution_result.get("compile_error") or execution_result.get("failure_detail") or "unknown compile error")
        failures.append(f"candidate failed to compile: {compile_error}")
    elif not evaluation.get("execution_passed") and not case_reports:
        failure_reason = str(execution_result.get("failure_reason") or execution_result.get("failure_detail") or "unknown execution failure")
        failures.append(f"candidate failed before running scaled oracle cases: {failure_reason}")

    for row in case_reports:
        if not bool(row.get("passed")):
            failures.extend(_summarize_case_execution_failure(row))

    failed_checks = evaluation.get("output_constraint_result", {}).get("failed_checks", [])
    if isinstance(failed_checks, list) and not evaluation.get("case_first_mode"):
        failures.extend(_summarize_output_constraint_failures(failed_checks, failed_case_ids))

    if len(failures) == 1:
        prior = [str(item) for item in (current.get("failure_reasons", []) or []) if str(item).strip()]
        failures.extend(prior or ["SCALED_GOLD_REGENERATION_FAILED"])
    return failures


def _case_execution_python_bin(config: dict[str, Any] | None) -> str | None:
    config = config or {}
    code_execution = config.get("code_execution") or {}
    if not isinstance(code_execution, dict):
        return None
    backend = str(code_execution.get("backend") or "local").strip().lower()
    if backend == "local":
        return str(code_execution.get("local_python_bin") or code_execution.get("python_bin") or "").strip() or None
    return str(code_execution.get("python_bin") or "").strip() or None


def evaluate_scaled_gold_candidate(
    env: ExecutableEnvSpec,
    candidate_code: str,
    scaled_oracle_cases: list[dict[str, Any]],
    hidden_tests: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    output_constraint_spec: dict[str, Any],
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution_result = execute_candidate_solution(env, candidate_code)
    case_execution = run_scaled_gold_on_validated_oracle_cases(
        env,
        candidate_code,
        scaled_oracle_cases,
        python_bin=_case_execution_python_bin(config),
    )
    case_first_mode = bool(scaled_oracle_cases)
    standalone_output_constraint_result = check_output_constraints(execution_result["output_signature"], output_constraint_spec or {})
    if case_first_mode:
        output_constraint_result = {
            "passed": True,
            "passed_checks": [],
            "failed_checks": [],
            "mode": "case_first_oracle_execution",
            "standalone_output_constraint_result": standalone_output_constraint_result,
        }
    else:
        output_constraint_result = standalone_output_constraint_result
    standalone_execution_ok = bool(execution_result["compile_passed"]) and str(execution_result.get("failure_reason") or "") != "execution_failed"
    execution_passed = bool(case_execution["execution_passed"]) if case_first_mode else standalone_execution_ok
    return {
        "compile_passed": bool(execution_result["compile_passed"]),
        "execution_passed": execution_passed,
        "case_first_mode": case_first_mode,
        "execution_result": execution_result,
        "scaled_executable_gold_code": case_execution.get("materialized_code", execution_result.get("materialized_code", "")),
        "scaled_gold_solution": case_execution.get("materialized_code", execution_result.get("materialized_code", "")),
        "scaled_ground_truth_output_signature": execution_result["output_signature"],
        "output_constraint_result": output_constraint_result,
        "scaled_oracle_cases": scaled_oracle_cases,
        "validated_oracle_cases": scaled_oracle_cases,
        "hidden_tests": [],
        "hidden_tests_passed": True,
        "hidden_test_result": {"passed": True, "errors": [], "mode": "disabled_in_case_first_stage05"},
        "scaled_gold_case_execution_report": case_execution["case_reports"],
    }


def _evaluation_passed(evaluation: dict[str, Any]) -> bool:
    if evaluation.get("case_first_mode"):
        return bool(evaluation.get("compile_passed")) and bool(evaluation.get("execution_passed"))
    return (
        bool(evaluation.get("compile_passed"))
        and bool(evaluation.get("execution_passed"))
        and bool((evaluation.get("output_constraint_result") or {}).get("passed"))
    )


def _generate_scaled_oracle_cases(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    *,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if target_count <= 0:
        return [], []
    prompt = _build_scaled_oracle_cases_prompt(
        env=env,
        operator_instances=operator_instances,
        semantic_test_specs=semantic_test_specs,
        target_count=target_count,
        prompt_runner=prompt_runner,
    )
    if llm_client is not None:
        response = llm_client.complete_json(
            task_name="scaled_oracle_cases_generator",
            prompt=prompt,
            context={
                "env_id": env.env_id,
                "problem": env.problem,
                "signature": env.signature,
                "seed_execution_case": env.seed_execution_case or {},
                "seed_ground_truth_output_signature": env.seed_ground_truth_output_signature or {},
                "operator_instances": operator_instances,
                "semantic_test_specs": semantic_test_specs,
                "output_requirement_metadata": env.output_requirement_metadata or [],
                "target_count": target_count,
            },
            mock_builder=_mock_scaled_oracle_cases_builder,
        )
        payload = response.payload
    else:
        payload = _mock_scaled_oracle_cases_builder(
            {
                "seed_execution_case": env.seed_execution_case or {},
                "seed_ground_truth_output_signature": env.seed_ground_truth_output_signature or {},
                "operator_instances": operator_instances,
                "semantic_test_specs": semantic_test_specs,
                "output_requirement_metadata": env.output_requirement_metadata or [],
                "target_count": target_count,
            }
        )
    raw_cases = payload.get("scaled_oracle_cases")
    if not isinstance(raw_cases, list):
        raw_cases = payload.get("oracle_examples") if isinstance(payload.get("oracle_examples"), list) else []
    return _normalize_scaled_oracle_cases(
        raw_cases,
        target_count=target_count,
    )


def _generate_scaled_gold_solution(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    scaled_oracle_cases: list[dict[str, Any]],
    hidden_tests: list[dict[str, Any]],
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
    previous_solution: str | None = None,
    previous_errors: list[str] | None = None,
    repair_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = _build_scaled_gold_prompt(
        env=env,
        operator_instances=operator_instances,
        semantic_test_specs=semantic_test_specs,
        scaled_oracle_cases=scaled_oracle_cases,
        hidden_tests=hidden_tests,
        previous_solution=previous_solution,
        previous_errors=previous_errors or [],
        prompt_runner=prompt_runner,
        repair_context=repair_context,
    )
    if llm_client is not None:
        response = llm_client.complete_json(
            task_name="scaled_gold_solver_generator",
            prompt=prompt,
            context={
                "env_id": env.env_id,
                "problem": env.problem,
                "signature": env.signature,
                "seed_executable_code": _seed_executable_code(env),
                "operator_instances": operator_instances,
                "semantic_test_specs": semantic_test_specs,
                "scaled_oracle_cases": scaled_oracle_cases,
                "hidden_tests": hidden_tests,
                "previous_errors": previous_errors or [],
            },
            mock_builder=_mock_scaled_gold_solution_builder,
        )
        payload = response.payload
    else:
        payload = _mock_scaled_gold_solution_builder(
            {
                "seed_executable_code": _seed_executable_code(env),
                "operator_instances": operator_instances,
                "semantic_test_specs": semantic_test_specs,
                "scaled_oracle_cases": scaled_oracle_cases,
                "hidden_tests": hidden_tests,
            }
        )
    scaled_executable_gold_code = str(payload.get("scaled_executable_gold_code") or payload.get("scaled_gold_solution") or "")
    failure_reasons = []
    if scaled_executable_gold_code and not looks_like_complete_program(
        scaled_executable_gold_code,
        placeholder=env.visible_state.get("placeholder_token", "<<insert solution here>>"),
    ):
        failure_reasons.append("FULL_EXECUTABLE_CODE_REQUIRED")
    return {
        "seed_gold_solution": env.seed_gold_solution or env.gold_solution,
        "scaled_gold_solution": scaled_executable_gold_code,
        "scaled_executable_gold_code": scaled_executable_gold_code,
        "scaled_oracle_cases": list(scaled_oracle_cases or []),
        "gold_changed": bool(payload.get("gold_changed", True)),
        "answer_invariant": bool(payload.get("answer_invariant", False)),
        "gold_generation_method": str(payload.get("gold_generation_method") or "llm_solver_code_sandbox_verified"),
        "gold_change_reason": str(payload.get("gold_change_reason") or ""),
        "seed_gold_compatible_with_scaled_task": bool(payload.get("seed_gold_compatible_with_scaled_task", False)),
        "covered_operator_ids": list(payload.get("covered_operator_ids", [])),
        "covered_requirements": list(payload.get("covered_requirements", [])),
        "compile_passed": False,
        "visible_tests_passed": False,
        "hidden_tests_passed": False,
        "repair_attempts": 0,
        "failure_reasons": failure_reasons,
    }


def _compile_and_execute_candidate(env: ExecutableEnvSpec, candidate_solution: str) -> dict[str, Any]:
    materialized = insert_solution(
        context=env.context,
        solution=candidate_solution,
        placeholder=env.visible_state.get("placeholder_token", "<<insert solution here>>"),
    )
    try:
        compile(materialized, f"{env.env_id}_scaled_gold.py", "exec")
    except SyntaxError:
        return {"compile_passed": False, "execution_passed": False}
    namespace: dict[str, Any] = {"__name__": "__main__"}
    try:
        exec(materialized, namespace, namespace)
    except Exception:
        return {"compile_passed": True, "execution_passed": False}
    return {"compile_passed": True, "execution_passed": True}


def _build_scaled_gold_prompt(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    scaled_oracle_cases: list[dict[str, Any]],
    hidden_tests: list[dict[str, Any]],
    previous_solution: str | None,
    previous_errors: list[str],
    prompt_runner: PromptRunner | None = None,
    repair_context: dict[str, Any] | None = None,
) -> str:
    render_kwargs = {
        "seed_executable_code": _seed_executable_code(env),
        "seed_execution_case": _json(env.seed_execution_case or {}),
        "final_user_prompt": env.user_prompt or env.problem,
        "signature_info": env.signature or "N/A",
        "operator_instances": _json(operator_instances),
        "scaled_oracle_cases": _json(scaled_oracle_cases),
    }
    if repair_context:
        failed_case_contracts = _collect_failed_case_contracts(
            scaled_oracle_cases,
            repair_context.get("scaled_gold_case_execution_report", []),
        )
        failed_case_diffs = repair_context.get("failed_case_diffs")
        if not failed_case_diffs:
            failed_case_diffs = _collect_failed_case_diffs(
                scaled_oracle_cases,
                repair_context.get("scaled_gold_case_execution_report", []),
            )
        repair_rules = repair_context.get("repair_rules")
        if not repair_rules:
            repair_rules = _derive_scaled_gold_repair_rules(
                failed_case_diffs if isinstance(failed_case_diffs, list) else []
            )
        repair_kwargs = {
            "seed_problem": env.problem,
            "seed_executable_code": _seed_executable_code(env),
            "seed_execution_case": _json(env.seed_execution_case or {}),
            "seed_ground_truth_output_signature": _json(env.seed_ground_truth_output_signature or {}),
            "final_user_prompt": env.user_prompt or env.problem,
            "signature_info": env.signature or "N/A",
            "operator_instances": _json(operator_instances),
            "output_requirement_metadata": _json(env.output_requirement_metadata or []),
            "scaled_case_plan": _json(env.scaled_case_plan or {}),
            "scaled_oracle_cases": _json(repair_context.get("fixed_scaled_oracle_cases", [])),
            "previous_scaled_gold_solution": previous_solution or "",
            "output_constraint_result": _json(repair_context.get("output_constraint_result", {})),
            "observed_output_signature": _json(repair_context.get("observed_output_signature", {})),
            "failure_summary": _json(repair_context.get("failure_summary", previous_errors)),
            "scaled_gold_case_execution_report": _json(repair_context.get("scaled_gold_case_execution_report", [])),
            "failed_case_contracts": _json(failed_case_contracts),
            "failed_case_diffs": _json(failed_case_diffs),
            "repair_rules": _json(repair_rules),
        }
        if prompt_runner is not None:
            return prompt_runner.render("scaled_gold_repair.jinja", **repair_kwargs)
        return _build_scaled_gold_repair_prompt_fallback(**repair_kwargs)
    if prompt_runner is not None:
        return prompt_runner.render("scaled_gold_generate.jinja", **render_kwargs)
    return _build_scaled_gold_generate_prompt_fallback(**render_kwargs)


def _collect_visible_examples(env: ExecutableEnvSpec) -> list[Any]:
    visible_state = env.visible_state or {}
    task_state = env.task_state or {}
    metadata = env.metadata or {}
    for key in ["visible_examples", "examples", "sample_io", "io_examples"]:
        for source in [visible_state, task_state, metadata]:
            value = source.get(key)
            if value:
                return value if isinstance(value, list) else [value]
    return []


def _collect_expected_output_format(env: ExecutableEnvSpec) -> dict[str, Any]:
    visible_state = env.visible_state or {}
    task_state = env.task_state or {}
    return {
        "output_format": task_state.get("output_format"),
        "output_constraints": visible_state.get("output_constraints", []),
        "format_constraints": visible_state.get("format_constraints", []),
    }


def _collect_return_contract_targets(scaled_oracle_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for case in scaled_oracle_cases or []:
        if not isinstance(case, dict):
            continue
        expected = case.get("expected_output_signature")
        if not isinstance(expected, dict):
            continue
        target = {
            "case_id": str(case.get("case_id") or ""),
            "case_kind": str(case.get("case_kind") or ""),
            "return_type": str(expected.get("return_type") or "").strip() or None,
            "return_keys": list(expected.get("return_keys") or []) if isinstance(expected.get("return_keys"), list) else None,
            "has_exact_return_value": "return_value" in expected,
            "exact_return_value": expected.get("return_value") if "return_value" in expected else None,
            "stdout_contains": list(expected.get("stdout_contains") or []) if isinstance(expected.get("stdout_contains"), list) else [],
            "file_artifacts": list(expected.get("file_artifacts") or []) if isinstance(expected.get("file_artifacts"), list) else [],
        }
        targets.append(target)
    return targets


def _collect_case_contract_digest(scaled_oracle_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    digest: list[dict[str, Any]] = []
    for case in scaled_oracle_cases or []:
        if not isinstance(case, dict):
            continue
        digest.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "case_kind": str(case.get("case_kind") or ""),
                "targets_operator_id": str(case.get("targets_operator_id") or ""),
                "axis": str(case.get("axis") or ""),
                "covered_requirements": list(case.get("covered_requirements") or []),
                "call_code": str(case.get("call_code") or ""),
                "expected_output_signature": case.get("expected_output_signature") if isinstance(case.get("expected_output_signature"), dict) else {},
            }
        )
    return digest


def _collect_failed_case_contracts(
    scaled_oracle_cases: list[dict[str, Any]],
    case_execution_report: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    case_by_id = {
        str(case.get("case_id") or ""): case
        for case in (scaled_oracle_cases or [])
        if isinstance(case, dict) and str(case.get("case_id") or "").strip()
    }
    failed_contracts: list[dict[str, Any]] = []
    for report in case_execution_report or []:
        if not isinstance(report, dict) or bool(report.get("passed")):
            continue
        case_id = str(report.get("case_id") or "")
        case = case_by_id.get(case_id, {})
        failed_contracts.append(
            {
                "case_id": case_id,
                "targets_operator_id": str(case.get("targets_operator_id") or report.get("targets_operator_id") or ""),
                "axis": str(case.get("axis") or report.get("axis") or ""),
                "call_code": str(case.get("call_code") or ""),
                "covered_requirements": list(case.get("covered_requirements") or []),
                "expected_output_signature": case.get("expected_output_signature") if isinstance(case.get("expected_output_signature"), dict) else {},
                "observed_output_signature": report.get("observed_output_signature") if isinstance(report.get("observed_output_signature"), dict) else {},
                "failure_reasons": list(report.get("failure_reasons") or []),
            }
        )
    return failed_contracts


def _collect_failed_case_diffs(
    scaled_oracle_cases: list[dict[str, Any]],
    case_execution_report: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    case_by_id = {
        str(case.get("case_id") or ""): case
        for case in (scaled_oracle_cases or [])
        if isinstance(case, dict) and str(case.get("case_id") or "").strip()
    }
    diffs: list[dict[str, Any]] = []
    for report in case_execution_report or []:
        if not isinstance(report, dict) or bool(report.get("passed")):
            continue
        case_id = str(report.get("case_id") or "")
        case = case_by_id.get(case_id, {})
        expected = report.get("expected_output_signature") if isinstance(report.get("expected_output_signature"), dict) else {}
        observed = report.get("observed_output_signature") if isinstance(report.get("observed_output_signature"), dict) else {}
        failure_reasons = [str(item).strip() for item in (report.get("failure_reasons") or []) if str(item).strip()]
        diff = {
            "case_id": case_id,
            "failure_types": _classify_case_failure_types(failure_reasons),
            "failure_reasons": failure_reasons,
            "setup_code": _truncate_text(str(case.get("setup_code") or ""), 2400),
            "call_code": _truncate_text(str(case.get("call_code") or ""), 2400),
            "target_constraint": str(case.get("target_constraint") or ""),
            "covered_requirements": list(case.get("covered_requirements") or []),
            "expected": _compact_expected_signature(expected, failure_reasons),
            "observed": _compact_observed_signature(observed),
            "missing_stdout_tokens": _missing_stdout_tokens(expected, failure_reasons),
            "missing_file_artifacts": _missing_file_artifacts(expected, failure_reasons),
            "unresolved_expected_artifact_templates": _unresolved_expected_artifact_templates(expected, failure_reasons),
            "exception": _case_execution_exception(failure_reasons),
        }
        diffs.append(diff)
    return diffs


def _derive_scaled_gold_repair_rules(failed_case_diffs: list[dict[str, Any]]) -> list[str]:
    rules = [
        "Treat FAILED CASE DIFFS as the main debugging input: for each failed case, compare expected vs observed and update only the solution code.",
        "Do not edit, relax, add, or remove scaled_oracle_cases; they are fixed contracts.",
        "Preserve all passed cases while repairing failed cases.",
    ]
    failure_types = {
        failure_type
        for diff in failed_case_diffs or []
        for failure_type in (diff.get("failure_types") or [])
        if str(failure_type).strip()
    }
    if "module_not_found" in failure_types:
        rules.append(
            "ModuleNotFoundError repair: remove imports of unavailable project-local modules and implement the needed helper behavior inline with the standard library or helpers visible in the prompt."
        )
    if "name_error" in failure_types:
        rules.append(
            "NameError repair: define the missing public function/class/helper in scaled_executable_gold_code; the evaluator runs the returned file as-is and will not merge seed scaffold code."
        )
    if "stdout_mismatch" in failure_types:
        rules.append(
            "Stdout repair: print every missing stdout token from the same execution path used by the case; match spelling, punctuation, capitalization, and line breaks closely."
        )
    if "stdout_regex_mismatch" in failure_types:
        rules.append(
            "Stdout regex repair: make printed stdout match the expected pattern, including header/order/version-like formatting when required."
        )
    if "file_artifact_missing" in failure_types:
        rules.append(
            "File artifact repair: create every expected artifact path relative to the case working directory, create parent directories first, and write non-empty deterministic content when the task requires an output file."
        )
    if "unresolved_artifact_template" in failure_types:
        rules.append(
            "Template-path artifact repair: if an expected artifact path contains braces such as `{name}/file.txt`, do not ignore it; either reproduce that exact relative path or derive the concrete path from variables in setup/call code, and create parent directories."
        )
    if "return_value_mismatch" in failure_types:
        rules.append(
            "Return-value repair: make the runtime variable `result` or the called API return exactly the expected nested structure, keys, scalar values, list-vs-tuple shapes, and numeric values."
        )
    if "return_type_mismatch" in failure_types:
        rules.append(
            "Return-type repair: return the expected type exactly; do not wrap results in debug dictionaries or custom objects when the contract expects a primitive/list/dict/string."
        )
    if "return_keys_mismatch" in failure_types:
        rules.append(
            "Return-keys repair: include all required keys and remove extra wrapper keys when they prevent the expected contract from matching."
        )
    if "exception_mismatch" in failure_types:
        rules.append(
            "Exception repair: align raised exception type/message with the case contract; catch invalid input only when the expected contract requires a returned/printed error instead of propagation."
        )
    return rules


def _classify_case_failure_types(failure_reasons: list[str]) -> list[str]:
    types: list[str] = []
    for reason in failure_reasons:
        if "ModuleNotFoundError" in reason:
            types.append("module_not_found")
        if "NameError" in reason:
            types.append("name_error")
        if reason.startswith("CASE_EXECUTION_ERROR:") and "ModuleNotFoundError" not in reason and "NameError" not in reason:
            types.append("exception_mismatch")
        check_id = reason.split(":", 1)[0]
        if check_id.endswith("_return_type"):
            types.append("return_type_mismatch")
        if check_id.endswith("_return_keys"):
            types.append("return_keys_mismatch")
        if check_id.endswith("_return_value_equals") or "_return_value_contains" in check_id:
            types.append("return_value_mismatch")
        if "_stdout_contains_" in check_id:
            types.append("stdout_mismatch")
        if "_stdout_regex_" in check_id:
            types.append("stdout_regex_mismatch")
        if "_artifact_exists_" in check_id or "missing file artifact" in reason:
            types.append("file_artifact_missing")
        if "{" in reason and "}" in reason and ("artifact" in reason or "path" in reason or "file" in reason):
            types.append("unresolved_artifact_template")
    return _dedupe_strings(types)


def _compact_expected_signature(expected: dict[str, Any], failure_reasons: list[str]) -> dict[str, Any]:
    return {
        "return_type": expected.get("return_type"),
        "return_keys": expected.get("return_keys"),
        "return_value": expected.get("return_value"),
        "stdout_contains": _relevant_indexed_values(expected.get("stdout_contains"), failure_reasons, "_stdout_contains_"),
        "stdout_regex": _relevant_indexed_values(expected.get("stdout_regex"), failure_reasons, "_stdout_regex_"),
        "file_artifacts": _relevant_file_artifacts(expected.get("file_artifacts"), failure_reasons),
    }


def _compact_observed_signature(observed: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_type": observed.get("return_type"),
        "return_value": observed.get("return_value"),
        "stdout": _truncate_text(str(observed.get("stdout") or ""), 3000),
        "stderr": _truncate_text(str(observed.get("stderr") or ""), 1200),
        "file_artifacts": observed.get("file_artifacts") if isinstance(observed.get("file_artifacts"), list) else [],
    }


def _missing_stdout_tokens(expected: dict[str, Any], failure_reasons: list[str]) -> list[str]:
    tokens: list[str] = []
    for reason in failure_reasons:
        check_id = reason.split(":", 1)[0]
        if "_stdout_contains_" not in check_id:
            continue
        token = _indexed_expected_token(expected.get("stdout_contains"), check_id)
        if token:
            tokens.append(token)
    return _dedupe_strings(tokens)


def _missing_file_artifacts(expected: dict[str, Any], failure_reasons: list[str]) -> list[str]:
    paths: list[str] = []
    for reason in failure_reasons:
        check_id = reason.split(":", 1)[0]
        if "_artifact_exists_" in check_id:
            path = _indexed_expected_artifact_path(expected.get("file_artifacts"), check_id)
            if path:
                paths.append(path)
                continue
        match = re.search(r"missing file artifact:\s*([^,]+)", reason)
        if match:
            paths.append(match.group(1).strip())
    return _dedupe_strings(paths)


def _unresolved_expected_artifact_templates(expected: dict[str, Any], failure_reasons: list[str]) -> list[str]:
    paths = [
        path
        for path in _missing_file_artifacts(expected, failure_reasons)
        if "{" in path and "}" in path
    ]
    artifacts = expected.get("file_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            path = str(artifact.get("path") if isinstance(artifact, dict) else artifact or "").strip()
            if "{" in path and "}" in path:
                paths.append(path)
    return _dedupe_strings(paths)


def _case_execution_exception(failure_reasons: list[str]) -> dict[str, str]:
    for reason in failure_reasons:
        if not reason.startswith("CASE_EXECUTION_ERROR:"):
            continue
        _, exc_type, message = (reason.split(":", 2) + ["", ""])[:3]
        return {"type": exc_type, "message": message}
    return {}


def _relevant_indexed_values(values: Any, failure_reasons: list[str], marker: str) -> list[str]:
    if not isinstance(values, list):
        return []
    selected: list[str] = []
    for reason in failure_reasons:
        check_id = reason.split(":", 1)[0]
        if marker not in check_id:
            continue
        token = _indexed_expected_token(values, check_id)
        if token:
            selected.append(token)
    return _dedupe_strings(selected) if selected else [str(item) for item in values]


def _relevant_file_artifacts(values: Any, failure_reasons: list[str]) -> list[Any]:
    if not isinstance(values, list):
        return []
    missing = set(_missing_file_artifacts({"file_artifacts": values}, failure_reasons))
    if not missing:
        return values
    relevant = []
    for item in values:
        path = str(item.get("path") if isinstance(item, dict) else item or "").strip()
        if path in missing:
            relevant.append(item)
    return relevant or values


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...<truncated {len(text) - limit} chars>"


def _summarize_case_execution_failure(report: dict[str, Any]) -> list[str]:
    case_id = str(report.get("case_id") or "unknown_case")
    expected = report.get("expected_output_signature") if isinstance(report.get("expected_output_signature"), dict) else {}
    observed = report.get("observed_output_signature") if isinstance(report.get("observed_output_signature"), dict) else {}
    reasons = [str(item).strip() for item in (report.get("failure_reasons") or []) if str(item).strip()]
    if not reasons:
        return [f"{case_id} failed because its observed output did not satisfy the expected contract"]
    return [_humanize_case_failure_reason(case_id, expected, observed, reason) for reason in reasons]


def _humanize_case_failure_reason(
    case_id: str,
    expected: dict[str, Any],
    observed: dict[str, Any],
    raw_reason: str,
) -> str:
    if raw_reason.startswith("CASE_EXECUTION_ERROR:"):
        _, exc_type, message = (raw_reason.split(":", 2) + ["", ""])[:3]
        return f"{case_id} failed because code raised {exc_type}: {message}".strip()
    if raw_reason.startswith("SCALED_GOLD_COMPILE_FAILED:"):
        return f"{case_id} failed because code did not compile: {raw_reason.split(':', 1)[1]}"
    if ":" not in raw_reason:
        return f"{case_id} failed because {raw_reason}"

    check_id, detail = raw_reason.split(":", 1)
    if check_id.endswith("_return_type"):
        expected_type = str(expected.get("return_type") or "unknown")
        observed_type = str(observed.get("return_type") or "None")
        return f"{case_id} failed because expected result to be {expected_type}, got {observed_type}"
    if check_id.endswith("_return_keys"):
        expected_keys = expected.get("return_keys") if isinstance(expected.get("return_keys"), list) else []
        observed_value = observed.get("return_value")
        observed_keys = list(observed_value.keys()) if isinstance(observed_value, dict) else observed.get("return_type")
        return f"{case_id} failed because expected return keys {expected_keys}, got {observed_keys}"
    if check_id.endswith("_return_value_equals"):
        return (
            f"{case_id} failed because expected return_value "
            f"{repr(expected.get('return_value'))}, got {repr(observed.get('return_value'))}"
        )
    if "_stdout_contains_" in check_id:
        token = _indexed_expected_token(expected.get("stdout_contains"), check_id)
        if token:
            return f"{case_id} failed because stdout missing token {token}"
        return f"{case_id} failed because {detail}"
    if "_stdout_regex_" in check_id:
        pattern = _indexed_expected_token(expected.get("stdout_regex"), check_id)
        if pattern:
            return f"{case_id} failed because stdout did not match regex {pattern}"
        return f"{case_id} failed because {detail}"
    if "_artifact_exists_" in check_id:
        path = _indexed_expected_artifact_path(expected.get("file_artifacts"), check_id)
        if path:
            return f"{case_id} failed because expected file {path} was not created"
        return f"{case_id} failed because {detail}"
    return f"{case_id} failed because {detail}"


def _summarize_output_constraint_failures(
    failed_checks: list[dict[str, Any]],
    failed_case_ids: set[str],
) -> list[str]:
    summaries: list[str] = []
    for item in failed_checks:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("check_id") or "").strip()
        reason = str(item.get("reason") or "unknown constraint failure").strip()
        if any(check_id.startswith(f"{case_id}_") for case_id in failed_case_ids if case_id):
            continue
        if check_id:
            summaries.append(f"global output constraint {check_id} failed because {reason}")
        else:
            summaries.append(f"global output constraint failed because {reason}")
    return summaries


def _indexed_expected_token(values: Any, check_id: str) -> str:
    if not isinstance(values, list):
        return ""
    index = _extract_suffix_index(check_id)
    if index is None or index <= 0 or index > len(values):
        return ""
    return str(values[index - 1]).strip()


def _indexed_expected_artifact_path(values: Any, check_id: str) -> str:
    if not isinstance(values, list):
        return ""
    index = _extract_suffix_index(check_id)
    if index is None or index <= 0 or index > len(values):
        return ""
    artifact = values[index - 1]
    if isinstance(artifact, dict):
        return str(artifact.get("path") or "").strip()
    return str(artifact).strip()


def _extract_suffix_index(check_id: str) -> int | None:
    try:
        return int(check_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _collect_case_design_targets(operator_instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for operator in operator_instances:
        if not isinstance(operator, dict):
            continue
        operator_id = str(operator.get("operator_id") or "").strip()
        axis = str(operator.get("axis") or "").strip()
        output_requirements = [str(item).strip() for item in (operator.get("output_requirements") or []) if str(item).strip()]
        state_updates = operator.get("state_updates") or {}
        visible = state_updates.get("visible_state_patch") or {}
        task = state_updates.get("task_state_patch") or {}
        visible_requirements: list[str] = []
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
                visible_requirements.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value or "").strip():
                visible_requirements.append(str(value).strip())
        for key in ["extra_constraints", "required_steps", "execution_steps", "implicit_requirements", "safety_critical_constraints"]:
            value = task.get(key)
            if isinstance(value, list):
                visible_requirements.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value or "").strip():
                visible_requirements.append(str(value).strip())
        semantic_targets = [str(spec.get("target_constraint") or "").strip() for spec in (operator.get("semantic_test_specs") or []) if str(spec.get("target_constraint") or "").strip()]
        output_checks = []
        spec = operator.get("output_constraint_spec") or {}
        if isinstance(spec, dict):
            for check in spec.get("checks", []) or []:
                if isinstance(check, dict):
                    output_checks.append(
                        {
                            "check_id": str(check.get("check_id") or ""),
                            "kind": str(check.get("kind") or ""),
                            "rule": str(check.get("rule") or ""),
                            "params": check.get("params") if isinstance(check.get("params"), dict) else {},
                        }
                    )
        targets.append(
            {
                "operator_id": operator_id,
                "axis": axis,
                "transformation_goal": str(operator.get("transformation_goal") or ""),
                "output_requirements": output_requirements,
                "visible_requirements": _dedupe_strings(visible_requirements),
                "semantic_targets": _dedupe_strings(semantic_targets),
                "output_checks": output_checks,
            }
        )
    return targets


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _seed_expected_output_signature(env: ExecutableEnvSpec) -> dict[str, Any]:
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    expected = seed_case.get("expected_output_signature")
    if isinstance(expected, dict) and expected:
        return stabilize_expected_output_signature(dict(expected))
    ground_truth = env.seed_ground_truth_output_signature if isinstance(env.seed_ground_truth_output_signature, dict) else {}
    stdout = str(ground_truth.get("stdout") or "")
    stdout_contains = [line.strip() for line in stdout.splitlines() if line.strip()]
    file_artifacts = [
        {"path": str(item.get("path") or "").strip()}
        for item in (ground_truth.get("file_artifacts") or [])
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    expected_signature: dict[str, Any] = {}
    if "return_value" in ground_truth:
        expected_signature["return_value"] = ground_truth.get("return_value")
    if stdout_contains:
        expected_signature["stdout_contains"] = stdout_contains
    if file_artifacts:
        expected_signature["file_artifacts"] = file_artifacts
    return stabilize_expected_output_signature(expected_signature)


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _build_scaled_gold_generate_prompt_fallback(**kwargs: str) -> str:
    return (
        "You are an expert Python reference-solution generator for code benchmark construction.\n\n"
        "Return JSON only.\n\n"
        f"Seed executable code:\n{kwargs['seed_executable_code']}\n\n"
        f"Seed execution case:\n{kwargs['seed_execution_case']}\n\n"
        f"Scaled final user prompt:\n{kwargs['final_user_prompt']}\n\n"
        f"Function / class signature:\n{kwargs['signature_info']}\n\n"
        f"Operator instances:\n{kwargs['operator_instances']}\n\n"
        f"Scaled oracle cases:\n{kwargs['scaled_oracle_cases']}\n\n"
        "scaled_executable_gold_code must be a complete runnable Python file/program. "
        "Do not return a function body, replacement snippet, patch, diff, or code intended for placeholder insertion. "
        "The evaluator executes it exactly as returned and will not merge it into the seed scaffold.\n\n"
        "Scaled requirements are additive unless an operator explicitly replaces seed behavior. "
        "If scaled_oracle_cases include case_kind=seed_regression, preserve that original seed behavior while implementing scaled additions.\n\n"
        "Do not return scaled_oracle_cases; they are fixed test contracts from the oracle-case stage.\n\n"
        "Return JSON with keys scaled_executable_gold_code, gold_changed, "
        "answer_invariant, gold_change_reason, seed_gold_compatible_with_scaled_task, "
        "covered_operator_ids, covered_requirements.\n"
    )


def _build_scaled_gold_repair_prompt_fallback(**kwargs: str) -> str:
    return (
        "You are an expert Python debugging assistant for benchmark oracle solutions.\n\n"
        "Return JSON only.\n\n"
        f"Seed problem:\n{kwargs.get('seed_problem', '')}\n\n"
        f"Seed executable code:\n{kwargs.get('seed_executable_code', '')}\n\n"
        f"Seed execution case:\n{kwargs.get('seed_execution_case', '')}\n\n"
        f"Seed ground truth output signature:\n{kwargs.get('seed_ground_truth_output_signature', '')}\n\n"
        f"Scaled final user prompt:\n{kwargs.get('final_user_prompt', '')}\n\n"
        f"Operator instances:\n{kwargs.get('operator_instances', '')}\n\n"
        f"Output requirement metadata:\n{kwargs.get('output_requirement_metadata', '')}\n\n"
        f"Scaled case plan:\n{kwargs.get('scaled_case_plan', '')}\n\n"
        f"Failed case contracts:\n{kwargs['failed_case_contracts']}\n\n"
        f"Failed case observed-vs-expected diffs:\n{kwargs['failed_case_diffs']}\n\n"
        f"Targeted repair rules:\n{kwargs['repair_rules']}\n\n"
        f"Scaled gold case execution report:\n{kwargs['scaled_gold_case_execution_report']}\n\n"
        f"Observed output signature:\n{kwargs['observed_output_signature']}\n\n"
        f"Failure summary:\n{kwargs['failure_summary']}\n\n"
        f"Function / class signature:\n{kwargs['signature_info']}\n\n"
        f"Scaled oracle cases:\n{kwargs['scaled_oracle_cases']}\n\n"
        f"Previous scaled gold solution:\n{kwargs['previous_scaled_gold_solution']}\n\n"
        f"Output constraint result:\n{kwargs['output_constraint_result']}\n\n"
        "scaled_executable_gold_code must be a complete runnable Python file/program. "
        "Do not return a function body, replacement snippet, patch, diff, or code intended for placeholder insertion. "
        "The evaluator executes it exactly as returned and will not merge it into the seed scaffold.\n\n"
        "Repair priority order: first restore every seed_regression behavior, including return value, stdout/stderr, "
        "and file artifacts; then satisfy scaled_seed_case_main without breaking seed_regression; then satisfy "
        "coverage-extension and operator-specific cases; preserve all currently passed cases throughout the repair. "
        "Use the final prompt, operator instances, and scaled case plan to implement general logic, not one-off branches.\n\n"
        "Use failed case diffs and targeted repair rules as mandatory debugging instructions. "
        "If a failed case has case_kind=seed_regression, restore original seed behavior first while keeping scaled cases that already pass. "
        "For ModuleNotFoundError, inline missing helpers instead of importing unavailable modules. "
        "For missing file artifacts, create the exact expected relative paths and parent directories. "
        "For stdout mismatch, print the missing expected tokens from the evaluated code path.\n\n"
        "Do not return scaled_oracle_cases; they are fixed test contracts from the oracle-case stage.\n\n"
        "Return JSON with keys scaled_executable_gold_code, repair_summary, "
        "covered_operator_ids, covered_requirements, remaining_risks.\n"
    )


def _build_scaled_oracle_cases_prompt(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    semantic_test_specs: list[dict[str, Any]],
    *,
    target_count: int,
    prompt_runner: PromptRunner | None = None,
) -> str:
    case_design_targets = _collect_case_design_targets(operator_instances)
    render_kwargs = {
        "seed_problem": env.problem,
        "seed_executable_code": _seed_executable_code(env),
        "seed_execution_case": _json(env.seed_execution_case or {}),
        "seed_ground_truth_output_signature": _json(env.seed_ground_truth_output_signature or {}),
        "final_user_prompt": env.user_prompt or env.problem,
        "signature_info": env.signature or "N/A",
        "operator_instances": _json(operator_instances),
        "semantic_test_specs": _json(semantic_test_specs),
        "output_requirements": _json(env.output_requirements or []),
        "output_requirement_metadata": _json(env.output_requirement_metadata or []),
        "output_constraint_spec": _json(env.output_constraint_spec or {}),
        "case_design_targets": _json(case_design_targets),
        "scaled_case_plan": _json(env.scaled_case_plan or {}),
        "target_count": str(target_count),
    }
    if prompt_runner is not None:
        return prompt_runner.render("scaled_oracle_examples_generate.jinja", **render_kwargs)
    return _build_scaled_oracle_cases_prompt_fallback(**render_kwargs)


def _repair_scaled_oracle_cases(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    current_cases: list[dict[str, Any]],
    validation_report: list[dict[str, Any]],
    failure_summary: list[dict[str, Any]],
    llm_client: LLMClient | None,
    prompt_runner: PromptRunner | None,
) -> list[dict[str, Any]]:
    prompt = _build_scaled_oracle_cases_repair_prompt(
        env=env,
        operator_instances=operator_instances,
        current_cases=current_cases,
        validation_report=validation_report,
        failure_summary=failure_summary,
        prompt_runner=prompt_runner,
    )
    if llm_client is not None:
        response = llm_client.complete_json(
            task_name="scaled_oracle_cases_repair",
            prompt=prompt,
            context={
                "env_id": env.env_id,
                "problem": env.problem,
                "signature": env.signature,
                "oracle_case_candidates": current_cases,
                "oracle_case_validation_report": validation_report,
                "failure_summary": failure_summary,
            },
            mock_builder=lambda context: {"scaled_oracle_cases": list(context.get("oracle_case_candidates") or [])},
        )
        payload = response.payload
    else:
        payload = {"scaled_oracle_cases": current_cases}
    raw_cases = payload.get("scaled_oracle_cases")
    return list(raw_cases) if isinstance(raw_cases, list) else []


def _build_scaled_oracle_cases_prompt_fallback(**kwargs: str) -> str:
    return (
        "You are an expert benchmark oracle-case writer for scaled coding tasks.\n\n"
        "Return JSON only.\n\n"
        f"Seed problem:\n{kwargs['seed_problem']}\n\n"
        f"Seed executable code:\n{kwargs['seed_executable_code']}\n\n"
        f"Seed execution case:\n{kwargs['seed_execution_case']}\n\n"
        f"Seed ground truth output signature:\n{kwargs['seed_ground_truth_output_signature']}\n\n"
        f"Scaled final user prompt:\n{kwargs['final_user_prompt']}\n\n"
        f"Function / class signature:\n{kwargs['signature_info']}\n\n"
        f"Operator instances:\n{kwargs['operator_instances']}\n\n"
        f"Semantic test specs:\n{kwargs['semantic_test_specs']}\n\n"
        f"Output requirements:\n{kwargs['output_requirements']}\n\n"
        f"Output requirement metadata:\n{kwargs['output_requirement_metadata']}\n\n"
        f"Output constraint spec:\n{kwargs['output_constraint_spec']}\n\n"
        f"Case design targets:\n{kwargs['case_design_targets']}\n\n"
        f"Scaled case plan:\n{kwargs['scaled_case_plan']}\n\n"
        f"Required minimum scaled oracle case count: {kwargs['target_count']}\n\n"
        "The pipeline separately adds the original seed_execution_case as case_kind=seed_regression for M2/M3/M4. "
        "Generate cases for newly added scaled requirements; do not replace or weaken the original seed behavior.\n\n"
        "Each oracle case must include covered_requirement_ids using exact IDs from output_requirement_metadata when available. "
        "Keep covered_requirements as the human-readable text for those same IDs. "
        "Every non-generic required requirement_id from output_requirement_metadata must be covered by at least one case. "
        "Do not satisfy coverage by ID only: setup_code, call_code, expected_failure_mode, or expected_output_signature "
        "must make the exact requirement text observable.\n\n"
        "Return JSON with key scaled_oracle_cases.\n"
    )


def _build_scaled_oracle_cases_repair_prompt(
    env: ExecutableEnvSpec,
    operator_instances: list[dict[str, Any]],
    current_cases: list[dict[str, Any]],
    validation_report: list[dict[str, Any]],
    failure_summary: list[dict[str, Any]],
    prompt_runner: PromptRunner | None = None,
) -> str:
    case_design_targets = _collect_case_design_targets(operator_instances)
    render_kwargs = {
        "final_user_prompt": env.user_prompt or env.problem,
        "signature_info": env.signature or "N/A",
        "seed_execution_case": _json(env.seed_execution_case or {}),
        "seed_ground_truth_output_signature": _json(env.seed_ground_truth_output_signature or {}),
        "scaling_plan": _json(env.scaling_plan or env.scaling or {}),
        "operator_instances": _json(operator_instances),
        "oracle_case_candidates": _json(current_cases),
        "oracle_case_validation_report": _json(validation_report),
        "failure_summary": _json(failure_summary),
        "output_requirement_metadata": _json(env.output_requirement_metadata or []),
        "case_design_targets": _json(case_design_targets),
        "scaled_case_plan": _json(env.scaled_case_plan or {}),
    }
    if prompt_runner is not None:
        return prompt_runner.render("scaled_oracle_cases_repair.jinja", **render_kwargs)
    return (
        "You are an expert benchmark oracle-case repair assistant for scaled coding tasks.\n\n"
        "Return JSON only.\n\n"
        f"Scaled final user prompt:\n{render_kwargs['final_user_prompt']}\n\n"
        f"Function / class signature:\n{render_kwargs['signature_info']}\n\n"
        f"Seed execution case:\n{render_kwargs['seed_execution_case']}\n\n"
        f"Seed ground truth output signature:\n{render_kwargs['seed_ground_truth_output_signature']}\n\n"
        f"Scaling plan:\n{render_kwargs['scaling_plan']}\n\n"
        f"Operator instances:\n{render_kwargs['operator_instances']}\n\n"
        f"Current oracle case candidates:\n{render_kwargs['oracle_case_candidates']}\n\n"
        f"Validation report:\n{render_kwargs['oracle_case_validation_report']}\n\n"
        f"Failure summary:\n{render_kwargs['failure_summary']}\n\n"
        f"Output requirement metadata:\n{render_kwargs['output_requirement_metadata']}\n\n"
        f"Case design targets:\n{render_kwargs['case_design_targets']}\n\n"
        f"Scaled case plan:\n{render_kwargs['scaled_case_plan']}\n\n"
        "The pipeline separately adds the original seed_execution_case as case_kind=seed_regression for M2/M3/M4. "
        "Repair generated scaled cases for newly added requirements; do not replace or weaken the original seed behavior.\n\n"
        "Repair rules for concrete requirements:\n"
        "- If CASE_REQUIREMENT_TOO_GENERIC appears, rewrite target_constraint, semantic_intent, description, "
        "and covered_requirements to name the exact changed input shape, field/key/format/value, boundary condition, "
        "ordering rule, error behavior, or output contract.\n"
        "- If CASE_REQUIREMENT_NOT_LINKED_TO_OPERATOR_SEMANTICS appears, align the case text with the target "
        "operator transformation_goal/state_updates/output_requirements.\n"
        "- If CASE_EXPECTATION_NOT_LINKED_TO_REQUIREMENT appears, update setup_code, call_code, expected_failure_mode, "
        "or expected_output_signature so the observable evidence exercises the same concrete requirement.\n\n"
        "Use covered_requirement_ids with exact IDs from output_requirement_metadata whenever possible. "
        "Do not invent requirement IDs. If failure_summary contains missing_requirement_id, "
        "MISSING_REQUIREMENT_COVERAGE, REQUIREMENT_ID_HAS_WEAK_CASE_ONLY, or QUALITY_GATE_MISSING_REQUIREMENT_COVERAGE, "
        "add or repair a case for that exact requirement_id. The repaired case must make the requirement observable in "
        "setup_code, call_code, expected_failure_mode, or expected_output_signature; ID-only coverage is invalid.\n\n"
        "Return JSON with key scaled_oracle_cases.\n"
    )


def _mock_scaled_gold_solution_builder(context: dict[str, Any]) -> dict[str, Any]:
    seed_executable_code = str(context.get("seed_executable_code") or "")
    operators = context.get("operator_instances", []) or []
    semantic_axes = {str(operator.get("axis") or "") for operator in operators if str(operator.get("axis") or "") != "V"}
    answer_invariant = not semantic_axes
    covered_operator_ids = [str(operator.get("operator_id") or "") for operator in operators]
    covered_requirements = [str(spec.get("spec_id") or "") for spec in context.get("semantic_test_specs", []) or []]
    if answer_invariant:
        return {
            "scaled_executable_gold_code": seed_executable_code,
            "scaled_oracle_cases": [],
            "gold_changed": False,
            "answer_invariant": True,
            "gold_change_reason": "V-axis strengthens verifier coverage only; task semantics unchanged.",
            "seed_gold_compatible_with_scaled_task": True,
            "covered_operator_ids": covered_operator_ids,
            "covered_requirements": covered_requirements,
        }
    return {
        "scaled_executable_gold_code": seed_executable_code,
        "scaled_oracle_cases": list(context.get("scaled_oracle_cases") or []),
        "gold_changed": True,
        "answer_invariant": False,
        "gold_change_reason": "Semantic-changing operators require a regenerated scaled gold solution.",
        "seed_gold_compatible_with_scaled_task": False,
        "covered_operator_ids": covered_operator_ids,
        "covered_requirements": covered_requirements,
    }


def _mock_scaled_oracle_cases_builder(context: dict[str, Any]) -> dict[str, Any]:
    target_count = int(context.get("target_count") or 0)
    semantic_specs = [spec for spec in (context.get("semantic_test_specs", []) or []) if isinstance(spec, dict)]
    seed_case = context.get("seed_execution_case") if isinstance(context.get("seed_execution_case"), dict) else {}
    operator_instances = [item for item in (context.get("operator_instances", []) or []) if isinstance(item, dict)]
    requirement_metadata = [item for item in (context.get("output_requirement_metadata", []) or []) if isinstance(item, dict)]
    semantic_targets = detect_semantic_change(operator_instances)
    oracle_operator_ids = _oracle_case_operator_ids(operator_instances, semantic_targets)
    combined_operator_ids = ",".join(oracle_operator_ids)
    combined_axes = ",".join(_dedupe_strings([str(op.get("axis") or "") for op in operator_instances if str(op.get("axis") or "")]))
    covered_requirements = _dedupe_strings(
        [str(spec.get("target_constraint") or spec.get("semantic_intent") or "").strip() for spec in semantic_specs if str(spec.get("target_constraint") or spec.get("semantic_intent") or "").strip()]
    )
    if not covered_requirements:
        covered_requirements = _dedupe_strings(
            [
                str(item).strip()
                for operator in operator_instances
                for item in (operator.get("output_requirements") or [])
                if str(item).strip()
            ]
        )
    cases: list[dict[str, Any]] = []
    if target_count > 0 and seed_case:
        covered_requirement_ids = _mock_requirement_ids_for_texts(requirement_metadata, covered_requirements, combined_operator_ids, combined_axes)
        cases.append(
            {
                "case_id": "scaled_seed_case_main",
                "description": "Main scaled case derived from the validated seed execution case.",
                "case_kind": "scaled_seed_case_main",
                "targets_operator_id": combined_operator_ids,
                "axis": combined_axes,
                "semantic_intent": "Preserve the seed execution path while satisfying the scaled task constraints.",
                "target_constraint": "; ".join(covered_requirements[:3]),
                "expected_failure_mode": "reuses the seed behavior without implementing the new scaled constraint",
                "setup_code": str(seed_case.get("setup_code") or ""),
                "call_code": str(seed_case.get("call_code") or "result = None"),
                "assertion_code": "",
                "expected_output_signature": dict(seed_case.get("expected_output_signature") or {}),
                "covered_requirement_ids": covered_requirement_ids,
                "covered_requirements": covered_requirements,
                "covers_requirements": covered_requirements,
            }
        )
    for index, spec in enumerate(semantic_specs, start=1):
        if len(cases) >= max(target_count, 1):
            break
        spec_id = str(spec.get("spec_id") or f"scaled_oracle_case_{index}")
        spec_requirements = [str(spec.get("target_constraint") or spec.get("semantic_intent") or spec_id)]
        covered_requirement_ids = _mock_requirement_ids_for_texts(
            requirement_metadata,
            spec_requirements,
            str(spec.get("targets_operator_id") or ""),
            str(spec.get("axis") or ""),
        )
        cases.append(
            {
                "case_id": spec_id,
                "description": str(spec.get("test_case_description") or spec.get("semantic_intent") or spec_id),
                "case_kind": "coverage_extension",
                "targets_operator_id": str(spec.get("targets_operator_id") or ""),
                "axis": str(spec.get("axis") or ""),
                "semantic_intent": str(spec.get("semantic_intent") or ""),
                "target_constraint": str(spec.get("target_constraint") or ""),
                "expected_failure_mode": str(spec.get("expected_failure_mode") or ""),
                "setup_code": str(seed_case.get("setup_code") or ""),
                "call_code": str(seed_case.get("call_code") or "result = None"),
                "assertion_code": "",
                "expected_output_signature": dict(seed_case.get("expected_output_signature") or {"return_value": None}),
                "covered_requirement_ids": covered_requirement_ids,
                "covered_requirements": spec_requirements,
                "covers_requirements": spec_requirements,
            }
        )
    return {"scaled_oracle_cases": cases}


def _mock_requirement_ids_for_texts(
    requirement_metadata: list[dict[str, Any]],
    requirements: list[str],
    targets_operator_id: str,
    axis: str,
) -> list[str]:
    target_ids = {item.strip() for item in str(targets_operator_id or "").split(",") if item.strip()}
    axes = {item.strip() for item in str(axis or "").split(",") if item.strip()}
    ids: list[str] = []
    for row in requirement_metadata:
        req_id = str(row.get("requirement_id") or "")
        if not req_id:
            continue
        row_operator = str(row.get("operator_id") or "")
        row_axis = str(row.get("axis") or "")
        if row_operator and target_ids and row_operator not in target_ids:
            continue
        if row_axis not in {"seed", "global"} and axes and row_axis not in axes:
            continue
        if any(requirements_match(str(row.get("text") or ""), requirement) for requirement in requirements):
            ids.append(req_id)
    return list(dict.fromkeys(ids))
