from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from medenvscale.agent import run_stage06_tool_agent
from medenvscale.agent.runner import agent_output_slug
from medenvscale.classify.llm_full_taxonomy_router import route_with_llm_full_taxonomy
from medenvscale.classify.routing_validator import validate_routing
from medenvscale.config import AppConfig, load_app_config, load_training_config
from medenvscale.distributed import barrier, is_main_process
from medenvscale.export.export_preference import export_preference_sample
from medenvscale.export.export_prm import export_prm_samples
from medenvscale.export.export_rlvr import export_rlvr_stub
from medenvscale.export.export_sft import export_sft_sample
from medenvscale.ingest.code_execution import validate_and_repair_code_rows
from medenvscale.ingest.load_medagentgym import load_raw_medagentgym
from medenvscale.ingest.normalize_medagentgym import normalize_row
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.axis_weight_planner import plan_axis_weights
from medenvscale.scaling.dynamic_verifiable_operator_planner import synthesize_dynamic_operator_instances
from medenvscale.scaling.generic_operator_validator import repair_operator_instances, validate_dynamic_operator_instances
from medenvscale.scaling.output_constraints import (
    normalize_output_constraint_spec,
)
from medenvscale.scaling.operator_applier import apply_operator_instances
from medenvscale.scaling.prompt_rewriter import rewrite_prompt
from medenvscale.scaling.quality_filter import build_quality_report, is_semantic_hidden_test, split_clean_and_rejected
from medenvscale.scaling.requirement_registry import build_output_requirement_metadata
from medenvscale.scaling.scaling_plan import build_scaling_plan
from medenvscale.scaling.scaled_gold_solver_generator import (
    build_scaled_case_plan,
    check_seed_case_admission,
    detect_semantic_change,
    generate_scaled_gold_solution_if_needed,
    repair_scaled_gold_solution,
)
from medenvscale.scaling.seed_case_clarifier import add_seed_behavior_requirements_to_env
from medenvscale.scaling.verifier_delta_validator import validate_verifier_delta
from medenvscale.scaling.verifier_delta_normalizer import normalize_verifier_delta
from medenvscale.rl import run_stage09_rlvr_grpo
from medenvscale.rubrics import build_requirement_rubrics
from medenvscale.schemas import (
    DifficultyProfile,
    ExecutableEnvSpec,
    MedAgentGymTask,
    QuestionPoint,
    RoutingResult,
    RubricCriterion,
)
from medenvscale.split_assignment import assign_dataset_splits, has_stage05_5_split, split_envs_by_assigned_split
from medenvscale.sft import generate_tool_sft_data
from medenvscale.train.train_sft_lora import run_train_sft
from medenvscale.utils import append_jsonl, ensure_dir, load_yaml, print_progress, read_jsonl, seeded_shuffle, slugify, stable_hash, write_jsonl
from medenvscale.validation.operator_realization_checker import check_operator_realizations
from medenvscale.validation.stage05_gate_runner import run_stage05_gates
from medenvscale.verifier.verifier_builder import build_verifier_spec
from tqdm.auto import tqdm


def get_llm_client(cfg: AppConfig, llm_mode: str | None = None) -> LLMClient:
    generation_cfg = cfg.values["generation"]
    mode = llm_mode or generation_cfg.get("llm_mode") or "mock"
    return LLMClient(
        config=cfg.llm_values,
        mode=mode,
        cache_dir=str(cfg.root / cfg.llm_values["cache"]["dir"]),
        trace_path=str(cfg.root / cfg.llm_values["trace"]["path"]),
    )


def get_agent_llm_client(
    cfg: AppConfig,
    llm_mode: str | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    local_overrides: dict[str, Any] | None = None,
) -> LLMClient:
    stage06_cfg = cfg.values.get("stage06", {}) or {}
    configured_path = stage06_cfg.get("llm_config") or "configs/agent_llm.yaml"
    agent_config_path = cfg.root / str(configured_path)
    agent_values = load_yaml(agent_config_path) if agent_config_path.exists() else cfg.llm_values
    agent_values = json.loads(json.dumps(agent_values))
    if model_path:
        agent_values.setdefault("local", {})
        agent_values["local"]["enabled"] = True
        agent_values["local"]["model_path"] = str(model_path)
    if adapter_path:
        agent_values.setdefault("local", {})
        agent_values["local"]["adapter_path"] = str(adapter_path)
    if local_overrides:
        agent_values.setdefault("local", {})
        agent_values["local"].update(local_overrides)
    local_model_path = str(((agent_values.get("local") or {}).get("model_path") or "")).strip()
    agent_model_identity = Path(local_model_path).name if local_model_path else str((agent_values.get("api") or {}).get("model") or "agent_model")
    agent_model_slug = slugify(agent_model_identity)
    if cfg.dataset_name:
        agent_values.setdefault("cache", {})
        agent_values["cache"]["dir"] = str(Path(".cache") / "agent_llm" / cfg.dataset_name / agent_model_slug)
        agent_values.setdefault("trace", {})
        agent_values["trace"]["path"] = str(
            Path("data") / cfg.dataset_name / "processed" / Path(str(agent_values["trace"].get("path", "agent_generation_trace.jsonl"))).name
        )
    mode = "local" if model_path else (llm_mode or stage06_cfg.get("llm_mode") or cfg.values["generation"].get("llm_mode") or "mock")
    return LLMClient(
        config=agent_values,
        mode=mode,
        cache_dir=str(cfg.root / agent_values["cache"]["dir"]),
        trace_path=str(cfg.root / agent_values["trace"]["path"]),
    )


def get_prompt_runner(cfg: AppConfig) -> PromptRunner:
    return PromptRunner(cfg.root / "prompts")


def raw_path(cfg: AppConfig, split: str = "train") -> Path:
    dataset_cfg = cfg.values["dataset"]
    raw_paths = dataset_cfg.get("local_raw_paths")
    if isinstance(raw_paths, dict) and raw_paths.get(split):
        return cfg.root / str(raw_paths[split])
    if split == "train" and dataset_cfg.get("local_raw_path"):
        return cfg.root / str(dataset_cfg["local_raw_path"])
    raw_dir = cfg.output_dirs["raw"]
    return raw_dir / f"{split}_tasks_raw.jsonl"


def raw_test_path(cfg: AppConfig) -> Path:
    return raw_path(cfg, "test")


def raw_rejected_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["raw"] / "prepare_rejected.jsonl"


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _dataset_slug(cfg: AppConfig) -> str:
    if cfg.dataset_name:
        return cfg.dataset_name
    return cfg.values["dataset"].get("dataset_slug") or slugify(str(cfg.values["dataset"].get("name", "dataset")), max_length=24)


def _categorize_stage05_failure_reason(reason: str) -> str:
    token = str(reason or "")
    generation_markers = (
        "NO_VALIDATED_ORACLE_CASES",
        "VALIDATED_ORACLE_CASES_TOO_FEW",
        "NO_VALID_ORACLE_CASE_FOR_OPERATOR",
        "SCALED_ORACLE_CASES_TOO_FEW",
        "SEED_CASE_ADMISSION_FAILED",
        "PARTIAL_CASE_REQUIREMENT_COVERAGE",
        "M4_COMBINED_ORACLE_CASE_MISSING",
        "STAGE05_LLM_GENERATION_ERROR",
    )
    repair_markers = (
        "CASE_FAILED:",
        "SCALED_GOLD_CASE_EXECUTION_FAILED:",
        "SCALED_GOLD_COMPILE_FAILED",
        "SCALED_GOLD_EXECUTION_FAILED",
        "SCALED_OUTPUT_CONSTRAINTS_FAILED",
        "OUTPUT_CONSTRAINT_FAILED:",
        "GOLD_CASE_EXECUTION_GATE_HARD_FAIL",
        "GOLD_CASE_EXECUTION_GATE_SOFT_FAIL",
        "EMPTY_SCALED_GOLD",
        "SCALED_GOLD_SOLUTION_MISSING",
        "STAGE05_LLM_REPAIR_ERROR",
    )
    realization_markers = (
        "ONLY_RATIONALE_SIGNAL",
        "SCALED_GOLD_DOES_NOT_MATCH_VERIFIER_SPECS",
        "OPERATOR_REALIZATION",
        "INTENSITY_",
        "D_",
        "C_",
        "A_",
        "V_",
    )
    if any(marker in token for marker in generation_markers):
        return "generation"
    if any(marker in token for marker in repair_markers):
        return "repair"
    if any(marker in token for marker in realization_markers):
        return "realization"
    return "other"


def _build_stage05_failure_breakdown(reasons: list[str]) -> dict[str, list[str]]:
    breakdown = {"generation": [], "repair": [], "realization": [], "other": []}
    for reason in reasons:
        breakdown[_categorize_stage05_failure_reason(reason)].append(reason)
    return breakdown


def _stage05_failed_gold_result(
    env: ExecutableEnvSpec,
    *,
    scaled_case_plan: dict,
    output_constraint_spec: dict,
    error: Exception,
    phase: str,
) -> dict:
    error_type = type(error).__name__
    error_text = str(error).replace("\n", " ")[:1000]
    failure_code = "STAGE05_LLM_REPAIR_ERROR" if phase == "repair" else "STAGE05_LLM_GENERATION_ERROR"
    failure_reason = f"{failure_code}:{error_type}:{error_text}"
    return {
        "seed_gold_solution": env.seed_gold_solution or env.gold_solution,
        "scaled_gold_solution": "",
        "scaled_executable_gold_code": "",
        "gold_changed": True,
        "answer_invariant": False,
        "gold_generation_method": f"failed_{phase}",
        "gold_change_reason": f"Stage05 LLM {phase} failed.",
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
        "scaled_oracle_cases": [],
        "validated_oracle_cases": [],
        "scaled_oracle_case_failures": [
            {
                "case_id": env.env_id,
                "failure_code": failure_code,
                "failure_message": error_text,
            }
        ],
        "scaled_oracle_coverage_summary": {},
        "oracle_case_validation_report": [],
        "oracle_case_rule_repair_report": [],
        "oracle_case_repair_trace": [],
        "scaled_gold_case_execution_report": [],
        "scaled_case_plan": scaled_case_plan,
        "output_constraint_spec_aligned": output_constraint_spec,
        "repair_trace": [
            {
                "phase": phase,
                "error_type": error_type,
                "error": error_text,
            }
        ],
        "repair_attempts": 0,
        "failure_reasons": [failure_reason],
    }


def _is_recoverable_stage05_llm_error(error: Exception) -> bool:
    text = str(error)
    recoverable_markers = (
        "LLM network error",
        "LLM transient HTTP error",
        "LLM returned non-JSON content",
    )
    return any(marker in text for marker in recoverable_markers)


def normalize_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "normalized_tasks.jsonl"


def routing_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "routed_tasks.jsonl"


def seed_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "seed_envs.jsonl"


def scaling_plan_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaling_plans.jsonl"


def tool_config_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "tool_configs.jsonl"


def operator_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "dynamic_operator_instances.jsonl"


def operator_instances_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "operator_instances.jsonl"


def scaled_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_envs_M1_M4.jsonl"


def scaled_raw_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_envs_raw.jsonl"


def scaled_clean_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_envs_clean.jsonl"


def scaled_split_clean_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_envs_clean_split.jsonl"


def scaled_rejected_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_envs_rejected.jsonl"


def verifier_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "verifier_specs.jsonl"


def hidden_tests_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "hidden_tests.jsonl"


def hidden_tests_clean_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "hidden_tests_clean.jsonl"


def scaled_oracle_cases_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_oracle_cases.jsonl"


def scaled_case_plans_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["interim"] / "scaled_case_plans.jsonl"


def scaled_oracle_case_validation_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "scaled_oracle_case_validation_report.jsonl"


def scaled_gold_case_execution_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "scaled_gold_case_execution_report.jsonl"


def quality_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "quality_report.jsonl"


def operator_realization_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "operator_realization_report.jsonl"


def hidden_tests_quality_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "hidden_tests_quality_report.jsonl"


def scaled_task_consistency_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "scaled_task_consistency_report.jsonl"


def artifact_admission_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "artifact_admission_report.jsonl"


def stage05_quality_report_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "stage05_quality_report.jsonl"


def stage05_failure_summary_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "stage05_failure_summary.jsonl"


def stage05_5_split_manifest_output_path(cfg: AppConfig) -> Path:
    return cfg.output_dirs["processed"] / "stage05_5_split_manifest.json"


def tool_sft_processed_dir(cfg: AppConfig, teacher_slug: str) -> Path:
    return cfg.output_dirs["processed"] / "tool_sft" / teacher_slug


def tool_sft_split_dir(cfg: AppConfig, teacher_slug: str) -> Path:
    return cfg.output_dirs["splits"] / "tool_sft" / teacher_slug


def tool_sft_trajectories_output_path(cfg: AppConfig, teacher_slug: str) -> Path:
    return tool_sft_processed_dir(cfg, teacher_slug) / "tool_sft_trajectories.jsonl"


def tool_sft_quality_report_output_path(cfg: AppConfig, teacher_slug: str) -> Path:
    return tool_sft_processed_dir(cfg, teacher_slug) / "tool_sft_quality_report.jsonl"


def tool_sft_manifest_output_path(cfg: AppConfig, teacher_slug: str) -> Path:
    return cfg.output_dirs["experiments"] / "tool_sft" / teacher_slug / "data_manifest.json"


def tool_sft_split_output_paths(cfg: AppConfig, teacher_slug: str) -> dict[str, Path]:
    split_dir = tool_sft_split_dir(cfg, teacher_slug)
    return {
        "train": split_dir / "tool_sft_train.jsonl",
        "dev": split_dir / "tool_sft_dev.jsonl",
        "test": split_dir / "tool_sft_test.jsonl",
    }


def _active_scaled_input_path(cfg: AppConfig) -> Path:
    split_path = scaled_split_clean_output_path(cfg)
    if split_path.exists():
        return split_path
    clean_path = scaled_clean_output_path(cfg)
    if clean_path.exists():
        return clean_path
    return scaled_output_path(cfg)


def stage_result_dir(cfg: AppConfig, stage_name: str) -> Path:
    target = cfg.output_dirs["result"] / stage_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def stage_resume_dir(cfg: AppConfig, stage_name: str) -> Path:
    return ensure_dir(cfg.output_dirs["result"] / ".resume" / stage_name)


def stage_checkpoint_path(cfg: AppConfig, stage_name: str) -> Path:
    return stage_resume_dir(cfg, stage_name) / "checkpoint.jsonl"


def stage_status_path(cfg: AppConfig, stage_name: str) -> Path:
    return stage_result_dir(cfg, stage_name) / "stage_status.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_stage_done_status(
    cfg: AppConfig,
    stage_name: str,
    *,
    outputs: list[Path],
    input_count: int | None = None,
    output_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    status = {
        "stage": stage_name,
        "status": "done",
        "input_count": input_count,
        "output_count": output_count,
        "outputs": [_display_path(path, cfg.root) for path in outputs],
        "metadata": metadata or {},
    }
    _write_json_atomic(stage_status_path(cfg, stage_name), status)


def _load_checkpoint_by_key(path: Path, key_name: str = "task_key") -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get(key_name) or "")
        if key:
            completed[key] = row
    return completed


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row)


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def _jsonl_count(path: Path) -> int:
    return len(read_jsonl(path))


def _same_ordered_values(left: list[str], right: list[str]) -> bool:
    return left == right


def publish_stage_results(cfg: AppConfig, stage_name: str, paths: list[Path]) -> None:
    target_dir = stage_result_dir(cfg, stage_name)
    for path in paths:
        if not path.exists():
            continue
        target = target_dir / path.name
        if path.resolve() == target.resolve():
            continue
        shutil.copy2(path, target)


def _load_models(path: Path, model_cls):
    return [model_cls.model_validate(row) for row in read_jsonl(path)]


def validate_seed_count(seed_envs: list[ExecutableEnvSpec], seed_limit: int | None) -> None:
    if seed_limit is not None and len(seed_envs) > seed_limit:
        raise ValueError(
            f"Expected at most {seed_limit} seed tasks, got {len(seed_envs)}. "
            "Check loader, filtering, sampling, or early quality filters."
        )


def _aggregate_verifier_delta(operator_instances: list) -> dict:
    aggregated = {
        "new_checks": [],
        "new_hidden_tests": [],
        "exception_tests": [],
        "numeric_tolerance_tests": [],
        "array_close_tests": [],
        "dataframe_equal_tests": [],
        "file_output_tests": [],
        "object_state_tests": [],
        "static_checks": [],
        "expected_failure_modes": [],
    }
    for op in operator_instances:
        normalized = normalize_verifier_delta(op.verifier_delta, owner_id=op.operator_id)
        for key in aggregated:
            aggregated[key].extend(normalized.get(key, []))
    return aggregated


def _collect_semantic_test_specs(
    operator_instances: list,
    global_level: str,
    original_task_id: str,
) -> list[dict]:
    specs: list[dict] = []
    semantic_changing: list[dict] = []
    for operator in operator_instances:
        dumped = operator.model_dump() if hasattr(operator, "model_dump") else dict(operator)
        op_specs = [dict(item) for item in dumped.get("semantic_test_specs", []) if isinstance(item, dict)]
        for spec in op_specs:
            spec.setdefault("targets_operator_id", dumped.get("operator_id"))
            spec.setdefault("axis", dumped.get("axis"))
        specs.extend(op_specs)
        if dumped.get("semantic_change"):
            semantic_changing.append(dumped)
    if global_level in {"M3", "M4"} and len(semantic_changing) >= 2:
        primary = semantic_changing[0]
        secondary = semantic_changing[1]
        specs.append(
            {
                "spec_id": f"{original_task_id}_{global_level.lower()}_combo_1",
                "targets_operator_id": f"{primary.get('operator_id')}+{secondary.get('operator_id')}",
                "axis": "V",
                "semantic_intent": "Check whether the solution jointly satisfies multiple interacting semantic changes.",
                "target_constraint": "The solution must satisfy the combined transformed task requirements rather than each axis in isolation.",
                "expected_failure_mode": "solution_handles_single_axis_changes_but_not_combined_semantics",
                "test_template_type": "semantic_coverage_case",
                "input_variant": {"kind": "combined_semantic_variant", "value": "multi_axis_case"},
                "expected_behavior": {"kind": "oracle_output_match", "mode": "combined_semantics"},
                "test_case_description": "Combined semantic hidden test for multiple interacting operators.",
                "materialization_status": "pending",
            }
        )
    return specs


def stage00_download(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    parallel_workers: int | None = None,
    resume: bool = False,
) -> dict:
    dataset_cfg = cfg.values["dataset"]
    train_destination = raw_path(cfg, "train")
    rejected_destination = raw_rejected_path(cfg)
    metadata_path = cfg.root / dataset_cfg["metadata_path"]
    task_files = dataset_cfg.get("task_files", {})
    split_inputs: dict[str, list[dict[str, Any]]] = {}
    files_used: list[str] = []

    for split_name, relative_path in task_files.items():
        source_path = cfg.root / str(relative_path)
        if not source_path.exists():
            continue
        rows = read_jsonl(source_path)
        if limit is not None:
            rows = rows[:limit]
        prepared_input_rows = []
        for row in rows:
            materialized = dict(row)
            materialized.setdefault("source_split", split_name)
            prepared_input_rows.append(materialized)
        split_inputs[str(split_name)] = prepared_input_rows
        files_used.append(_display_path(source_path, cfg.root))

    if not split_inputs:
        fallback_rows = read_jsonl(train_destination)
        if not fallback_rows:
            raise FileNotFoundError(
                "No MedAgentGym raw tasks found. Provide dataset.task_files or place JSONL rows at "
                f"{train_destination}."
            )
        if limit is not None:
            fallback_rows = fallback_rows[:limit]
        split_inputs = {"train": fallback_rows}

    split_outputs = {split_name: raw_path(cfg, split_name) for split_name in split_inputs}
    expected_input_count = sum(len(rows) for rows in split_inputs.values())

    if resume and _stage00_split_outputs_complete(
        metadata_path=metadata_path,
        raw_outputs=split_outputs,
        rejected_output=rejected_destination,
        expected_input_count=expected_input_count,
    ):
        print(f"Stage00 resume: outputs complete; skipping ({train_destination})")
        return _json_file(metadata_path)

    llm_client = get_llm_client(cfg, llm_mode=llm_mode) if expected_input_count else None
    prompt_runner = get_prompt_runner(cfg) if expected_input_count else None
    all_rejected_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    validation_summary = {
        "input_rows": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "direct_pass_rows": 0,
        "repaired_pass_rows": 0,
    }
    for split_name, rows in split_inputs.items():
        prepared_rows, rejected_rows, split_summary = validate_and_repair_code_rows(
            rows,
            cfg=cfg,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            parallel_workers=parallel_workers,
            resume=resume,
            checkpoint_path=stage_checkpoint_path(cfg, "00") if resume else None,
        )
        split_output = split_outputs[split_name]
        write_jsonl(split_output, prepared_rows)
        for rejected in rejected_rows:
            row = dict(rejected)
            row.setdefault("source_split", split_name)
            all_rejected_rows.append(row)
        split_summaries[split_name] = {
            **split_summary,
            "output": _display_path(split_output, cfg.root),
        }
        for key in validation_summary:
            validation_summary[key] += int(split_summary.get(key, 0) or 0)
    write_jsonl(rejected_destination, all_rejected_rows)

    metadata = {
        "dataset_name": dataset_cfg.get("name", "medagentgym"),
        "download_method": "local_jsonl_split_code_validation",
        "rows_written": validation_summary["accepted_rows"],
        **validation_summary,
        "splits_requested": list(split_inputs.keys()),
        "split_outputs": split_summaries,
        "files_used": files_used or [_display_path(train_destination, cfg.root)],
        "rejected_output": _display_path(rejected_destination, cfg.root),
    }
    _write_json_atomic(metadata_path, metadata)
    publish_stage_results(cfg, "00", [*split_outputs.values(), rejected_destination, metadata_path])
    _write_stage_done_status(
        cfg,
        "00",
        outputs=[*split_outputs.values(), rejected_destination, metadata_path],
        input_count=expected_input_count,
        output_count=validation_summary["accepted_rows"],
        metadata={"rejected_rows": validation_summary["rejected_rows"], "split_outputs": split_summaries},
    )
    return metadata


def _stage00_outputs_complete(
    *,
    metadata_path: Path,
    raw_output: Path,
    rejected_output: Path,
    expected_input_count: int,
) -> bool:
    if not metadata_path.exists() or not raw_output.exists() or not rejected_output.exists():
        return False
    try:
        metadata = _json_file(metadata_path)
        input_rows = int(metadata.get("input_rows", -1))
        accepted_rows = int(metadata.get("accepted_rows", -1))
        rejected_rows = int(metadata.get("rejected_rows", -1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if input_rows != expected_input_count:
        return False
    if accepted_rows + rejected_rows != input_rows:
        return False
    return _jsonl_count(raw_output) == accepted_rows and _jsonl_count(rejected_output) == rejected_rows


def _stage00_split_outputs_complete(
    *,
    metadata_path: Path,
    raw_outputs: dict[str, Path],
    rejected_output: Path,
    expected_input_count: int,
) -> bool:
    if not metadata_path.exists() or not rejected_output.exists():
        return False
    if any(not path.exists() for path in raw_outputs.values()):
        return False
    try:
        metadata = _json_file(metadata_path)
        input_rows = int(metadata.get("input_rows", -1))
        accepted_rows = int(metadata.get("accepted_rows", -1))
        rejected_rows = int(metadata.get("rejected_rows", -1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if input_rows != expected_input_count:
        return False
    if accepted_rows + rejected_rows != input_rows:
        return False
    return sum(_jsonl_count(path) for path in raw_outputs.values()) == accepted_rows and _jsonl_count(rejected_output) == rejected_rows


def stage01_normalize(cfg: AppConfig, limit: int | None = None, resume: bool = False) -> list[MedAgentGymTask]:
    rows = load_raw_medagentgym(raw_path(cfg), limit=limit)
    output = normalize_output_path(cfg)
    if resume and _stage01_outputs_complete(output, expected_count=len(rows)):
        print(f"Stage01 resume: outputs complete; skipping ({output})")
        return _load_models(output, MedAgentGymTask)

    completed = _load_checkpoint_by_key(stage_checkpoint_path(cfg, "01")) if resume else {}
    if completed:
        print(f"Stage01 resume: loaded {len(completed)} checkpoint rows from {stage_checkpoint_path(cfg, '01')}")
    normalized_by_index: list[tuple[int, MedAgentGymTask]] = []
    for index, row in enumerate(rows, start=1):
        key = _stage01_checkpoint_key(row, index)
        if key in completed:
            normalized_by_index.append((index, MedAgentGymTask.model_validate(completed[key]["task"])))
            continue
        task = normalize_row(row, str(row.get("source_split") or cfg.values["dataset"]["split_source"]), index)
        normalized_by_index.append((index, task))
        if resume:
            _append_checkpoint(stage_checkpoint_path(cfg, "01"), {"task_key": key, "task": task.model_dump()})
    normalized = [task for _, task in sorted(normalized_by_index, key=lambda item: item[0])]
    write_jsonl(output, [item.model_dump() for item in normalized])
    publish_stage_results(cfg, "01", [output])
    _write_stage_done_status(cfg, "01", outputs=[output], input_count=len(rows), output_count=len(normalized))
    return normalized


def _stage01_outputs_complete(output: Path, *, expected_count: int) -> bool:
    if not output.exists():
        return False
    try:
        return len(_load_models(output, MedAgentGymTask)) == expected_count
    except Exception:
        return False


def _stage01_checkpoint_key(row: dict[str, Any], index: int) -> str:
    explicit_id = row.get("task_id") or row.get("id") or row.get("instance_id") or row.get("question_id") or row.get("idx")
    source_split = row.get("source_split") or "unknown"
    if explicit_id:
        return f"{source_split}:{explicit_id}"
    return f"{source_split}:row_{index}:{stable_hash(row)}"


def stage02_route(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    parallel_workers: int | None = None,
    resume: bool = False,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> list[RoutingResult]:
    normalized = _load_models(input_path or normalize_output_path(cfg), MedAgentGymTask)
    if limit is not None:
        normalized = normalized[:limit]
    taxonomy = load_yaml(cfg.dataset_config_path("domain_taxonomy.yaml"))
    allowed_domains = list(taxonomy["domains"].keys())
    allowed_task_types = taxonomy["task_types"]
    allowed_solution_forms = taxonomy["solution_forms"]
    output = output_path or routing_output_path(cfg)
    expected_task_ids = [item.task_id for item in normalized]
    if resume and _stage02_outputs_complete(output, expected_task_ids=expected_task_ids):
        print(f"Stage02 resume: outputs complete; skipping ({output})")
        return _load_models(output, RoutingResult)

    llm_client = get_llm_client(cfg, llm_mode=llm_mode)
    prompt_runner = get_prompt_runner(cfg)
    worker_count = max(1, int(parallel_workers or 1))
    if llm_client.mode == "local" and worker_count > 1:
        print("Stage02 workers is forced to 1 for local LLM mode to avoid concurrent model.generate calls.")
        worker_count = 1
    checkpoint_path = stage_checkpoint_path(cfg, "02")
    completed = _load_checkpoint_by_key(checkpoint_path) if resume else {}
    if completed:
        print(f"Stage02 resume: loaded {len(completed)} checkpoint rows from {checkpoint_path}")
    routed_by_index: list[tuple[int, RoutingResult]] = []
    pending: list[tuple[int, MedAgentGymTask]] = []
    for index, item in enumerate(normalized):
        row = completed.get(item.task_id)
        if row:
            routed_by_index.append((index, RoutingResult.model_validate(row["routing"])))
        else:
            pending.append((index, item))
    total = len(normalized)
    if worker_count == 1:
        if total:
            print_progress(len(routed_by_index), total, label="Routing MedAgentGym")
        for index, item in pending:
            routing = _route_one_medagentgym_task(
                item=item,
                llm_client=llm_client,
                prompt_runner=prompt_runner,
                allowed_domains=allowed_domains,
                allowed_task_types=allowed_task_types,
                allowed_solution_forms=allowed_solution_forms,
                cfg=cfg,
            )
            routed_by_index.append((index, routing))
            if resume:
                _append_checkpoint(checkpoint_path, {"task_key": item.task_id, "routing": routing.model_dump()})
            if total:
                print_progress(len(routed_by_index), total, label="Routing MedAgentGym")
    else:
        progress = tqdm(total=total, initial=len(routed_by_index), desc="Routing MedAgentGym", unit="task", leave=True)
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _route_one_medagentgym_task,
                        item=item,
                        llm_client=llm_client,
                        prompt_runner=prompt_runner,
                        allowed_domains=allowed_domains,
                        allowed_task_types=allowed_task_types,
                        allowed_solution_forms=allowed_solution_forms,
                        cfg=cfg,
                    ): index
                    for index, item in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    routing = future.result()
                    routed_by_index.append((index, routing))
                    if resume:
                        _append_checkpoint(checkpoint_path, {"task_key": routing.task_id, "routing": routing.model_dump()})
                    progress.update(1)
        finally:
            progress.close()
    routed = [result for _, result in sorted(routed_by_index, key=lambda item: item[0])]
    write_jsonl(output, [item.model_dump() for item in routed])
    publish_stage_results(cfg, "02", [output])
    _write_stage_done_status(cfg, "02", outputs=[output], input_count=len(normalized), output_count=len(routed))
    return routed


def _stage02_outputs_complete(output: Path, *, expected_task_ids: list[str]) -> bool:
    if not output.exists():
        return False
    try:
        routed = _load_models(output, RoutingResult)
    except Exception:
        return False
    return _same_ordered_values([item.task_id for item in routed], expected_task_ids)


def _route_one_medagentgym_task(
    *,
    item: MedAgentGymTask,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    allowed_domains: list[str],
    allowed_task_types: list[str],
    allowed_solution_forms: list[str],
    cfg: AppConfig,
) -> RoutingResult:
    llm_payload = route_with_llm_full_taxonomy(item, llm_client, prompt_runner, allowed_domains, allowed_task_types)
    return validate_routing(
        item=item,
        routing=llm_payload,
        allowed_domains=allowed_domains,
        allowed_task_types=allowed_task_types,
        allowed_solution_forms=allowed_solution_forms,
        min_confidence=cfg.values["routing"]["min_confidence"],
        review_confidence=cfg.values["routing"]["review_confidence"],
    )


def stage03_seed(cfg: AppConfig, limit: int | None = None, resume: bool = False) -> list[ExecutableEnvSpec]:
    tasks = {item.task_id: item for item in _load_models(normalize_output_path(cfg), MedAgentGymTask)}
    routed = _load_models(routing_output_path(cfg), RoutingResult)
    if limit is not None:
        routed = routed[:limit]
    expected_task_ids = [route.task_id for route in routed]
    output = seed_output_path(cfg)
    if resume and _stage03_outputs_complete(output, expected_task_ids=expected_task_ids):
        print(f"Stage03 resume: outputs complete; skipping ({output})")
        return _load_models(output, ExecutableEnvSpec)

    checkpoint_path = stage_checkpoint_path(cfg, "03")
    completed = _load_checkpoint_by_key(checkpoint_path) if resume else {}
    if completed:
        print(f"Stage03 resume: loaded {len(completed)} checkpoint rows from {checkpoint_path}")
    seed_by_index: list[tuple[int, ExecutableEnvSpec]] = []
    for index, route in enumerate(routed):
        if route.task_id in completed:
            seed_by_index.append((index, ExecutableEnvSpec.model_validate(completed[route.task_id]["env"])))
            continue
        item = tasks[route.task_id]
        seed_env = ExecutableEnvSpec(
            env_id=f"seed_{item.task_id}",
            original_task_id=item.task_id,
            split=item.source_split,
            primary_domain=route.primary_domain,
            secondary_domains=route.secondary_domains,
            primary_task_type=route.primary_task_type,
            secondary_task_types=route.secondary_task_types,
            solution_form=route.solution_form,
            verifier_type_hint=route.verifier_type_hint,
            problem=item.problem,
            context=item.context,
            signature=item.signature,
            code=item.code,
            gold_solution=item.solution,
            seed_ground_truth_output_signature=item.ground_truth_output_signature,
            scaled_ground_truth_output_signature=item.ground_truth_output_signature,
            seed_execution_case=item.seed_execution_case,
            seed_case_audit=item.seed_case_audit,
            resource_files=item.resource_files,
            task_state={"required_capabilities": route.required_capabilities},
            data_state={"resource_manifest": item.resource_files},
            tool_state={},
            visible_state={"placeholder_token": item.placeholder_token, "context_summary": item.context_summary},
            gold_state={"operator_mode": cfg.values["generation"]["operator_mode"]},
            verifier_state={"verifier_type_hint": route.verifier_type_hint},
            test_state={},
            turn_state={},
            resource_manifest=[{"path": path} for path in item.resource_files],
            immutable_fields={"original_task_id": item.task_id, "split": item.source_split},
            metadata={"domain_concepts": route.domain_concepts},
        )
        seed_by_index.append((index, seed_env))
        if resume:
            _append_checkpoint(checkpoint_path, {"task_key": route.task_id, "env": seed_env.model_dump()})
    seed_envs = [env for _, env in sorted(seed_by_index, key=lambda item: item[0])]
    validate_seed_count(seed_envs, limit)
    write_jsonl(output, [seed.model_dump() for seed in seed_envs])
    publish_stage_results(cfg, "03", [output])
    _write_stage_done_status(cfg, "03", outputs=[output], input_count=len(routed), output_count=len(seed_envs))
    return seed_envs


def _stage03_outputs_complete(output: Path, *, expected_task_ids: list[str]) -> bool:
    if not output.exists():
        return False
    try:
        envs = _load_models(output, ExecutableEnvSpec)
    except Exception:
        return False
    return _same_ordered_values([env.original_task_id for env in envs], expected_task_ids)


def stage04_skeleton(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    resume: bool = False,
) -> list[ExecutableEnvSpec]:
    if resume and seed_output_path(cfg).exists():
        seed_envs = _load_models(seed_output_path(cfg), ExecutableEnvSpec)
        if limit is not None:
            seed_envs = seed_envs[:limit]
        print(f"Stage04 resume: seed output exists; publishing skeleton from {seed_output_path(cfg)}")
    else:
        seed_envs = stage03_seed(cfg, limit=limit, resume=resume)
    publish_stage_results(cfg, "04", [seed_output_path(cfg)])
    _write_stage_done_status(cfg, "04", outputs=[seed_output_path(cfg)], input_count=len(seed_envs), output_count=len(seed_envs))
    return seed_envs


def stage05_scale(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    sample_seed: int | None = None,
    parallel_workers: int | None = None,
    resume: bool = False,
) -> list[ExecutableEnvSpec]:
    seed_envs = _load_models(seed_output_path(cfg), ExecutableEnvSpec)
    if limit is not None:
        if sample_seed is not None:
            seed_envs = seeded_shuffle(seed_envs, sample_seed)[:limit]
        else:
            seed_envs = seed_envs[:limit]
    axis_cfg = load_yaml(cfg.dataset_config_path("task_axis_priority.yaml"))
    axis_definitions_cfg = load_yaml(
        cfg.dataset_config_path_with_fallback("axis_definitions_4axis.yaml", "axis_definitions_7axis.yaml")
    )
    budgets_cfg = load_yaml(
        cfg.dataset_config_path_with_fallback("m_level_budgets_4axis.yaml", "m_level_budgets_7axis.yaml")
    )
    fusion_cfg = load_yaml(cfg.dataset_config_path("axis_weight_fusion.yaml"))
    stage05_cfg = cfg.values.get("stage05", {}) or {}
    worker_count = max(1, int(parallel_workers or stage05_cfg.get("parallel_workers", 1) or 1))
    resume = bool(resume or stage05_cfg.get("resume", False))
    levels = list(cfg.values["generation"]["levels"])
    tasks = [
        (task_index, seed_env, level)
        for task_index, (seed_env, level) in enumerate((seed_env, level) for seed_env in seed_envs for level in levels)
    ]
    expected_env_ids = [f"env_{seed_env.original_task_id}_{level}" for _, seed_env, level in tasks]
    if resume and _stage05_outputs_complete(cfg, expected_env_ids=expected_env_ids):
        print(f"Stage05 resume: outputs complete; skipping ({scaled_output_path(cfg)})")
        return _load_models(scaled_output_path(cfg), ExecutableEnvSpec)
    existing_results_by_id = _load_stage05_checkpoint_results(cfg) if resume else {}
    existing_by_id = _load_stage05_existing_envs(cfg) if resume else {}

    scaled_envs: list[ExecutableEnvSpec] = []
    scaling_plan_rows = []
    scaled_case_plan_rows = []
    tool_config_rows = []
    operator_rows = []
    verifier_rows = []
    hidden_tests = []
    operator_realization_rows = []
    oracle_case_validation_rows = []
    scaled_gold_case_execution_rows = []
    stage05_quality_rows = []
    stage05_failure_summary_rows = []
    progress = tqdm(total=len(tasks), desc="Stage05 Scaling", unit="env", leave=True)

    try:
        results: list[dict[str, Any]] = []
        completed_keys = set(existing_results_by_id) | set(existing_by_id)
        if resume and expected_env_ids and all(env_id in completed_keys for env_id in expected_env_ids):
            tqdm.write(f"Stage05 resume: rebuilding outputs from {len(expected_env_ids)} completed checkpoint/formal envs")
            for task_index, seed_env, level in tasks:
                env_id = f"env_{seed_env.original_task_id}_{level}"
                if env_id in existing_results_by_id:
                    result = _stage05_result_from_checkpoint(task_index=task_index, row=existing_results_by_id[env_id])
                else:
                    result = _stage05_result_from_existing(task_index=task_index, env=existing_by_id[env_id], level=level)
                results.append(result)
                progress.update(1)
        else:
            llm_client = get_llm_client(cfg, llm_mode=llm_mode)
            prompt_runner = get_prompt_runner(cfg)
            if llm_client.mode == "local" and worker_count > 1:
                print("Stage05 parallel_workers is forced to 1 for local LLM mode to avoid concurrent model.generate calls.")
                worker_count = 1
            if worker_count == 1:
                for task_index, seed_env, level in tasks:
                    result = _stage05_process_task(
                        task_index=task_index,
                        seed_env=seed_env,
                        level=level,
                        cfg=cfg,
                        axis_cfg=axis_cfg,
                        axis_definitions_cfg=axis_definitions_cfg,
                        budgets_cfg=budgets_cfg,
                        fusion_cfg=fusion_cfg,
                        stage05_cfg=stage05_cfg,
                        llm_client=llm_client,
                        prompt_runner=prompt_runner,
                        existing_by_id=existing_by_id,
                        existing_results_by_id=existing_results_by_id,
                    )
                    results.append(result)
                    if resume and not result.get("reused"):
                        _append_stage05_checkpoint(cfg, result)
                    progress.update(1)
            else:
                tqdm.write(f"Stage05 parallel workers: {worker_count}; resume={'on' if resume else 'off'}")
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(
                            _stage05_process_task,
                            task_index=task_index,
                            seed_env=seed_env,
                            level=level,
                            cfg=cfg,
                            axis_cfg=axis_cfg,
                            axis_definitions_cfg=axis_definitions_cfg,
                            budgets_cfg=budgets_cfg,
                            fusion_cfg=fusion_cfg,
                            stage05_cfg=stage05_cfg,
                            llm_client=llm_client,
                            prompt_runner=prompt_runner,
                            existing_by_id=existing_by_id,
                            existing_results_by_id=existing_results_by_id,
                        )
                        for task_index, seed_env, level in tasks
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        if resume and not result.get("reused"):
                            _append_stage05_checkpoint(cfg, result)
                        progress.update(1)
        for result in sorted(results, key=lambda item: int(item["task_index"])):
            _collect_stage05_result(
                result=result,
                scaled_envs=scaled_envs,
                scaling_plan_rows=scaling_plan_rows,
                scaled_case_plan_rows=scaled_case_plan_rows,
                operator_rows=operator_rows,
                verifier_rows=verifier_rows,
                hidden_tests=hidden_tests,
                operator_realization_rows=operator_realization_rows,
                oracle_case_validation_rows=oracle_case_validation_rows,
                scaled_gold_case_execution_rows=scaled_gold_case_execution_rows,
                stage05_quality_rows=stage05_quality_rows,
                stage05_failure_summary_rows=stage05_failure_summary_rows,
            )
    finally:
        progress.close()

    expected_scaled_count = len(seed_envs) * len(levels)
    if len(scaled_envs) != expected_scaled_count:
        raise ValueError(
            f"Expected {expected_scaled_count} scaled envs from {len(seed_envs)} seed tasks and {len(levels)} levels, "
            f"got {len(scaled_envs)}."
        )

    clean_envs, rejected_envs = split_clean_and_rejected(scaled_envs)
    clean_hidden_tests = []
    for env in clean_envs:
        if env.hidden_tests_mode == "disabled_in_case_first_stage05":
            continue
        for test in env.hidden_tests:
            if isinstance(test, dict) and is_semantic_hidden_test(test):
                clean_hidden_tests.append({"env_id": env.env_id, **test})
    quality_report = build_quality_report(clean_envs + rejected_envs)

    write_jsonl(scaling_plan_output_path(cfg), scaling_plan_rows)
    write_jsonl(tool_config_output_path(cfg), tool_config_rows)
    write_jsonl(operator_output_path(cfg), operator_rows)
    write_jsonl(operator_instances_output_path(cfg), operator_rows)
    write_jsonl(verifier_output_path(cfg), verifier_rows)
    write_jsonl(hidden_tests_output_path(cfg), hidden_tests)
    write_jsonl(hidden_tests_clean_output_path(cfg), clean_hidden_tests)
    write_jsonl(
        scaled_oracle_cases_output_path(cfg),
        [{"env_id": env.env_id, **case} for env in scaled_envs for case in (env.scaled_oracle_cases or []) if isinstance(case, dict)],
    )
    write_jsonl(scaled_case_plans_output_path(cfg), scaled_case_plan_rows)
    write_jsonl(scaled_oracle_case_validation_report_output_path(cfg), oracle_case_validation_rows)
    write_jsonl(scaled_gold_case_execution_report_output_path(cfg), scaled_gold_case_execution_rows)
    write_jsonl(scaled_output_path(cfg), [env.model_dump() for env in scaled_envs])
    write_jsonl(scaled_raw_output_path(cfg), [env.model_dump() for env in scaled_envs])
    write_jsonl(scaled_clean_output_path(cfg), [env.model_dump() for env in clean_envs])
    write_jsonl(scaled_rejected_output_path(cfg), [env.model_dump() for env in rejected_envs])
    write_jsonl(quality_report_output_path(cfg), quality_report)
    write_jsonl(operator_realization_report_output_path(cfg), operator_realization_rows)
    write_jsonl(hidden_tests_quality_report_output_path(cfg), [])
    write_jsonl(scaled_task_consistency_report_output_path(cfg), [])
    write_jsonl(artifact_admission_report_output_path(cfg), [])
    write_jsonl(stage05_quality_report_output_path(cfg), stage05_quality_rows)
    write_jsonl(stage05_failure_summary_output_path(cfg), stage05_failure_summary_rows)
    publish_stage_results(
        cfg,
        "05",
        [
            scaling_plan_output_path(cfg),
            tool_config_output_path(cfg),
            operator_output_path(cfg),
            operator_instances_output_path(cfg),
            verifier_output_path(cfg),
            hidden_tests_output_path(cfg),
            hidden_tests_clean_output_path(cfg),
            scaled_oracle_cases_output_path(cfg),
            scaled_case_plans_output_path(cfg),
            scaled_oracle_case_validation_report_output_path(cfg),
            scaled_gold_case_execution_report_output_path(cfg),
            scaled_output_path(cfg),
            scaled_raw_output_path(cfg),
            scaled_clean_output_path(cfg),
            scaled_rejected_output_path(cfg),
            quality_report_output_path(cfg),
            operator_realization_report_output_path(cfg),
            hidden_tests_quality_report_output_path(cfg),
            scaled_task_consistency_report_output_path(cfg),
            artifact_admission_report_output_path(cfg),
            stage05_quality_report_output_path(cfg),
            stage05_failure_summary_output_path(cfg),
        ],
    )
    _write_stage_done_status(
        cfg,
        "05",
        outputs=[
            scaled_output_path(cfg),
            scaled_raw_output_path(cfg),
            scaled_clean_output_path(cfg),
            scaled_rejected_output_path(cfg),
        ],
        input_count=len(tasks),
        output_count=len(scaled_envs),
        metadata={"clean_envs": len(clean_envs), "rejected_envs": len(rejected_envs)},
    )
    return scaled_envs


def _stage05_process_task(
    *,
    task_index: int,
    seed_env: ExecutableEnvSpec,
    level: str,
    cfg: AppConfig,
    axis_cfg: dict[str, Any],
    axis_definitions_cfg: dict[str, Any],
    budgets_cfg: dict[str, Any],
    fusion_cfg: dict[str, Any],
    stage05_cfg: dict[str, Any],
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    existing_by_id: dict[str, ExecutableEnvSpec],
    existing_results_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    env_id = f"env_{seed_env.original_task_id}_{level}"
    if env_id in existing_results_by_id:
        return _stage05_result_from_checkpoint(task_index=task_index, row=existing_results_by_id[env_id])
    if env_id in existing_by_id:
        return _stage05_result_from_existing(task_index=task_index, env=existing_by_id[env_id], level=level)
    return _stage05_build_new_result(
        task_index=task_index,
        seed_env=seed_env,
        level=level,
        cfg=cfg,
        axis_cfg=axis_cfg,
        axis_definitions_cfg=axis_definitions_cfg,
        budgets_cfg=budgets_cfg,
        fusion_cfg=fusion_cfg,
        stage05_cfg=stage05_cfg,
        llm_client=llm_client,
        prompt_runner=prompt_runner,
    )


def _stage05_build_new_result(
    *,
    task_index: int,
    seed_env: ExecutableEnvSpec,
    level: str,
    cfg: AppConfig,
    axis_cfg: dict[str, Any],
    axis_definitions_cfg: dict[str, Any],
    budgets_cfg: dict[str, Any],
    fusion_cfg: dict[str, Any],
    stage05_cfg: dict[str, Any],
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
) -> dict[str, Any]:
    env_id = f"env_{seed_env.original_task_id}_{level}"
    axis_weight_result, axis_weight_source, axis_weight_trace = plan_axis_weights(
        task_type=seed_env.task_type,
        secondary_task_types=seed_env.secondary_task_types,
        task_axis_priority_cfg=axis_cfg,
        problem=seed_env.problem,
        context_summary=seed_env.visible_state.get("context_summary", ""),
        signature=seed_env.signature,
        verifier_type_hint=seed_env.verifier_type_hint,
        llm_client=llm_client,
        prompt_runner=prompt_runner,
        domain=seed_env.primary_domain,
        solution_form=seed_env.solution_form,
    )
    scaling_plan = build_scaling_plan(
        env_id=env_id,
        global_level=level,
        task_type=seed_env.task_type,
        secondary_task_types=seed_env.secondary_task_types,
        domain=seed_env.domain,
        solution_form=seed_env.solution_form,
        axis_priority_cfg=axis_cfg,
        budgets_cfg=budgets_cfg,
        fusion_cfg=fusion_cfg,
        axis_weights=axis_weight_result,
        axis_weight_source=axis_weight_source,
    )
    if level == "M1":
        operator_instances = []
    else:
        operator_instances = synthesize_dynamic_operator_instances(
            env_id=env_id,
            task_id=seed_env.original_task_id,
            task_type=seed_env.task_type,
            domain=seed_env.domain,
            secondary_domains=seed_env.secondary_domains,
            solution_form=seed_env.solution_form,
            scaling_plan=scaling_plan.model_dump(),
            tool_config={},
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            seed_task={
                "task_id": seed_env.original_task_id,
                "problem": seed_env.problem,
                "context": seed_env.context,
                "signature": seed_env.signature,
            },
            base_environment=seed_env,
            domain_concepts=seed_env.metadata.get("domain_concepts", []),
            intensity_rubric=axis_definitions_cfg,
        )
        operator_instances = repair_operator_instances(operator_instances)
    quality_flags = []
    quality_flags.extend(axis_weight_trace)
    quality_flags.extend(validate_dynamic_operator_instances(operator_instances, scaling_plan.model_dump()))
    for op in operator_instances:
        quality_flags.extend(validate_verifier_delta(op))
    try:
        aggregated_verifier_delta = _aggregate_verifier_delta(operator_instances)
    except Exception as exc:
        aggregated_verifier_delta = {
            "new_checks": [],
            "new_hidden_tests": [],
            "exception_tests": [],
            "numeric_tolerance_tests": [],
            "array_close_tests": [],
            "dataframe_equal_tests": [],
            "file_output_tests": [],
            "object_state_tests": [],
            "static_checks": [],
            "expected_failure_modes": [],
        }
        quality_flags.append(f"verifier_build_failed: {exc}")

    env = seed_env.model_copy(
        update={
            "env_id": env_id,
            "action_space": ["submit_answer"],
            "execution_config": {
                "primary_domain": seed_env.primary_domain,
                "secondary_domains": [item.model_dump() for item in seed_env.secondary_domains],
            },
            "safety_gate_required": scaling_plan.require_safety_gate,
            "robust_verifier_required": scaling_plan.axis_intensity.get("V", 0) >= 2,
            "suitable_for_dpo_or_stress_test": level in {"M3", "M4"},
            "base_difficulty": {
                "global_level": "seed",
                "selected_axes": [],
                "total_intensity": 0,
            },
            "difficulty": DifficultyProfile(
                global_level=level,
                D=scaling_plan.axis_intensity["D"],
                C=scaling_plan.axis_intensity["C"],
                A=scaling_plan.axis_intensity["A"],
                V=scaling_plan.axis_intensity["V"],
                selected_axes=scaling_plan.selected_axes,
                total_intensity=scaling_plan.total_intensity,
                applied_operators=[op.operator_id for op in operator_instances],
            ),
            "tool_config": None,
            "scaling": scaling_plan.model_dump(),
            "scaling_plan": scaling_plan.model_dump(),
            "operator_mode": cfg.values["generation"]["operator_mode"],
            "verifier_delta": aggregated_verifier_delta,
            "rubrics": [],
        }
    )
    env = apply_operator_instances(env, operator_instances)
    env = add_seed_behavior_requirements_to_env(env)
    semantic_test_specs = _collect_semantic_test_specs(
        operator_instances=operator_instances,
        global_level=level,
        original_task_id=seed_env.original_task_id,
    )
    output_constraint_spec = normalize_output_constraint_spec(
        environment=env,
        operator_instances=[op.model_dump() for op in operator_instances],
    )
    output_requirements = [str(item).strip() for item in (env.output_requirements or []) if str(item).strip()]
    for op in operator_instances:
        output_requirements.extend(str(item).strip() for item in (op.output_requirements or []) if str(item).strip())
    output_requirements = list(dict.fromkeys(output_requirements))
    env = env.model_copy(update={"output_requirements": output_requirements})
    output_requirement_metadata = build_output_requirement_metadata(
        env,
        operator_instances=[op.model_dump() for op in operator_instances],
    )
    env = env.model_copy(
        update={
            "semantic_test_specs": semantic_test_specs,
            "output_requirement_metadata": output_requirement_metadata,
            "output_constraint_spec": output_constraint_spec,
            "seed_ground_truth_output_signature": seed_env.seed_ground_truth_output_signature or seed_env.scaled_ground_truth_output_signature,
        }
    )
    env = rewrite_prompt(env, scaling_plan, operator_instances, None)
    seed_case_admission = check_seed_case_admission(seed_env)
    scaled_case_plan = build_scaled_case_plan(
        env=env,
        operator_instances=[op.model_dump() for op in operator_instances],
        semantic_test_specs=semantic_test_specs,
    )
    env = env.model_copy(update={"scaled_case_plan": scaled_case_plan})
    if not seed_case_admission["passed"]:
        quality_flags.extend(f"SEED_CASE_ADMISSION_FAILED:{reason}" for reason in seed_case_admission["failure_reasons"])
    try:
        gold_result = generate_scaled_gold_solution_if_needed(
            env=env,
            operator_instances=[op.model_dump() for op in operator_instances],
            semantic_test_specs=semantic_test_specs,
            output_constraint_spec=output_constraint_spec,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            config={
                "max_gold_repair_attempts": 3,
                "stage05_cfg": stage05_cfg,
                "code_execution": cfg.values.get("dataset", {}).get("code_execution", {}),
            },
        )
    except RuntimeError as exc:
        if not _is_recoverable_stage05_llm_error(exc):
            raise
        tqdm.write(f"Stage05 LLM generation error on {env.env_id}: {type(exc).__name__}: {exc}")
        gold_result = _stage05_failed_gold_result(
            env,
            scaled_case_plan=scaled_case_plan,
            output_constraint_spec=output_constraint_spec,
            error=exc,
            phase="generation",
        )
    if gold_result.get("failure_reasons"):
        quality_flags.extend(str(item) for item in gold_result["failure_reasons"])
    env = env.model_copy(
        update={
            "seed_gold_solution": seed_env.gold_solution,
            "scaled_gold_solution": gold_result["scaled_gold_solution"],
            "scaled_executable_gold_code": gold_result.get("scaled_executable_gold_code") or "",
            "gold_solution": gold_result.get("scaled_executable_gold_code") or gold_result["scaled_gold_solution"],
            "scaled_oracle_cases": gold_result.get("scaled_oracle_cases", []),
            "validated_oracle_cases": gold_result.get("validated_oracle_cases", []),
            "scaled_oracle_case_failures": gold_result.get("scaled_oracle_case_failures", []),
            "scaled_oracle_coverage_summary": gold_result.get("scaled_oracle_coverage_summary", {}),
            "scaled_case_plan": gold_result.get("scaled_case_plan") or scaled_case_plan,
            "oracle_case_validation_report": gold_result.get("oracle_case_validation_report", []),
            "oracle_case_rule_repair_report": gold_result.get("oracle_case_rule_repair_report", []),
            "oracle_case_repair_trace": gold_result.get("oracle_case_repair_trace", []),
            "scaled_gold_case_execution_report": gold_result.get("scaled_gold_case_execution_report", []),
            "output_constraint_spec": gold_result.get("output_constraint_spec_aligned") or output_constraint_spec,
            "scaled_ground_truth_output_signature": gold_result.get("scaled_ground_truth_output_signature") or {},
            "gold_change_metadata": gold_result,
            "gold_state": {**(env.gold_state or {}), **{k: v for k, v in gold_result.items() if k not in {"scaled_gold_solution", "seed_gold_solution"}}},
        }
    )
    env = rewrite_prompt(env, scaling_plan, operator_instances, None)
    try:
        gold_result = repair_scaled_gold_solution(
            env=env,
            gold_result=gold_result,
            semantic_test_specs=semantic_test_specs,
            output_constraint_spec=output_constraint_spec,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
            config={
                "max_gold_repair_attempts": 3,
                "stage05_cfg": stage05_cfg,
                "code_execution": cfg.values.get("dataset", {}).get("code_execution", {}),
            },
        )
    except RuntimeError as exc:
        if not _is_recoverable_stage05_llm_error(exc):
            raise
        tqdm.write(f"Stage05 LLM repair error on {env.env_id}: {type(exc).__name__}: {exc}")
        repair_failure = _stage05_failed_gold_result(
            env,
            scaled_case_plan=scaled_case_plan,
            output_constraint_spec=output_constraint_spec,
            error=exc,
            phase="repair",
        )
        gold_result = {
            **gold_result,
            "failure_reasons": list(gold_result.get("failure_reasons") or []) + list(repair_failure["failure_reasons"]),
            "repair_trace": list(gold_result.get("repair_trace") or []) + list(repair_failure["repair_trace"]),
            "scaled_oracle_case_failures": list(gold_result.get("scaled_oracle_case_failures") or []) + list(repair_failure["scaled_oracle_case_failures"]),
        }
    env = env.model_copy(
        update={
            "scaled_gold_solution": gold_result["scaled_gold_solution"],
            "scaled_executable_gold_code": gold_result.get("scaled_executable_gold_code") or "",
            "gold_solution": gold_result.get("scaled_executable_gold_code") or gold_result["scaled_gold_solution"],
            "scaled_oracle_cases": gold_result.get("scaled_oracle_cases", []),
            "validated_oracle_cases": gold_result.get("validated_oracle_cases", []),
            "scaled_oracle_case_failures": gold_result.get("scaled_oracle_case_failures", []),
            "scaled_oracle_coverage_summary": gold_result.get("scaled_oracle_coverage_summary", {}),
            "scaled_case_plan": gold_result.get("scaled_case_plan") or scaled_case_plan,
            "oracle_case_validation_report": gold_result.get("oracle_case_validation_report", []),
            "oracle_case_rule_repair_report": gold_result.get("oracle_case_rule_repair_report", []),
            "oracle_case_repair_trace": gold_result.get("oracle_case_repair_trace", []),
            "scaled_gold_case_execution_report": gold_result.get("scaled_gold_case_execution_report", []),
            "output_constraint_spec": gold_result.get("output_constraint_spec_aligned") or output_constraint_spec,
            "scaled_ground_truth_output_signature": gold_result.get("scaled_ground_truth_output_signature") or {},
            "hidden_tests": [],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "gold_change_metadata": gold_result,
            "repair_trace": gold_result.get("repair_trace", []),
            "gold_state": {**(env.gold_state or {}), **{k: v for k, v in gold_result.items() if k not in {"scaled_gold_solution", "seed_gold_solution"}}},
        }
    )
    env = rewrite_prompt(env, scaling_plan, operator_instances, None)
    if gold_result.get("failure_reasons"):
        quality_flags.extend(str(item) for item in gold_result["failure_reasons"])
    semantic_change_info = detect_semantic_change([op.model_dump() for op in operator_instances])
    semantic_operator_ids = set(semantic_change_info["semantic_operator_ids"])
    validated_oracle_cases = list(env.validated_oracle_cases or env.scaled_oracle_cases or [])
    if semantic_operator_ids:
        covered_ops = {
            str(case.get("targets_operator_id") or "")
            for case in validated_oracle_cases
            if isinstance(case, dict) and str(case.get("targets_operator_id") or "")
        }
        for operator_id in semantic_operator_ids:
            if not any(operator_id in covered for covered in covered_ops):
                quality_flags.append(f"NO_VALID_ORACLE_CASE_FOR_OPERATOR:{operator_id}")
    if level in {"M2", "M3", "M4"} and semantic_change_info["semantic_change"] and not validated_oracle_cases:
        quality_flags.append("NO_VALIDATED_ORACLE_CASES")
    if semantic_change_info["semantic_change"] and not semantic_change_info["v_only"]:
        if level == "M4":
            combined = [item for item in validated_oracle_cases if isinstance(item, dict) and "," in str(item.get("axis") or "")]
            if len(validated_oracle_cases) > 1 and not combined:
                quality_flags.append("M4_COMBINED_ORACLE_CASE_MISSING")
    verifier_spec = build_verifier_spec(env, operator_instances, hidden_tests=[])
    requirement_rubrics = build_requirement_rubrics(env)
    env = env.model_copy(
        update={
            "verifier_spec": verifier_spec.model_dump(),
            "hidden_tests": [],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "operator_instances": [op.model_dump() for op in operator_instances],
            "rubrics": requirement_rubrics,
            "rubric_ids": [str(item.get("rubric_id") or "") for item in requirement_rubrics],
        }
    )
    env, prune_audit = _prune_redundant_invalid_oracle_cases_if_safe(
        seed_env=seed_env,
        env=env,
        budgets_cfg=budgets_cfg,
        stage05_cfg=stage05_cfg,
    )
    if prune_audit:
        dropped = ",".join(prune_audit.get("dropped_invalid_case_ids", []))
        quality_flags.append(f"DROPPED_INVALID_REDUNDANT_ORACLE_CASES:{dropped}")
    operator_realization = [] if level == "M1" else check_operator_realizations(seed_env, env)
    for report in operator_realization:
        operator_id = report["operator_id"]
        severity = report["severity"]
        if severity in {"soft_fail", "hard_fail"}:
            quality_flags.append(f"operator_realization_{severity}:{operator_id}:{','.join(report['failure_reasons'])}")
        for warning in report.get("warnings", []):
            quality_flags.append(f"operator_realization_warning:{operator_id}:{warning}")
    gate_report = run_stage05_gates(
        {
            "sample_id": env.env_id,
            "seed_task": seed_env,
            "scaled_task": env,
            "operator_realization_report": operator_realization,
        },
        config={"budgets_cfg": budgets_cfg, "stage05_cfg": stage05_cfg},
    )
    for gate_name, gate_result in gate_report["gate_results"].items():
        severity = str(gate_result.get("severity") or "")
        if severity in {"soft_fail", "hard_fail"}:
            quality_flags.append(f"stage05_gate_{severity}:{gate_name}")
        for warning in gate_result.get("warnings", []):
            quality_flags.append(f"stage05_gate_warning:{gate_name}:{warning}")
    env = env.model_copy(
        update={
            "quality_flags": quality_flags,
            "operator_realization_report": operator_realization,
            "gate_results": gate_report["gate_results"],
            "stage05_quality_report": gate_report,
            "stage05_passed": gate_report["stage05_passed"],
            "hidden_tests_mode": "disabled_in_case_first_stage05",
            "rejected_answer": "# incorrect shortcut solution\npass\n",
        }
    )
    return _stage05_result_from_parts(
        task_index=task_index,
        env=env,
        level=level,
        scaling_plan=scaling_plan.model_dump(),
        scaled_case_plan=scaled_case_plan,
        operator_rows=[{"env_id": env_id, **op.model_dump()} for op in operator_instances],
        verifier_row=verifier_spec.model_dump(),
        operator_realization_rows=operator_realization,
        stage05_quality_row=gate_report,
    )


def _prune_redundant_invalid_oracle_cases_if_safe(
    *,
    seed_env: ExecutableEnvSpec,
    env: ExecutableEnvSpec,
    budgets_cfg: dict[str, Any],
    stage05_cfg: dict[str, Any],
) -> tuple[ExecutableEnvSpec, dict[str, Any] | None]:
    validation_report = [row for row in (env.oracle_case_validation_report or []) if isinstance(row, dict)]
    invalid_rows = [row for row in validation_report if not bool(row.get("valid"))]
    if not invalid_rows:
        return env, None

    validated_cases = [case for case in (env.validated_oracle_cases or []) if isinstance(case, dict)]
    if not validated_cases:
        return env, None

    valid_case_ids = {str(case.get("case_id") or "") for case in validated_cases if str(case.get("case_id") or "")}
    pruned_validation_report = [
        row
        for row in validation_report
        if bool(row.get("valid")) and (not valid_case_ids or str(row.get("case_id") or "") in valid_case_ids)
    ]
    pruned_gold_report = [
        row
        for row in (env.scaled_gold_case_execution_report or [])
        if isinstance(row, dict) and (not valid_case_ids or str(row.get("case_id") or "") in valid_case_ids)
    ]
    dropped_case_ids = [
        str(row.get("case_id") or "")
        for row in invalid_rows
        if str(row.get("case_id") or "")
    ]
    existing_metadata = dict(env.metadata or {})
    prune_audit = {
        "strategy": "drop_redundant_invalid_oracle_cases",
        "dropped_invalid_case_ids": dropped_case_ids,
        "validated_case_ids": sorted(valid_case_ids),
    }
    candidate = env.model_copy(
        update={
            "scaled_oracle_cases": validated_cases,
            "validated_oracle_cases": validated_cases,
            "oracle_case_validation_report": pruned_validation_report,
            "scaled_oracle_case_failures": [],
            "scaled_gold_case_execution_report": pruned_gold_report,
            "metadata": {
                **existing_metadata,
                "oracle_case_pruning": prune_audit,
            },
        }
    )
    level = str((candidate.difficulty.global_level if candidate.difficulty else "") or "")
    operator_realization = [] if level == "M1" else check_operator_realizations(seed_env, candidate)
    gate_report = run_stage05_gates(
        {
            "sample_id": candidate.env_id,
            "seed_task": seed_env,
            "scaled_task": candidate,
            "operator_realization_report": operator_realization,
        },
        config={"budgets_cfg": budgets_cfg, "stage05_cfg": stage05_cfg},
    )
    if not bool(gate_report.get("stage05_passed")):
        return env, None

    prune_audit = {
        **prune_audit,
        "stage05_passed_after_prune": True,
        "final_decision_after_prune": str(gate_report.get("final_decision") or ""),
    }
    candidate = candidate.model_copy(
        update={
            "metadata": {
                **existing_metadata,
                "oracle_case_pruning": prune_audit,
            }
        }
    )
    return candidate, prune_audit


def _stage05_result_from_existing(*, task_index: int, env: ExecutableEnvSpec, level: str) -> dict[str, Any]:
    scaling_plan = env.scaling_plan or env.scaling or {}
    scaled_case_plan = env.scaled_case_plan or {}
    operator_rows = [{"env_id": env.env_id, **op} for op in (env.operator_instances or []) if isinstance(op, dict)]
    verifier_row = env.verifier_spec or {}
    stage05_quality_row = env.stage05_quality_report or {}
    operator_realization_rows = [row for row in (env.operator_realization_report or []) if isinstance(row, dict)]
    return _stage05_result_from_parts(
        task_index=task_index,
        env=env,
        level=level,
        scaling_plan=scaling_plan,
        scaled_case_plan=scaled_case_plan,
        operator_rows=operator_rows,
        verifier_row=verifier_row,
        operator_realization_rows=operator_realization_rows,
        stage05_quality_row=stage05_quality_row,
        reused=True,
    )


def _stage05_result_from_parts(
    *,
    task_index: int,
    env: ExecutableEnvSpec,
    level: str,
    scaling_plan: dict[str, Any],
    scaled_case_plan: dict[str, Any],
    operator_rows: list[dict[str, Any]],
    verifier_row: dict[str, Any],
    operator_realization_rows: list[dict[str, Any]],
    stage05_quality_row: dict[str, Any],
    reused: bool = False,
) -> dict[str, Any]:
    failure_reasons = list(stage05_quality_row.get("rejection_reasons", []) or [])
    failure_stage_breakdown = _build_stage05_failure_breakdown(failure_reasons)
    primary_failure_stage = next((stage for stage, items in failure_stage_breakdown.items() if items), "other")
    return {
        "task_index": task_index,
        "reused": reused,
        "env": env,
        "scaling_plan_row": scaling_plan,
        "scaled_case_plan_row": {
            "env_id": env.env_id,
            "original_task_id": env.original_task_id,
            "difficulty": level,
            **scaled_case_plan,
        },
        "operator_rows": operator_rows,
        "verifier_row": verifier_row,
        "hidden_tests": [
            {"env_id": env.env_id, "source": "optional_export_compatibility", **case}
            for case in (env.scaled_oracle_cases or [])
            if isinstance(case, dict)
        ],
        "operator_realization_rows": operator_realization_rows,
        "oracle_case_validation_rows": list(env.oracle_case_validation_report or []),
        "scaled_gold_case_execution_rows": list(env.scaled_gold_case_execution_report or []),
        "stage05_quality_row": stage05_quality_row,
        "stage05_failure_summary_row": {
            "env_id": env.env_id,
            "level": level,
            "failure_reasons": failure_reasons,
            "failure_stage_breakdown": failure_stage_breakdown,
            "primary_failure_stage": primary_failure_stage,
            "stage05_passed": bool(stage05_quality_row.get("stage05_passed")),
            "final_decision": str(stage05_quality_row.get("final_decision") or ""),
        },
    }


def _collect_stage05_result(
    *,
    result: dict[str, Any],
    scaled_envs: list[ExecutableEnvSpec],
    scaling_plan_rows: list[dict[str, Any]],
    scaled_case_plan_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    hidden_tests: list[dict[str, Any]],
    operator_realization_rows: list[dict[str, Any]],
    oracle_case_validation_rows: list[dict[str, Any]],
    scaled_gold_case_execution_rows: list[dict[str, Any]],
    stage05_quality_rows: list[dict[str, Any]],
    stage05_failure_summary_rows: list[dict[str, Any]],
) -> None:
    scaled_envs.append(result["env"])
    scaling_plan_rows.append(result["scaling_plan_row"])
    scaled_case_plan_rows.append(result["scaled_case_plan_row"])
    operator_rows.extend(result["operator_rows"])
    verifier_rows.append(result["verifier_row"])
    hidden_tests.extend(result["hidden_tests"])
    operator_realization_rows.extend(result["operator_realization_rows"])
    oracle_case_validation_rows.extend(result["oracle_case_validation_rows"])
    scaled_gold_case_execution_rows.extend(result["scaled_gold_case_execution_rows"])
    stage05_quality_rows.append(result["stage05_quality_row"])
    stage05_failure_summary_rows.append(result["stage05_failure_summary_row"])


def _load_stage05_existing_envs(cfg: AppConfig) -> dict[str, ExecutableEnvSpec]:
    existing: dict[str, ExecutableEnvSpec] = {}
    for row in read_jsonl(scaled_output_path(cfg)):
        env_id = str(row.get("env_id") or "")
        if not env_id:
            continue
        existing[env_id] = ExecutableEnvSpec.model_validate(row)
    if existing:
        print(f"Stage05 resume: loaded {len(existing)} existing envs from {scaled_output_path(cfg)}")
    return existing


def _stage05_outputs_complete(cfg: AppConfig, *, expected_env_ids: list[str]) -> bool:
    if not scaled_output_path(cfg).exists():
        return False
    try:
        envs = _load_models(scaled_output_path(cfg), ExecutableEnvSpec)
    except Exception:
        return False
    if not _same_ordered_values([env.env_id for env in envs], expected_env_ids):
        return False
    return scaled_clean_output_path(cfg).exists() and scaled_rejected_output_path(cfg).exists()


def _load_stage05_checkpoint_results(cfg: AppConfig) -> dict[str, dict[str, Any]]:
    path = stage_checkpoint_path(cfg, "05")
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("task_key") or "")
        result = row.get("result")
        if key and isinstance(result, dict):
            completed[key] = row
    if completed:
        print(f"Stage05 resume: loaded {len(completed)} checkpoint rows from {path}")
    return completed


def _append_stage05_checkpoint(cfg: AppConfig, result: dict[str, Any]) -> None:
    env = result["env"]
    env_id = env.env_id if isinstance(env, ExecutableEnvSpec) else str(env.get("env_id"))
    _append_checkpoint(
        stage_checkpoint_path(cfg, "05"),
        {
            "task_key": env_id,
            "result": _stage05_result_payload(result),
        },
    )


def _stage05_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    env = payload.get("env")
    if isinstance(env, ExecutableEnvSpec):
        payload["env"] = env.model_dump()
    return payload


def _stage05_result_from_checkpoint(*, task_index: int, row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("result") or {})
    payload["task_index"] = task_index
    payload["reused"] = True
    payload["env"] = ExecutableEnvSpec.model_validate(payload["env"])
    return payload


def stage05_5_assign_splits(cfg: AppConfig, resume: bool = False) -> dict[str, Any]:
    clean_path = scaled_clean_output_path(cfg)
    if not clean_path.exists():
        raise FileNotFoundError(f"Stage05 clean env file not found: {clean_path}")
    environments = _load_models(clean_path, ExecutableEnvSpec)
    split_path = scaled_split_clean_output_path(cfg)
    manifest_path = stage05_5_split_manifest_output_path(cfg)
    expected_env_ids = [env.env_id for env in environments]
    if resume and _stage05_5_outputs_complete(split_path, manifest_path, expected_env_ids=expected_env_ids):
        print(f"Stage05_5 resume: outputs complete; skipping ({split_path})")
        assigned = _load_models(split_path, ExecutableEnvSpec)
        return {"environments": assigned, "manifest": _json_file(manifest_path), "output_path": str(split_path)}

    stage_cfg = cfg.values.get("stage05_5", {}) or {}
    split_cfg = {
        "seed": stage_cfg.get("seed", (cfg.values.get("splits") or {}).get("seed", 1337)),
        "split_ratios": stage_cfg.get("split_ratios") or {
            "train": (cfg.values.get("splits") or {}).get("train", 0.7),
            "dev": (cfg.values.get("splits") or {}).get("dev", 0.1),
            "test": (cfg.values.get("splits") or {}).get("test", 0.2),
        },
    }
    checkpoint_path = stage_checkpoint_path(cfg, "05_5")
    completed = _load_checkpoint_by_key(checkpoint_path) if resume else {}
    assigned: list[ExecutableEnvSpec]
    manifest: dict[str, Any]
    if resume and completed and all(env_id in completed for env_id in expected_env_ids):
        print(f"Stage05_5 resume: rebuilding split output from {len(completed)} checkpoint rows")
        by_id = {env.env_id: env for env in environments}
        assigned = []
        fallback_order = {env_id: index for index, env_id in enumerate(expected_env_ids)}
        ordered_rows = sorted(
            (completed[env_id] for env_id in expected_env_ids),
            key=lambda row: int(row["order_index"]) if row.get("order_index") is not None else fallback_order.get(str(row.get("task_key") or ""), 0),
        )
        for row in ordered_rows:
            env_id = str(row.get("task_key") or "")
            env = by_id[env_id]
            split_name = str(row.get("assigned_split") or row.get("split") or env.split)
            assigned.append(_assign_env_split(env, split_name))
        manifest = _build_stage05_5_manifest(assigned, split_cfg)
    else:
        assigned, manifest = assign_dataset_splits(environments, split_cfg)
        if resume:
            for order_index, env in enumerate(assigned):
                _append_checkpoint(
                    checkpoint_path,
                    {
                        "task_key": env.env_id,
                        "assigned_split": env.split,
                        "order_index": order_index,
                    },
                )
    write_jsonl(split_path, [env.model_dump() for env in assigned])
    _write_json_atomic(manifest_path, manifest)

    result_dir = stage_result_dir(cfg, "05_5")
    write_jsonl(result_dir / "scaled_envs_clean.jsonl", [env.model_dump() for env in assigned])
    publish_stage_results(cfg, "05_5", [split_path, manifest_path])
    _write_stage_done_status(cfg, "05_5", outputs=[split_path, manifest_path], input_count=len(environments), output_count=len(assigned))
    return {"environments": assigned, "manifest": manifest, "output_path": str(split_path)}


def _stage05_5_outputs_complete(split_path: Path, manifest_path: Path, *, expected_env_ids: list[str]) -> bool:
    if not split_path.exists() or not manifest_path.exists():
        return False
    try:
        assigned = _load_models(split_path, ExecutableEnvSpec)
        manifest = _json_file(manifest_path)
    except Exception:
        return False
    if not _same_ordered_values([env.env_id for env in assigned], expected_env_ids):
        return False
    if len(assigned) != int(manifest.get("num_envs", -1)):
        return False
    return all(str(env.split or "") in {"train", "dev", "test"} for env in assigned)


def _assign_env_split(env: ExecutableEnvSpec, split_name: str) -> ExecutableEnvSpec:
    metadata = dict(env.metadata or {})
    metadata.update(
        {
            "dataset_split": split_name,
            "split_stage": "05_5",
            "split_group_key": env.original_task_id,
            "source_split_before_stage05_5": metadata.get("source_split_before_stage05_5", env.split),
        }
    )
    return env.model_copy(update={"split": split_name, "metadata": metadata})


def _build_stage05_5_manifest(assigned: list[ExecutableEnvSpec], split_cfg: dict[str, Any]) -> dict[str, Any]:
    env_counts = {name: 0 for name in ("train", "dev", "test")}
    for env in assigned:
        env_counts.setdefault(str(env.split), 0)
        env_counts[str(env.split)] += 1
    return {
        "split_stage": "05_5",
        "split_policy": "resume_checkpoint",
        "split_seed": int(split_cfg.get("seed", 1337)),
        "num_envs": len(assigned),
        "num_groups": len({env.original_task_id for env in assigned}),
        "env_counts": env_counts,
        "split_config": split_cfg,
    }


def stage06_tool_agent(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    retry_failed: bool = False,
    model_path: str | None = None,
    user_feedback: bool = False,
    split: str | None = None,
    resume: bool = False,
    parallel_workers: int | None = None,
) -> dict[str, list]:
    effective_split = str(split or "all").strip().lower()
    if effective_split not in {"train", "dev", "test", "all"}:
        raise ValueError(f"Unsupported Stage06 split: {effective_split!r}. Expected train, dev, test, or all.")
    all_environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    environments, split_metadata = _select_stage09_environments_from_stage05_5_split(
        environments=all_environments,
        split=effective_split,
    )
    llm_client = get_agent_llm_client(cfg, llm_mode=llm_mode, model_path=model_path)
    output_dir = stage_result_dir(cfg, "06") if effective_split == "all" else stage_result_dir(cfg, "06") / effective_split
    result = run_stage06_tool_agent(
        cfg=cfg,
        environments=environments,
        llm_client=llm_client,
        output_dir=output_dir,
        limit=limit,
        retry_failed=retry_failed,
        user_feedback=user_feedback,
        resume=resume,
        parallel_workers=parallel_workers,
    )
    result["summary"].update(
        {
            "split": effective_split,
            "split_source": split_metadata.get("split_source"),
            "split_env_count": split_metadata.get("split_env_count"),
            "output_dir": str(output_dir / agent_output_slug(llm_client)),
        }
    )
    summary_path = output_dir / agent_output_slug(llm_client) / "summary.json"
    summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def stage07_tool_sft_data(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    user_feedback: bool = False,
    resume: bool = False,
    parallel_workers: int | None = None,
) -> dict[str, Any]:
    environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    llm_client = get_agent_llm_client(cfg, llm_mode=llm_mode)
    teacher_slug = agent_output_slug(llm_client)
    split_paths = tool_sft_split_output_paths(cfg, teacher_slug)
    output_paths = {
        "trajectories": tool_sft_trajectories_output_path(cfg, teacher_slug),
        "quality_report": tool_sft_quality_report_output_path(cfg, teacher_slug),
        "manifest": tool_sft_manifest_output_path(cfg, teacher_slug),
        "split_train": split_paths["train"],
        "split_dev": split_paths["dev"],
        "split_test": split_paths["test"],
    }
    result = generate_tool_sft_data(
        cfg=cfg,
        environments=environments,
        llm_client=llm_client,
        prompt_runner=get_prompt_runner(cfg),
        output_paths=output_paths,
        limit=limit,
        user_feedback=user_feedback,
        resume=resume,
        parallel_workers=parallel_workers,
        checkpoint_path=stage_resume_dir(cfg, "07") / f"{teacher_slug}_checkpoint.jsonl" if resume else None,
    )
    publish_stage_results(
        cfg,
        f"07/{teacher_slug}",
        [
            output_paths["trajectories"],
            output_paths["quality_report"],
            output_paths["manifest"],
            split_paths["train"],
            split_paths["dev"],
            split_paths["test"],
        ],
    )
    return result


def stage07_qpoints_rubrics(cfg: AppConfig, limit: int | None = None, llm_mode: str | None = None) -> dict[str, list]:
    target_path = _active_scaled_input_path(cfg)
    environments = _load_models(target_path, ExecutableEnvSpec)
    if limit is not None:
        environments = environments[: limit * len(cfg.values["generation"]["levels"])]
    all_points: list[QuestionPoint] = []
    all_rubrics: list[RubricCriterion] = []
    sampled_trajectories: list[dict] = []
    updated_envs = []

    for env in environments:
        points = build_question_points(env)
        rubrics = build_rubrics(env, points)
        sampled_trajectories.extend(build_sampled_trajectories(env, points))
        env = env.model_copy(
            update={
                "question_point_ids": [point.point_id for point in points],
                "rubric_ids": [rubric.rubric_id for rubric in rubrics],
                "rubrics": [rubric.model_dump() for rubric in rubrics],
            }
        )
        updated_envs.append(env)
        all_points.extend(points)
        all_rubrics.extend(rubrics)

    question_points_path = cfg.output_dirs["processed"] / "question_points.jsonl"
    rubrics_path = cfg.output_dirs["processed"] / "rubrics.jsonl"
    sampled_trajectories_path = cfg.output_dirs["processed"] / "sampled_trajectories.jsonl"
    write_jsonl(question_points_path, [point.model_dump() for point in all_points])
    write_jsonl(rubrics_path, [rubric.model_dump() for rubric in all_rubrics])
    write_jsonl(sampled_trajectories_path, sampled_trajectories)
    write_jsonl(target_path, [env.model_dump() for env in updated_envs])
    publish_stage_results(cfg, "07", [target_path, question_points_path, rubrics_path, sampled_trajectories_path])
    return {"environments": updated_envs, "question_points": all_points, "rubrics": all_rubrics}


def stage08_safety(cfg: AppConfig) -> list[dict]:
    report_path = quality_report_output_path(cfg)
    if report_path.exists():
        extra_paths = [
            report_path,
            stage05_quality_report_output_path(cfg),
            hidden_tests_quality_report_output_path(cfg),
            scaled_task_consistency_report_output_path(cfg),
            artifact_admission_report_output_path(cfg),
        ]
        publish_stage_results(cfg, "08", extra_paths)
        return read_jsonl(report_path)
    environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    clean, rejected = split_clean_and_rejected(environments)
    report = build_quality_report(clean + rejected)
    write_jsonl(report_path, report)
    publish_stage_results(
        cfg,
        "08",
        [
            report_path,
            stage05_quality_report_output_path(cfg),
            hidden_tests_quality_report_output_path(cfg),
            scaled_task_consistency_report_output_path(cfg),
            artifact_admission_report_output_path(cfg),
        ],
    )
    return report


def stage08_train_sft(
    cfg: AppConfig,
    *,
    train_config: str | None = None,
    max_steps: int | None = None,
    model_name_or_path: str | None = None,
    teacher_slug: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(train_config) if train_config else cfg.dataset_config_path("train_sft.yaml")
    if not config_path.is_absolute():
        config_path = cfg.root / config_path
    config_path_for_result = str(config_path)
    train_root, train_cfg = load_training_config(config_path, dataset=cfg.dataset_name)
    if model_name_or_path:
        train_cfg["model_name_or_path"] = str(model_name_or_path)
    if teacher_slug:
        train_cfg["teacher_slug"] = str(teacher_slug)
    if max_steps is not None:
        train_cfg["max_steps"] = int(max_steps)
    output_dir = train_root / str(train_cfg.get("output_dir") or "experiments/biocoder/tool_sft_lora")
    manifest_path = output_dir / "train_manifest.json"
    if resume and manifest_path.exists():
        existing_manifest = _json_file(manifest_path)
        adapter_dir = output_dir / "adapter"
        if existing_manifest.get("status") == "completed" and (adapter_dir / "adapter_config.json").exists():
            print(f"Stage08 resume: completed adapter exists; skipping ({adapter_dir})")
            return existing_manifest
    result = run_train_sft(
        config_path_for_result,
        max_steps=max_steps,
        dataset=cfg.dataset_name,
        model_name_or_path=model_name_or_path,
        teacher_slug=teacher_slug,
        dry_run=dry_run,
        resume=resume,
    )
    prepared_paths = [Path(result["prepared_train_path"])]
    if result.get("prepared_eval_path"):
        prepared_paths.append(Path(result["prepared_eval_path"]))
    if is_main_process():
        publish_stage_results(cfg, "08", [Path(result["output_dir"]) / "train_manifest.json", *prepared_paths])
    barrier()
    return result


def stage08_5_eval_sft(
    cfg: AppConfig,
    limit: int | None = None,
    *,
    split: str = "dev",
    train_config: str | None = None,
    llm_mode: str | None = None,
    model_path: str | None = None,
    sft_adapter: str | None = None,
    retry_failed: bool = False,
    user_feedback: bool = False,
    resume: bool = False,
    parallel_workers: int | None = None,
) -> dict[str, Any]:
    config_path = Path(train_config) if train_config else cfg.dataset_config_path("train_sft.yaml")
    if not config_path.is_absolute():
        config_path = cfg.root / config_path
    train_root, train_cfg = load_training_config(config_path, dataset=cfg.dataset_name)
    base_model = str(
        model_path
        or train_cfg.get("model_name_or_path")
        or (cfg.values.get("stage09_rlvr_grpo", {}) or {}).get("base_model")
        or ""
    ).strip()
    adapter_path = str(sft_adapter or "").strip()
    if not adapter_path:
        output_dir = train_root / str(train_cfg.get("output_dir") or "experiments/biocoder/tool_sft_lora")
        adapter_path = str(output_dir / "adapter")
    if not Path(adapter_path).is_absolute():
        adapter_path = str(cfg.root / adapter_path)
    if not base_model:
        raise ValueError("Stage08_5 requires a base model path from --model_path or configs/biocoder/train_sft.yaml.")
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Stage08_5 SFT adapter not found: {adapter_path}")

    stage08_5_cfg = cfg.values.get("stage08_5_eval_sft", {}) or {}
    effective_split = str(split or stage08_5_cfg.get("split") or "dev").strip().lower()
    if effective_split not in {"train", "dev", "test", "all"}:
        raise ValueError(f"Unsupported Stage08_5 split: {effective_split!r}. Expected train, dev, test, or all.")
    all_environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    environments, split_metadata = _select_stage09_environments_from_stage05_5_split(
        environments=all_environments,
        split=effective_split,
    )
    local_overrides = {
        "do_sample": bool(stage08_5_cfg.get("do_sample", False)),
        "temperature": float(stage08_5_cfg.get("temperature", 0.0)),
        "top_p": float(stage08_5_cfg.get("top_p", 1.0)),
        "max_new_tokens": int(stage08_5_cfg.get("max_new_tokens", (cfg.values.get("stage09_rlvr_grpo", {}) or {}).get("max_new_tokens", 2048))),
        "trust_remote_code": bool(train_cfg.get("trust_remote_code", True)),
        "device_map": stage08_5_cfg.get("device_map", train_cfg.get("device_map", "auto")),
        "torch_dtype": stage08_5_cfg.get("torch_dtype", train_cfg.get("torch_dtype", "auto")),
    }
    llm_client = get_agent_llm_client(
        cfg,
        llm_mode=llm_mode or "local",
        model_path=base_model,
        adapter_path=adapter_path,
        local_overrides=local_overrides,
    )
    adapter_name = Path(adapter_path).parent.name if Path(adapter_path).name == "adapter" else Path(adapter_path).name
    eval_slug = slugify(f"{adapter_name}_{effective_split}", max_length=96)
    output_parent = stage_result_dir(cfg, "08_5") / eval_slug
    result = run_stage06_tool_agent(
        cfg=cfg,
        environments=environments,
        llm_client=llm_client,
        output_dir=output_parent,
        limit=limit,
        retry_failed=retry_failed,
        user_feedback=user_feedback,
        resume=resume,
        parallel_workers=parallel_workers,
    )
    leaf_output_dir = output_parent / agent_output_slug(llm_client)
    result["summary"].update(
        {
            "stage": "08_5_eval_sft",
            "split": effective_split,
            "split_source": split_metadata.get("split_source"),
            "split_env_count": split_metadata.get("split_env_count"),
            "base_model": base_model,
            "sft_adapter": adapter_path,
            "tool_format_version": "simplified_tool_json_v1",
            "output_dir": str(leaf_output_dir),
        }
    )
    (leaf_output_dir / "summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "stage": "08_5_eval_sft",
        "split": effective_split,
        "split_metadata": split_metadata,
        "limit": limit,
        "base_model": base_model,
        "sft_adapter": adapter_path,
        "tool_format_version": "simplified_tool_json_v1",
        "llm_mode": llm_client.mode,
        "output_dir": str(leaf_output_dir),
        "summary": result["summary"],
    }
    (leaf_output_dir / "eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_stage_results(
        cfg,
        f"08_5/{eval_slug}",
        [
            leaf_output_dir / "agent_runs.jsonl",
            leaf_output_dir / "agent_traces.jsonl",
            leaf_output_dir / "agent_eval_report.jsonl",
            leaf_output_dir / "summary.json",
            leaf_output_dir / "eval_manifest.json",
        ],
    )
    return result


def stage09_rlvr_grpo(
    cfg: AppConfig,
    limit: int | None = None,
    *,
    rollout_only: bool = True,
    train: bool = False,
    collect_rollouts: bool = False,
    split: str = "train",
    eval_split: str | None = None,
    eval_steps: int | None = None,
    llm_mode: str | None = None,
    model_path: str | None = None,
    sft_adapter: str | None = None,
    user_feedback: bool = False,
    use_rubric_reward: bool | None = None,
    resume: bool = False,
    use_existing_rollouts: bool = False,
) -> dict[str, Any]:
    stage_cfg = dict(cfg.values.get("stage09_rlvr_grpo", {}) or {})
    base_model = model_path or stage_cfg.get("base_model") or "/archive/zengjiaqi/Medical_LLM/Qwen3.5-2B-Base"
    adapter_path = sft_adapter or stage_cfg.get("sft_adapter") or "experiments/biocoder/tool_sft_lora/adapter"
    if not Path(str(adapter_path)).is_absolute():
        adapter_path = str(cfg.root / str(adapter_path))
    stage_cfg["base_model"] = str(base_model)
    stage_cfg["sft_adapter"] = str(adapter_path)
    effective_split = str(split or stage_cfg.get("split") or "train").strip().lower()
    if effective_split not in {"train", "dev", "test", "all"}:
        raise ValueError(f"Unsupported Stage09 split: {effective_split!r}. Expected train, dev, test, or all.")
    stage_cfg["split"] = effective_split
    effective_eval_split = str(eval_split or stage_cfg.get("eval_split") or "").strip().lower()
    if effective_eval_split:
        if effective_eval_split not in {"dev", "test", "all"}:
            raise ValueError(f"Unsupported Stage09 eval split: {effective_eval_split!r}. Expected dev, test, or all.")
        stage_cfg["eval_split"] = effective_eval_split
    else:
        stage_cfg.pop("eval_split", None)
    if eval_steps is not None:
        if int(eval_steps) <= 0:
            raise ValueError("--eval_steps must be a positive integer.")
        stage_cfg["eval_steps"] = int(eval_steps)
    cfg.values["stage09_rlvr_grpo"] = stage_cfg
    local_overrides = {
        "do_sample": True,
        "temperature": float(stage_cfg.get("temperature", 0.7)),
        "top_p": float(stage_cfg.get("top_p", 0.95)),
        "max_new_tokens": int(stage_cfg.get("max_new_tokens", 2048)),
        "trust_remote_code": bool(stage_cfg.get("trust_remote_code", True)),
        "device_map": stage_cfg.get("device_map", "auto"),
        "torch_dtype": stage_cfg.get("torch_dtype", "auto"),
    }
    effective_mode = llm_mode or "local"
    llm_client = get_agent_llm_client(
        cfg,
        llm_mode=effective_mode,
        model_path=str(base_model) if effective_mode == "local" else None,
        adapter_path=str(adapter_path) if effective_mode == "local" else None,
        local_overrides=local_overrides if effective_mode == "local" else None,
    )
    all_environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    environments, split_metadata = _select_stage09_environments_from_stage05_5_split(
        environments=all_environments,
        split=effective_split,
    )
    stage_cfg.update(split_metadata)
    eval_environments: list[ExecutableEnvSpec] = []
    if effective_eval_split:
        eval_environments, eval_metadata = _select_stage09_environments_from_stage05_5_split(
            environments=all_environments,
            split=effective_eval_split,
        )
        stage_cfg["eval_split"] = effective_eval_split
        stage_cfg["eval_split_source"] = eval_metadata.get("split_source")
        stage_cfg["eval_split_env_count"] = eval_metadata.get("split_env_count")
    cfg.values["stage09_rlvr_grpo"] = stage_cfg
    result = run_stage09_rlvr_grpo(
        cfg=cfg,
        environments=environments,
        eval_environments=eval_environments,
        llm_client=llm_client,
        output_dir=stage_result_dir(cfg, "09"),
        experiment_dir=cfg.output_dirs["experiments"],
        limit=limit,
        rollout_only=rollout_only,
        train=train,
        collect_rollouts=collect_rollouts,
        user_feedback=user_feedback,
        use_rubric_reward=use_rubric_reward,
        resume=resume,
        use_existing_rollouts=use_existing_rollouts,
    )
    manifest_train = result.get("manifest", {}).get("train", {})
    adapter_dir = manifest_train.get("adapter_dir")
    publish_paths = [
        Path(result["manifest"]["rollout_path"]),
        Path(result["manifest"]["reward_path"]),
        Path(result["manifest"]["summary_path"]),
        Path(result["manifest"]["train"]["output_dir"]) / "train_manifest.json",
    ]
    if adapter_dir:
        publish_paths.append(Path(adapter_dir) / "adapter_config.json")
    if is_main_process():
        publish_stage_results(cfg, f"09/{result['manifest']['policy_slug']}", publish_paths)
    barrier()
    return result


def stage09_5_eval_rl_adapter(
    cfg: AppConfig,
    limit: int | None = None,
    *,
    split: str | None = None,
    llm_mode: str | None = None,
    model_path: str | None = None,
    rl_adapter: str | None = None,
    retry_failed: bool = False,
    user_feedback: bool = False,
    resume: bool = False,
    parallel_workers: int | None = None,
) -> dict[str, Any]:
    stage09_cfg = cfg.values.get("stage09_rlvr_grpo", {}) or {}
    stage09_5_cfg = cfg.values.get("stage09_5_eval_rl", {}) or {}
    effective_split = str(split or stage09_5_cfg.get("split") or "test").strip().lower()
    if effective_split not in {"train", "dev", "test", "all"}:
        raise ValueError(f"Unsupported Stage09_5 split: {effective_split!r}. Expected train, dev, test, or all.")
    base_model = str(
        model_path
        or stage09_5_cfg.get("base_model")
        or stage09_cfg.get("base_model")
        or "/archive/zengjiaqi/Medical_LLM/Qwen3.5-2B-Base"
    ).strip()
    adapter_path = str(rl_adapter or stage09_5_cfg.get("rl_adapter") or "").strip()
    if not adapter_path:
        adapter_path = str(_latest_stage09_adapter_dir(cfg))
    if not Path(adapter_path).is_absolute():
        adapter_path = str(cfg.root / adapter_path)
    if not base_model:
        raise ValueError("Stage09_5 requires a base model path from --model_path or stage09_rlvr_grpo.base_model.")
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Stage09_5 RL adapter not found: {adapter_path}")

    all_environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    environments, split_metadata = _select_stage09_environments_from_stage05_5_split(
        environments=all_environments,
        split=effective_split,
    )
    local_overrides = {
        "do_sample": bool(stage09_5_cfg.get("do_sample", False)),
        "temperature": float(stage09_5_cfg.get("temperature", 0.0)),
        "top_p": float(stage09_5_cfg.get("top_p", 1.0)),
        "max_new_tokens": int(stage09_5_cfg.get("max_new_tokens", stage09_cfg.get("max_new_tokens", 2048))),
        "trust_remote_code": bool(stage09_5_cfg.get("trust_remote_code", stage09_cfg.get("trust_remote_code", True))),
        "device_map": stage09_5_cfg.get("device_map", stage09_cfg.get("device_map", "auto")),
        "torch_dtype": stage09_5_cfg.get("torch_dtype", stage09_cfg.get("torch_dtype", "auto")),
    }
    effective_mode = llm_mode or stage09_5_cfg.get("llm_mode") or "local"
    llm_client = get_agent_llm_client(
        cfg,
        llm_mode=effective_mode,
        model_path=base_model if effective_mode == "local" else None,
        adapter_path=adapter_path if effective_mode == "local" else None,
        local_overrides=local_overrides if effective_mode == "local" else None,
    )
    adapter_name = Path(adapter_path).parent.name if Path(adapter_path).name == "adapter" else Path(adapter_path).name
    eval_slug = slugify(f"{adapter_name}_{effective_split}", max_length=96)
    output_parent = stage_result_dir(cfg, "09_5") / eval_slug
    result = run_stage06_tool_agent(
        cfg=cfg,
        environments=environments,
        llm_client=llm_client,
        output_dir=output_parent,
        limit=limit,
        retry_failed=retry_failed,
        user_feedback=user_feedback,
        resume=resume,
        parallel_workers=parallel_workers,
    )
    leaf_output_dir = output_parent / agent_output_slug(llm_client)
    summary = result["summary"]
    summary.update(
        {
            "stage": "09_5_eval_rl_adapter",
            "eval_mode": "tool_agent_final_code_oracle",
            "split": effective_split,
            "split_source": split_metadata.get("split_source"),
            "split_env_count": split_metadata.get("split_env_count"),
            "base_model": base_model,
            "rl_adapter": adapter_path,
            "tool_format_version": "simplified_tool_json_v1",
            "output_dir": str(leaf_output_dir),
        }
    )
    (leaf_output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "stage": "09_5_eval_rl_adapter",
        "eval_mode": "tool_agent_final_code_oracle",
        "split": effective_split,
        "split_metadata": split_metadata,
        "limit": limit,
        "base_model": base_model,
        "rl_adapter": adapter_path,
        "tool_format_version": "simplified_tool_json_v1",
        "llm_mode": llm_client.mode,
        "output_dir": str(leaf_output_dir),
        "summary": summary,
    }
    (leaf_output_dir / "eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_stage_results(
        cfg,
        f"09_5/{eval_slug}",
        [
            leaf_output_dir / "agent_runs.jsonl",
            leaf_output_dir / "agent_traces.jsonl",
            leaf_output_dir / "agent_eval_report.jsonl",
            leaf_output_dir / "summary.json",
            leaf_output_dir / "eval_manifest.json",
        ],
    )
    result["summary"] = summary
    result["manifest"] = manifest
    return result


def _latest_stage09_adapter_dir(cfg: AppConfig) -> Path:
    root = cfg.output_dirs["experiments"] / "tool_rl_grpo_lora"
    candidates = [
        path.parent
        for path in root.glob("*/adapter/adapter_model.safetensors")
        if (path.parent / "adapter_config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No Stage09 RL adapter found under {root}. Run scripts/09_train_rlvr_grpo.py --train first, "
            "or pass --rl_adapter explicitly."
        )
    return max(candidates, key=lambda path: (path / "adapter_model.safetensors").stat().st_mtime)


def _select_stage09_environments_from_stage05_5_split(
    *,
    environments: list[ExecutableEnvSpec],
    split: str,
) -> tuple[list[ExecutableEnvSpec], dict[str, Any]]:
    if split == "all":
        return environments, {
            "split_source": "stage05_5_assigned_split" if has_stage05_5_split(environments) else "stage05_active_scaled_input",
            "split_filter": "all",
            "split_env_count": len(environments),
        }
    if not has_stage05_5_split(environments):
        raise ValueError(
            "Stage09 train/dev/test selection now requires Stage05_5 split labels. "
            "Run scripts/05_5_assign_splits.py first, or use --split all for legacy/debug mode."
        )
    split_map = split_envs_by_assigned_split(environments)
    selected = list(split_map.get(split, []))
    return selected, {
        "split_source": "stage05_5_assigned_split",
        "split_filter": split,
        "split_env_count": len(selected),
    }


def stage09_export(cfg: AppConfig) -> dict[str, list]:
    environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    point_rows = read_jsonl(cfg.output_dirs["processed"] / "question_points.jsonl")
    rubric_rows = read_jsonl(cfg.output_dirs["processed"] / "rubrics.jsonl")
    points_by_env: dict[str, list] = {}
    for row in point_rows:
        points_by_env.setdefault(row["env_id"], []).append(row)
    rubrics_by_env: dict[str, list] = {}
    for row in rubric_rows:
        rubrics_by_env.setdefault(row["env_id"], []).append(row)

    sft_samples = []
    pref_samples = []
    prm_samples = []
    rlvr_envs = []
    for env in environments:
        rubric_models = [RubricCriterion.model_validate(row) for row in rubrics_by_env.get(env.env_id, [])]
        point_models = [QuestionPoint.model_validate(row) for row in points_by_env.get(env.env_id, [])]
        sft_samples.append(
            export_sft_sample(
                env,
                rubric_models,
                system_prompt=cfg.values["generation"]["system_prompt"] if cfg.values["generation"]["include_system_prompt_in_sft"] else None,
            )
        )
        pref_samples.append(export_preference_sample(env, rubric_models))
        prm_samples.extend(export_prm_samples(env, point_models))
        rlvr_envs.append(export_rlvr_stub(env))

    sft_path = cfg.output_dirs["processed"] / "sft.jsonl"
    dpo_path = cfg.output_dirs["processed"] / "dpo.jsonl"
    preference_path = cfg.output_dirs["processed"] / "preference.jsonl"
    prm_path = cfg.output_dirs["processed"] / "prm.jsonl"
    prm_steps_path = cfg.output_dirs["processed"] / "prm_steps.jsonl"
    rlvr_path = cfg.output_dirs["processed"] / "rlvr_envs.jsonl"
    write_jsonl(sft_path, [sample.model_dump() for sample in sft_samples])
    write_jsonl(dpo_path, [sample.model_dump() for sample in pref_samples])
    write_jsonl(preference_path, [sample.model_dump() for sample in pref_samples])
    write_jsonl(prm_path, [sample.model_dump() for sample in prm_samples])
    write_jsonl(prm_steps_path, [sample.model_dump() for sample in prm_samples])
    write_jsonl(rlvr_path, [sample.model_dump() for sample in rlvr_envs])
    publish_stage_results(cfg, "09", [sft_path, dpo_path, preference_path, prm_path, prm_steps_path, rlvr_path])
    return {"sft": sft_samples, "preference": pref_samples, "prm": prm_samples, "rlvr": rlvr_envs}


def stage10_original_model_eval(
    cfg: AppConfig,
    limit: int | None = None,
    *,
    llm_mode: str | None = None,
    model_path: str | None = None,
    sft_adapter: str | None = None,
    rl_adapter: str | None = None,
    eval_sft: bool = True,
    eval_rl: bool = True,
    retry_failed: bool = False,
    user_feedback: bool = False,
    resume: bool = False,
    parallel_workers: int | None = None,
) -> dict[str, Any]:
    environments = _stage10_original_environments(cfg)
    if limit is not None:
        environments = environments[:limit]
    output_root = stage_result_dir(cfg, "10") / "original_model_eval"
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "original_eval_envs.jsonl", [env.model_dump() for env in environments])

    model_specs = _stage10_model_specs(
        cfg=cfg,
        llm_mode=llm_mode,
        model_path=model_path,
        sft_adapter=sft_adapter,
        rl_adapter=rl_adapter,
        eval_sft=eval_sft,
        eval_rl=eval_rl,
    )
    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for spec in model_specs:
        print(_stage10_model_status_line(spec), flush=True)
        llm_client = get_agent_llm_client(
            cfg,
            llm_mode=spec["llm_mode"],
            model_path=spec.get("model_path"),
            adapter_path=spec.get("adapter_path"),
            local_overrides=spec.get("local_overrides"),
        )
        model_output_parent = output_root / spec["label"]
        result = run_stage06_tool_agent(
            cfg=cfg,
            environments=environments,
            llm_client=llm_client,
            output_dir=model_output_parent,
            limit=None,
            retry_failed=retry_failed,
            user_feedback=user_feedback,
            resume=resume,
            parallel_workers=parallel_workers,
        )
        leaf_output_dir = model_output_parent / agent_output_slug(llm_client)
        summary = dict(result["summary"])
        summary.update(
            {
                "stage": "10_original_model_eval",
                "model_label": spec["label"],
                "model_kind": spec["kind"],
                "base_model": spec.get("model_path") or "",
                "adapter": spec.get("adapter_path") or "",
                "input_path": str(raw_test_path(cfg)),
                "ground_truth_source": "stage00_seed_execution_case_expected_output_signature",
                "num_original_envs": len(environments),
                "output_dir": str(leaf_output_dir),
            }
        )
        (leaf_output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "stage": "10_original_model_eval",
            "model_spec": spec,
            "input_path": str(raw_test_path(cfg)),
            "ground_truth_source": "stage00_seed_execution_case_expected_output_signature",
            "num_original_envs": len(environments),
            "summary": summary,
            "output_dir": str(leaf_output_dir),
        }
        (leaf_output_dir / "eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        results[spec["label"]] = {"summary": summary, "manifest": manifest, "output_dir": str(leaf_output_dir)}
        comparison_rows.append(_stage10_comparison_row(summary))
    comparison = {
        "stage": "10_original_model_eval",
        "input_path": str(raw_test_path(cfg)),
        "ground_truth_source": "stage00_seed_execution_case_expected_output_signature",
        "num_original_envs": len(environments),
        "models": comparison_rows,
        "deltas_vs_base": _stage10_deltas_vs_base(comparison_rows),
    }
    comparison_path = output_root / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_stage_results(cfg, "10/original_model_eval", [output_root / "original_eval_envs.jsonl", comparison_path])
    return {"comparison": comparison, "models": results}


def _stage10_model_status_line(spec: dict[str, Any]) -> str:
    label_names = {
        "base": "base",
        "sft": "sft",
        "rl": "sft+RL",
    }
    label = label_names.get(str(spec.get("label") or ""), str(spec.get("label") or spec.get("kind") or "model"))
    parts = [f"Stage10 evaluating {label} model"]
    model_path = str(spec.get("model_path") or "").strip()
    adapter_path = str(spec.get("adapter_path") or "").strip()
    if model_path:
        parts.append(f"base={model_path}")
    if adapter_path:
        parts.append(f"adapter={adapter_path}")
    return " | ".join(parts)


def _stage10_original_environments(cfg: AppConfig) -> list[ExecutableEnvSpec]:
    input_path = raw_test_path(cfg)
    rows = read_jsonl(input_path)
    environments: list[ExecutableEnvSpec] = []
    for index, row in enumerate(rows, start=1):
        seed_case = row.get("seed_execution_case") if isinstance(row.get("seed_execution_case"), dict) else {}
        expected = seed_case.get("expected_output_signature") if isinstance(seed_case.get("expected_output_signature"), dict) else {}
        if not seed_case or not expected:
            continue
        task_id = str(row.get("task_id") or row.get("idx") or f"row_{index}")
        code = str(row.get("code") or "").strip()
        if not code:
            continue
        source_split = str(row.get("source_split") or "original")
        environments.append(
            ExecutableEnvSpec(
                env_id=f"original_{slugify(task_id, max_length=72)}",
                original_task_id=task_id,
                split=source_split,
                problem=str(row.get("problem") or ""),
                context=str(row.get("context") or ""),
                signature=row.get("signature"),
                solution_form=str(row.get("solution_form") or "complete_executable_program"),
                primary_domain=str(row.get("primary_domain") or row.get("domain") or "scientific_software_engineering"),
                primary_task_type=str(row.get("primary_task_type") or row.get("task_type") or "code_generation"),
                verifier_type_hint="stage00_ground_truth_replay",
                code=code,
                gold_solution=code,
                seed_gold_solution=code,
                scaled_gold_solution=code,
                scaled_executable_gold_code=code,
                seed_execution_case=dict(seed_case),
                seed_ground_truth_output_signature=dict(row.get("ground_truth_output_signature") or expected),
                seed_case_audit=dict(row.get("seed_case_audit") or {}),
                resource_files=list(row.get("resource_files") or []),
                visible_state={"include": [], "hide": [], "placeholder_token": row.get("placeholder_token") or "<<insert solution here>>"},
                metadata={
                    "stage10_source": "stage00_raw",
                    "source_split": source_split,
                    "idx": row.get("idx"),
                    "execution_status": row.get("execution_status"),
                    "repair_attempts": row.get("repair_attempts", 0),
                },
            )
        )
    if not environments:
        raise ValueError(f"No Stage10 original eval environments could be built from {input_path}.")
    return environments


def _stage10_model_specs(
    *,
    cfg: AppConfig,
    llm_mode: str | None,
    model_path: str | None,
    sft_adapter: str | None,
    rl_adapter: str | None,
    eval_sft: bool,
    eval_rl: bool,
) -> list[dict[str, Any]]:
    base_model = _stage10_base_model_path(cfg, model_path)
    local_overrides = _stage10_local_overrides(cfg)
    effective_mode = llm_mode or ("local" if base_model else cfg.values["generation"].get("llm_mode") or "mock")
    specs: list[dict[str, Any]] = [
        {
            "label": "base",
            "kind": "base",
            "llm_mode": effective_mode,
            "model_path": base_model if effective_mode == "local" else None,
            "adapter_path": None,
            "local_overrides": local_overrides if effective_mode == "local" else None,
        }
    ]
    if effective_mode != "local" or not base_model:
        return specs
    if eval_sft:
        adapter = _stage10_sft_adapter_path(cfg, sft_adapter)
        if sft_adapter and (not adapter or not Path(adapter).exists()):
            raise FileNotFoundError(f"Stage10 SFT adapter not found: {adapter}")
        if adapter and Path(adapter).exists():
            specs.append(
                {
                    "label": "sft",
                    "kind": "sft_adapter",
                    "llm_mode": "local",
                    "model_path": base_model,
                    "adapter_path": adapter,
                    "local_overrides": local_overrides,
                }
            )
    if eval_rl:
        adapter = _stage10_rl_adapter_path(cfg, rl_adapter)
        if rl_adapter and (not adapter or not Path(adapter).exists()):
            raise FileNotFoundError(f"Stage10 RL adapter not found: {adapter}")
        if adapter and Path(adapter).exists():
            specs.append(
                {
                    "label": "rl",
                    "kind": "rl_adapter",
                    "llm_mode": "local",
                    "model_path": base_model,
                    "adapter_path": adapter,
                    "local_overrides": local_overrides,
                }
            )
    return specs


def _stage10_base_model_path(cfg: AppConfig, model_path: str | None) -> str:
    if model_path:
        return str(model_path)
    stage09_cfg = cfg.values.get("stage09_rlvr_grpo", {}) or {}
    if stage09_cfg.get("base_model"):
        return str(stage09_cfg["base_model"])
    train_config = cfg.dataset_config_path("train_sft.yaml")
    if train_config.exists():
        _, train_cfg = load_training_config(train_config, dataset=cfg.dataset_name)
        if train_cfg.get("model_name_or_path"):
            return str(train_cfg["model_name_or_path"])
    stage06_cfg = cfg.values.get("stage06", {}) or {}
    configured_path = cfg.root / str(stage06_cfg.get("llm_config") or "configs/agent_llm.yaml")
    if configured_path.exists():
        agent_values = load_yaml(configured_path)
        local_model = ((agent_values.get("local") or {}).get("model_path") or "")
        if str(local_model).strip():
            return str(local_model)
    return ""


def _stage10_sft_adapter_path(cfg: AppConfig, sft_adapter: str | None) -> str:
    if sft_adapter:
        return str(Path(sft_adapter) if Path(sft_adapter).is_absolute() else cfg.root / sft_adapter)
    train_config = cfg.dataset_config_path("train_sft.yaml")
    if train_config.exists():
        train_root, train_cfg = load_training_config(train_config, dataset=cfg.dataset_name)
        return str(train_root / str(train_cfg.get("output_dir") or "experiments/biocoder/tool_sft_lora") / "adapter")
    return str(cfg.output_dirs["experiments"] / "tool_sft_lora" / "adapter")


def _stage10_rl_adapter_path(cfg: AppConfig, rl_adapter: str | None) -> str:
    if rl_adapter:
        return str(Path(rl_adapter) if Path(rl_adapter).is_absolute() else cfg.root / rl_adapter)
    try:
        return str(_latest_stage09_adapter_dir(cfg))
    except FileNotFoundError:
        return ""


def _stage10_local_overrides(cfg: AppConfig) -> dict[str, Any]:
    stage10_cfg = cfg.values.get("stage10_original_eval", {}) or {}
    fallback = cfg.values.get("stage09_5_eval_rl", {}) or cfg.values.get("stage08_5_eval_sft", {}) or {}
    return {
        "do_sample": bool(stage10_cfg.get("do_sample", fallback.get("do_sample", False))),
        "temperature": float(stage10_cfg.get("temperature", fallback.get("temperature", 0.0))),
        "top_p": float(stage10_cfg.get("top_p", fallback.get("top_p", 1.0))),
        "max_new_tokens": int(stage10_cfg.get("max_new_tokens", fallback.get("max_new_tokens", 2048))),
        "device_map": stage10_cfg.get("device_map", fallback.get("device_map", "auto")),
        "torch_dtype": stage10_cfg.get("torch_dtype", fallback.get("torch_dtype", "auto")),
        "trust_remote_code": bool(stage10_cfg.get("trust_remote_code", True)),
    }


def _stage10_comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_label": summary.get("model_label"),
        "model_kind": summary.get("model_kind"),
        "num_envs": summary.get("num_envs", summary.get("num_original_envs")),
        "sample_pass_rate": summary.get("sample_pass_rate", 0.0),
        "case_pass_rate": summary.get("case_pass_rate", 0.0),
        "case_pass_rate_nonzero_only": summary.get("case_pass_rate_nonzero_only", 0.0),
        "nonzero_case_sample_pass_rate": summary.get("nonzero_case_sample_pass_rate", 0.0),
        "passed_cases": summary.get("passed_cases", 0),
        "total_cases": summary.get("total_cases", 0),
        "output_dir": summary.get("output_dir"),
        "adapter": summary.get("adapter"),
    }


def _stage10_deltas_vs_base(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = next((row for row in rows if row.get("model_label") == "base"), None)
    if not base:
        return []
    deltas = []
    for row in rows:
        if row is base:
            continue
        deltas.append(
            {
                "model_label": row.get("model_label"),
                "sample_pass_rate_delta": round(float(row.get("sample_pass_rate") or 0.0) - float(base.get("sample_pass_rate") or 0.0), 6),
                "case_pass_rate_delta": round(float(row.get("case_pass_rate") or 0.0) - float(base.get("case_pass_rate") or 0.0), 6),
            }
        )
    return deltas


def stage10_quality_filter(cfg: AppConfig) -> list[dict]:
    report = stage08_safety(cfg)
    publish_stage_results(
        cfg,
        "10",
        [
            quality_report_output_path(cfg),
            scaled_clean_output_path(cfg),
            scaled_rejected_output_path(cfg),
            scaled_raw_output_path(cfg),
        ],
    )
    return report


def _split_seed_ids(cfg: AppConfig, seed_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    seed_ids = seeded_shuffle(seed_ids, cfg.values["splits"]["seed"])
    total = len(seed_ids)
    if total >= 3:
        dev_count = max(1, int(total * cfg.values["splits"]["dev"]))
        test_count = max(1, int(total * cfg.values["splits"]["test"]))
        train_count = max(1, total - dev_count - test_count)
        if train_count + dev_count + test_count > total:
            train_count = total - dev_count - test_count
        train_end = train_count
        dev_end = train_count + dev_count
    else:
        train_end = max(1, int(total * cfg.values["splits"]["train"]))
        dev_end = train_end + int(total * cfg.values["splits"]["dev"])
    train_ids = seed_ids[:train_end]
    dev_ids = seed_ids[train_end:dev_end]
    test_ids = seed_ids[dev_end:]
    return train_ids, dev_ids, test_ids


def stage11_make_splits(cfg: AppConfig) -> dict[str, list]:
    seed_ids = sorted({row["env_id"] for row in read_jsonl(cfg.output_dirs["processed"] / "sft.jsonl")})
    train_ids, dev_ids, test_ids = _split_seed_ids(cfg, seed_ids)
    train_ids_path = cfg.output_dirs["splits"] / "train_seed_ids.json"
    dev_ids_path = cfg.output_dirs["splits"] / "dev_seed_ids.json"
    test_ids_path = cfg.output_dirs["splits"] / "test_seed_ids.json"
    train_ids_path.write_text(json.dumps(train_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    dev_ids_path.write_text(json.dumps(dev_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    test_ids_path.write_text(json.dumps(test_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    split_paths = [train_ids_path, dev_ids_path, test_ids_path]
    for name, path in {
        "sft": cfg.output_dirs["processed"] / "sft.jsonl",
        "dpo": cfg.output_dirs["processed"] / "dpo.jsonl",
        "prm": cfg.output_dirs["processed"] / "prm.jsonl",
    }.items():
        rows = read_jsonl(path)
        for split_name, split_ids in {"train": train_ids, "dev": dev_ids, "test": test_ids}.items():
            subset = [row for row in rows if row.get("env_id") in split_ids]
            split_path = cfg.output_dirs["splits"] / f"{name}_{split_name}.jsonl"
            write_jsonl(split_path, subset)
            split_paths.append(split_path)
    publish_stage_results(cfg, "11", split_paths)
    return {"train": train_ids, "dev": dev_ids, "test": test_ids}


def stage15_eval(cfg: AppConfig) -> dict:
    environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    pref_rows = read_jsonl(cfg.output_dirs["processed"] / "dpo.jsonl")
    metrics = {
        "num_environments": len(environments),
        "num_dpo_pairs": len(pref_rows),
        "level_breakdown": {},
        "primary_domain_breakdown": {},
        "secondary_domain_breakdown": {},
        "mean_hidden_tests_per_env": round(
            sum(len(env.hidden_tests) for env in environments) / max(len(environments), 1),
            3,
        ),
    }
    for level in cfg.values["generation"]["levels"]:
        envs = [env for env in environments if env.difficulty and env.difficulty.global_level == level]
        metrics["level_breakdown"][level] = {"count": len(envs)}
    for env in environments:
        metrics["primary_domain_breakdown"][env.primary_domain] = metrics["primary_domain_breakdown"].get(env.primary_domain, 0) + 1
        for item in env.secondary_domains:
            metrics["secondary_domain_breakdown"][item.domain] = metrics["secondary_domain_breakdown"].get(item.domain, 0) + 1
    report_path = cfg.output_dirs["experiments"] / "reports" / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_stage_results(cfg, "15", [report_path])
    return metrics


def build_question_points(env: ExecutableEnvSpec) -> list[QuestionPoint]:
    points = [
        QuestionPoint(
            env_id=env.env_id,
            point_id=f"{env.env_id}_qp_contract",
            title="Preserve code contract",
            description="Return code that correctly fits the required solution_form and placeholder location.",
            related_axes=["C", "V"],
        ),
        QuestionPoint(
            env_id=env.env_id,
            point_id=f"{env.env_id}_qp_verifier",
            title="Satisfy verifier-facing checks",
            description=f"Meet the expected verifier type {env.verifier_type_hint or 'unit_test'} and hidden checks.",
            related_axes=["V"],
        ),
    ]
    secondary_domain_names = [item.domain for item in env.secondary_domains]
    if secondary_domain_names:
        points.append(
            QuestionPoint(
                env_id=env.env_id,
                point_id=f"{env.env_id}_qp_secondary_domains",
                title="Use Secondary Domain Context",
                description=f"Use auxiliary semantic context from secondary domains: {', '.join(secondary_domain_names)}.",
                related_axes=["D", "A"],
            )
        )
    for axis in (env.difficulty.selected_axes if env.difficulty else []):
        points.append(
            QuestionPoint(
                env_id=env.env_id,
                point_id=f"{env.env_id}_qp_{axis.lower()}",
                title=f"Axis {axis} requirement",
                description=f"Handle the {axis} difficulty axis requirements without violating the core task contract.",
                related_axes=[axis],
            )
        )
    return points


def build_rubrics(env: ExecutableEnvSpec, points: list[QuestionPoint]) -> list[RubricCriterion]:
    rubrics = []
    secondary_domain_names = [item.domain for item in env.secondary_domains]
    for index, point in enumerate(points, start=1):
        related = point.related_axes[0] if point.related_axes else "general"
        weight = 2 if related in (env.difficulty.selected_axes if env.difficulty else []) else 1
        if point.point_id.endswith("secondary_domains"):
            weight = max(weight, 2)
        rubrics.append(
            RubricCriterion(
                env_id=env.env_id,
                rubric_id=f"{env.env_id}_rubric_{index:02d}",
                source_point_id=point.point_id,
                criterion=point.description
                if not secondary_domain_names
                else f"{point.description} Primary domain={env.primary_domain}; secondary domains={', '.join(secondary_domain_names)}.",
                score_type="binary",
                weight=weight,
                category=related,
            )
        )
    return rubrics


def build_sampled_trajectories(env: ExecutableEnvSpec, points: list[QuestionPoint]) -> list[dict]:
    return [
        {
            "trajectory_id": f"traj_{env.env_id}_pass",
            "env_id": env.env_id,
            "outcome": "pass",
            "steps": [point.description for point in points],
        },
        {
            "trajectory_id": f"traj_{env.env_id}_fail",
            "env_id": env.env_id,
            "outcome": "fail",
            "steps": ["skip resource inspection", "submit shortcut code"],
        },
    ]


def load_config(config_path: str | Path, dataset: str | None = None) -> AppConfig:
    return load_app_config(config_path, dataset=dataset)


# Backward-compatible names for older tests and scripts. The canonical numbering
# is now Stage06 tool-agent, then the old post-scaling stages shifted by one.
stage06_qpoints_rubrics = stage07_qpoints_rubrics
stage07_safety = stage08_safety
stage08_export = stage09_export
stage09_quality_filter = stage10_quality_filter
stage10_make_splits = stage11_make_splits
stage14_eval = stage15_eval
