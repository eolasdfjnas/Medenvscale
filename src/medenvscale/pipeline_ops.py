from __future__ import annotations

import json
import shutil
from pathlib import Path

from medenvscale.agent import run_stage06_tool_agent
from medenvscale.classify.llm_full_taxonomy_router import route_with_llm_full_taxonomy
from medenvscale.classify.routing_validator import validate_routing
from medenvscale.config import AppConfig, load_app_config
from medenvscale.export.export_preference import export_preference_sample
from medenvscale.export.export_prm import export_prm_samples
from medenvscale.export.export_rlvr import export_rlvr_stub
from medenvscale.export.export_sft import export_sft_sample
from medenvscale.ingest.code_execution import validate_and_repair_code_rows
from medenvscale.ingest.load_medagentgym import load_raw_medagentgym
from medenvscale.ingest.normalize_medagentgym import normalize_rows
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
from medenvscale.scaling.scaling_plan import build_scaling_plan
from medenvscale.scaling.scaled_gold_solver_generator import (
    build_scaled_case_plan,
    check_seed_case_admission,
    detect_semantic_change,
    generate_scaled_gold_solution_if_needed,
    repair_scaled_gold_solution,
)
from medenvscale.scaling.verifier_delta_validator import validate_verifier_delta
from medenvscale.scaling.verifier_delta_normalizer import normalize_verifier_delta
from medenvscale.schemas import (
    DifficultyProfile,
    ExecutableEnvSpec,
    MedAgentGymTask,
    QuestionPoint,
    RoutingResult,
    RubricCriterion,
)
from medenvscale.utils import load_yaml, print_progress, read_jsonl, seeded_shuffle, slugify, write_jsonl
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


def get_agent_llm_client(cfg: AppConfig, llm_mode: str | None = None) -> LLMClient:
    stage06_cfg = cfg.values.get("stage06", {}) or {}
    configured_path = stage06_cfg.get("llm_config") or "configs/agent_llm.yaml"
    agent_config_path = cfg.root / str(configured_path)
    agent_values = load_yaml(agent_config_path) if agent_config_path.exists() else cfg.llm_values
    agent_values = json.loads(json.dumps(agent_values))
    agent_model_slug = slugify(str((agent_values.get("api") or {}).get("model") or "agent_model"))
    if cfg.dataset_name:
        agent_values.setdefault("cache", {})
        agent_values["cache"]["dir"] = str(Path(".cache") / "agent_llm" / cfg.dataset_name / agent_model_slug)
        agent_values.setdefault("trace", {})
        agent_values["trace"]["path"] = str(
            Path("data") / cfg.dataset_name / "processed" / Path(str(agent_values["trace"].get("path", "agent_generation_trace.jsonl"))).name
        )
    mode = llm_mode or stage06_cfg.get("llm_mode") or cfg.values["generation"].get("llm_mode") or "mock"
    return LLMClient(
        config=agent_values,
        mode=mode,
        cache_dir=str(cfg.root / agent_values["cache"]["dir"]),
        trace_path=str(cfg.root / agent_values["trace"]["path"]),
    )


def get_prompt_runner(cfg: AppConfig) -> PromptRunner:
    return PromptRunner(cfg.root / "prompts")


def raw_path(cfg: AppConfig) -> Path:
    return cfg.root / cfg.values["dataset"]["local_raw_path"]


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


def _active_scaled_input_path(cfg: AppConfig) -> Path:
    clean_path = scaled_clean_output_path(cfg)
    if clean_path.exists():
        return clean_path
    return scaled_output_path(cfg)


def stage_result_dir(cfg: AppConfig, stage_name: str) -> Path:
    target = cfg.output_dirs["result"] / stage_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def publish_stage_results(cfg: AppConfig, stage_name: str, paths: list[Path]) -> None:
    target_dir = stage_result_dir(cfg, stage_name)
    for path in paths:
        if not path.exists():
            continue
        shutil.copy2(path, target_dir / path.name)


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


def stage00_download(cfg: AppConfig, limit: int | None = None, llm_mode: str | None = None) -> dict:
    dataset_cfg = cfg.values["dataset"]
    destination = raw_path(cfg)
    rejected_destination = raw_rejected_path(cfg)
    task_files = dataset_cfg.get("task_files", {})
    merged_rows = []
    files_used: list[str] = []

    for split_name, relative_path in task_files.items():
        source_path = cfg.root / str(relative_path)
        if not source_path.exists():
            continue
        rows = read_jsonl(source_path)
        for row in rows:
            materialized = dict(row)
            materialized.setdefault("source_split", split_name)
            merged_rows.append(materialized)
        files_used.append(_display_path(source_path, cfg.root))

    if merged_rows:
        if limit is not None:
            merged_rows = merged_rows[:limit]
    else:
        merged_rows = read_jsonl(destination)
        if not merged_rows:
            raise FileNotFoundError(
                "No MedAgentGym raw tasks found. Provide dataset.task_files or place JSONL rows at "
                f"{destination}."
            )
        if limit is not None:
            merged_rows = merged_rows[:limit]

    llm_client = get_llm_client(cfg, llm_mode=llm_mode) if merged_rows else None
    prompt_runner = get_prompt_runner(cfg) if merged_rows else None
    prepared_rows, rejected_rows, validation_summary = validate_and_repair_code_rows(
        merged_rows,
        cfg=cfg,
        llm_client=llm_client,
        prompt_runner=prompt_runner,
    )
    write_jsonl(destination, prepared_rows)
    write_jsonl(rejected_destination, rejected_rows)

    metadata = {
        "dataset_name": dataset_cfg.get("name", "medagentgym"),
        "download_method": "local_jsonl_merge_with_code_validation",
        "rows_written": len(prepared_rows),
        **validation_summary,
        "splits_requested": list(task_files.keys()) or sorted({row.get("source_split", "train") for row in merged_rows}),
        "files_used": files_used or [_display_path(destination, cfg.root)],
        "rejected_output": _display_path(rejected_destination, cfg.root),
    }
    metadata_path = cfg.root / dataset_cfg["metadata_path"]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_stage_results(cfg, "00", [destination, rejected_destination, metadata_path])
    return metadata


def stage01_normalize(cfg: AppConfig, limit: int | None = None) -> list[MedAgentGymTask]:
    rows = load_raw_medagentgym(raw_path(cfg), limit=limit)
    normalized = normalize_rows(rows, cfg.values["dataset"]["split_source"])
    output = normalize_output_path(cfg)
    write_jsonl(output, [item.model_dump() for item in normalized])
    publish_stage_results(cfg, "01", [output])
    return normalized


def stage02_route(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
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
    llm_client = get_llm_client(cfg, llm_mode=llm_mode)
    prompt_runner = get_prompt_runner(cfg)
    routed: list[RoutingResult] = []
    total = len(normalized)
    if total:
        print_progress(0, total, label="Routing MedAgentGym")
    for index, item in enumerate(normalized, start=1):
        llm_payload = route_with_llm_full_taxonomy(item, llm_client, prompt_runner, allowed_domains, allowed_task_types)
        routed.append(
            validate_routing(
                item=item,
                routing=llm_payload,
                allowed_domains=allowed_domains,
                allowed_task_types=allowed_task_types,
                allowed_solution_forms=allowed_solution_forms,
                min_confidence=cfg.values["routing"]["min_confidence"],
                review_confidence=cfg.values["routing"]["review_confidence"],
            )
        )
        if total:
            print_progress(index, total, label="Routing MedAgentGym")
    output = output_path or routing_output_path(cfg)
    write_jsonl(output, [item.model_dump() for item in routed])
    publish_stage_results(cfg, "02", [output])
    return routed


def stage03_seed(cfg: AppConfig, limit: int | None = None) -> list[ExecutableEnvSpec]:
    tasks = {item.task_id: item for item in _load_models(normalize_output_path(cfg), MedAgentGymTask)}
    routed = _load_models(routing_output_path(cfg), RoutingResult)
    if limit is not None:
        routed = routed[:limit]
    seed_envs: list[ExecutableEnvSpec] = []
    for route in routed:
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
        seed_envs.append(seed_env)
    validate_seed_count(seed_envs, limit)
    output = seed_output_path(cfg)
    write_jsonl(output, [seed.model_dump() for seed in seed_envs])
    publish_stage_results(cfg, "03", [output])
    return seed_envs


def stage04_skeleton(cfg: AppConfig, limit: int | None = None, llm_mode: str | None = None) -> list[ExecutableEnvSpec]:
    seed_envs = stage03_seed(cfg, limit=limit)
    publish_stage_results(cfg, "04", [seed_output_path(cfg)])
    return seed_envs


def stage05_scale(
    cfg: AppConfig,
    limit: int | None = None,
    llm_mode: str | None = None,
    sample_seed: int | None = None,
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
    llm_client = get_llm_client(cfg, llm_mode=llm_mode)
    prompt_runner = get_prompt_runner(cfg)

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
    levels = list(cfg.values["generation"]["levels"])
    progress = tqdm(total=len(seed_envs) * len(levels), desc="Stage05 Scaling", unit="env", leave=True)

    try:
        for seed_env in seed_envs:
            for level in levels:
                env_id = f"env_{seed_env.original_task_id}_{level}"
                axis_weights = plan_axis_weights(
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
                axis_weight_result, axis_weight_source, axis_weight_trace = axis_weights
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
                semantic_test_specs = _collect_semantic_test_specs(
                    operator_instances=operator_instances,
                    global_level=level,
                    original_task_id=seed_env.original_task_id,
                )
                output_constraint_spec = normalize_output_constraint_spec(
                    environment=env,
                    operator_instances=[op.model_dump() for op in operator_instances],
                )
                output_requirements = []
                for op in operator_instances:
                    output_requirements.extend(str(item).strip() for item in (op.output_requirements or []) if str(item).strip())
                env = env.model_copy(
                    update={
                        "semantic_test_specs": semantic_test_specs,
                        "output_requirements": output_requirements,
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
                scaled_case_plan_rows.append(
                    {
                        "env_id": env.env_id,
                        "original_task_id": env.original_task_id,
                        "difficulty": level,
                        **scaled_case_plan,
                    }
                )
                if not seed_case_admission["passed"]:
                    quality_flags.extend(
                        f"SEED_CASE_ADMISSION_FAILED:{reason}"
                        for reason in seed_case_admission["failure_reasons"]
                    )
                gold_result = generate_scaled_gold_solution_if_needed(
                    env=env,
                    operator_instances=[op.model_dump() for op in operator_instances],
                    semantic_test_specs=semantic_test_specs,
                    output_constraint_spec=output_constraint_spec,
                    llm_client=llm_client,
                    prompt_runner=prompt_runner,
                    config={"max_gold_repair_attempts": 3, "stage05_cfg": stage05_cfg},
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
                gold_result = repair_scaled_gold_solution(
                    env=env,
                    gold_result=gold_result,
                    semantic_test_specs=semantic_test_specs,
                    output_constraint_spec=output_constraint_spec,
                    llm_client=llm_client,
                    prompt_runner=prompt_runner,
                    config={"max_gold_repair_attempts": 3, "stage05_cfg": stage05_cfg},
                )
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
                        combined = [
                            item for item in validated_oracle_cases
                            if isinstance(item, dict) and "," in str(item.get("axis") or "")
                        ]
                        if len(validated_oracle_cases) > 1 and not combined:
                            quality_flags.append("M4_COMBINED_ORACLE_CASE_MISSING")
                verifier_spec = build_verifier_spec(env, operator_instances, hidden_tests=[])
                env = env.model_copy(
                    update={
                        "verifier_spec": verifier_spec.model_dump(),
                        "hidden_tests": [],
                        "hidden_tests_mode": "disabled_in_case_first_stage05",
                        "operator_instances": [op.model_dump() for op in operator_instances],
                    }
                )
                operator_realization = [] if level == "M1" else check_operator_realizations(seed_env, env)
                operator_realization_rows.extend(operator_realization)
                for report in operator_realization:
                    operator_id = report["operator_id"]
                    severity = report["severity"]
                    if severity in {"soft_fail", "hard_fail"}:
                        quality_flags.append(
                            f"operator_realization_{severity}:{operator_id}:{','.join(report['failure_reasons'])}"
                        )
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
                stage05_quality_rows.append(gate_report)
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
                scaled_envs.append(env)
                scaling_plan_rows.append(scaling_plan.model_dump())
                verifier_rows.append(verifier_spec.model_dump())
                oracle_case_validation_rows.extend(env.oracle_case_validation_report)
                scaled_gold_case_execution_rows.extend(env.scaled_gold_case_execution_report)
                failure_reasons = list(gate_report.get("rejection_reasons", []) or [])
                failure_stage_breakdown = _build_stage05_failure_breakdown(failure_reasons)
                primary_failure_stage = next((stage for stage, items in failure_stage_breakdown.items() if items), "other")
                stage05_failure_summary_rows.append(
                    {
                        "env_id": env.env_id,
                        "level": level,
                        "failure_reasons": failure_reasons,
                        "failure_stage_breakdown": failure_stage_breakdown,
                        "primary_failure_stage": primary_failure_stage,
                        "stage05_passed": bool(gate_report.get("stage05_passed")),
                        "final_decision": str(gate_report.get("final_decision") or ""),
                    }
                )
                for op in operator_instances:
                    operator_rows.append({"env_id": env_id, **op.model_dump()})
                for case in env.scaled_oracle_cases:
                    if isinstance(case, dict):
                        hidden_tests.append({"env_id": env_id, "source": "optional_export_compatibility", **case})
                progress.update(1)
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
    return scaled_envs


def stage06_tool_agent(cfg: AppConfig, limit: int | None = None, llm_mode: str | None = None) -> dict[str, list]:
    environments = _load_models(_active_scaled_input_path(cfg), ExecutableEnvSpec)
    llm_client = get_agent_llm_client(cfg, llm_mode=llm_mode)
    return run_stage06_tool_agent(
        cfg=cfg,
        environments=environments,
        llm_client=llm_client,
        output_dir=stage_result_dir(cfg, "06"),
        limit=limit,
    )


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
