from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import (
    load_config,
    stage00_download,
    stage01_normalize,
    stage02_route,
    stage03_seed,
    stage04_skeleton,
    stage05_scale,
    stage05_5_assign_splits,
    stage06_tool_agent,
    stage07_qpoints_rubrics,
    stage07_tool_sft_data,
    stage08_safety,
    stage08_5_eval_sft,
    stage08_train_sft,
    stage09_export,
    stage09_5_eval_rl_adapter,
    stage09_rlvr_grpo,
    stage10_original_model_eval,
    stage10_quality_filter,
    stage11_make_splits,
    stage15_eval,
)


STAGES = {
    "load_tasks": stage00_download,
    "normalize_tasks": stage01_normalize,
    "route_tasks": stage02_route,
    "build_seed_envs": stage03_seed,
    "build_base_envs": stage04_skeleton,
    "scale_envs": stage05_scale,
    "assign_splits_05_5": stage05_5_assign_splits,
    "assign_splits": stage05_5_assign_splits,
    "tool_agent": stage06_tool_agent,
    "tool_sft_data": stage07_tool_sft_data,
    "qpoints_rubrics_legacy": stage07_qpoints_rubrics,
    "train_sft": stage08_train_sft,
    "eval_sft_08_5": stage08_5_eval_sft,
    "eval_sft": stage08_5_eval_sft,
    "quality_report_legacy": stage08_safety,
    "rlvr_grpo": stage09_rlvr_grpo,
    "eval_rl_09_5": stage09_5_eval_rl_adapter,
    "eval_rl": stage09_5_eval_rl_adapter,
    "export_views_legacy": stage09_export,
    "original_eval": stage10_original_model_eval,
    "original_model_eval": stage10_original_model_eval,
    "quality_filter": stage10_quality_filter,
    "make_splits": stage11_make_splits,
    "eval": stage15_eval,
}


if __name__ == "__main__":
    parser = base_parser("Run a single MedEnvScale pipeline stage")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES.keys()))
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    stage_fn = STAGES[args.stage]
    kwargs = {}
    if "limit" in stage_fn.__code__.co_varnames:
        kwargs["limit"] = args.limit
    if "sample_seed" in stage_fn.__code__.co_varnames:
        kwargs["sample_seed"] = args.sample_seed
    if "llm_mode" in stage_fn.__code__.co_varnames:
        kwargs["llm_mode"] = args.llm_mode
    if "parallel_workers" in stage_fn.__code__.co_varnames:
        kwargs["parallel_workers"] = args.workers
    if "resume" in stage_fn.__code__.co_varnames:
        kwargs["resume"] = args.resume or args.resume_stage05
    result = stage_fn(cfg, **kwargs)
    size = len(result) if isinstance(result, list) else (len(result) if isinstance(result, dict) else 1)
    print(f"Stage {args.stage} complete: output_size={size}")
