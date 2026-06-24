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
    stage06_tool_agent,
    stage07_qpoints_rubrics,
    stage08_safety,
    stage09_export,
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
    "tool_agent": stage06_tool_agent,
    "qpoints_rubrics": stage07_qpoints_rubrics,
    "quality_report": stage08_safety,
    "export_views": stage09_export,
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
    result = stage_fn(cfg, **kwargs)
    size = len(result) if isinstance(result, list) else (len(result) if isinstance(result, dict) else 1)
    print(f"Stage {args.stage} complete: output_size={size}")
