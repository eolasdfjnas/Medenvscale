from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import (
    load_config,
    stage00_download,
    stage01_normalize,
    stage02_route,
    stage03_seed,
    stage05_scale,
    stage06_tool_agent,
    stage07_qpoints_rubrics,
    stage08_safety,
    stage09_export,
    stage11_make_splits,
    stage15_eval,
)


if __name__ == "__main__":
    args = base_parser("Run the MedEnvScale MedAgentGym 7-axis pipeline").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    stage00_download(cfg, limit=args.limit)
    stage01_normalize(cfg, limit=args.limit)
    stage02_route(cfg, limit=args.limit, llm_mode=args.llm_mode)
    stage03_seed(cfg, limit=args.limit)
    stage05_scale(cfg, limit=args.limit, llm_mode=args.llm_mode, sample_seed=args.sample_seed)
    stage06_tool_agent(cfg, limit=args.limit, llm_mode=args.llm_mode)
    stage07_qpoints_rubrics(cfg, limit=args.limit, llm_mode=args.llm_mode)
    stage08_safety(cfg)
    stage09_export(cfg)
    stage11_make_splits(cfg)
    metrics = stage15_eval(cfg)
    print(
        "Pipeline complete: "
        f"envs={metrics['num_environments']} | dpo_pairs={metrics['num_dpo_pairs']} | "
        f"mean_hidden_tests={metrics['mean_hidden_tests_per_env']}"
    )
