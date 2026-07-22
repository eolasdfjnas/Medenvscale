from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import (
    load_config,
    stage00_download,
    stage01_normalize,
    stage02_route,
    stage03_seed,
    stage05_scale,
    stage05_5_assign_splits,
    stage06_tool_agent,
    stage07_tool_sft_data,
)


if __name__ == "__main__":
    args = base_parser("Run the MedEnvScale MedAgentGym 7-axis pipeline").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    resume = args.resume or args.resume_stage05
    stage00_download(cfg, limit=args.limit, llm_mode=args.llm_mode, parallel_workers=args.workers, resume=resume)
    stage01_normalize(cfg, limit=args.limit, resume=resume)
    stage02_route(cfg, limit=args.limit, llm_mode=args.llm_mode, parallel_workers=args.workers, resume=resume)
    stage03_seed(cfg, limit=args.limit, resume=resume)
    stage05_scale(
        cfg,
        limit=args.limit,
        llm_mode=args.llm_mode,
        sample_seed=args.sample_seed,
        parallel_workers=args.workers,
        resume=resume,
    )
    stage05_5_assign_splits(cfg, resume=resume)
    stage06_tool_agent(cfg, limit=args.limit, llm_mode=args.llm_mode)
    sft_result = stage07_tool_sft_data(cfg, limit=args.limit, llm_mode=args.llm_mode)
    manifest = sft_result["manifest"]
    print(
        "Pipeline complete through Stage07 tool SFT data: "
        f"samples={manifest['num_samples']} | rejected={manifest['num_rejected']} | "
        f"splits={manifest['splits']}"
    )
