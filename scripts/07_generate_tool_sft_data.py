from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage07_tool_sft_data


if __name__ == "__main__":
    parser = base_parser("Generate Stage07 tool-calling SFT trajectories")
    parser.add_argument(
        "--user_feedback",
        action="store_true",
        help="Append a user continuation message after public submit_final_code preflight failures.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage07_tool_sft_data(
        cfg,
        limit=args.limit,
        llm_mode=args.llm_mode,
        user_feedback=args.user_feedback,
        resume=args.resume or args.resume_stage05,
        parallel_workers=args.workers,
    )
    manifest = result["manifest"]
    print(
        f"Tool SFT samples: {manifest['num_samples']}, "
        f"rejected: {manifest['num_rejected']}, "
        f"teacher: {manifest['teacher_output_slug']}, "
        f"splits: {manifest['splits']}"
    )
