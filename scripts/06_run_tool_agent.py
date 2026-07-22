from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage06_tool_agent


if __name__ == "__main__":
    parser = base_parser("Run tool-calling coding agent on scaled environments")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Stage05_5 split to evaluate. Defaults to all for backward compatibility.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Rerun retryable Stage06 LLM/network failures saved in the model output directory.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to a local Hugging Face causal LM checkpoint for Stage06 local evaluation.",
    )
    parser.add_argument(
        "--user_feedback",
        action="store_true",
        help="Append a user continuation message after public submit_final_code preflight failures.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage06_tool_agent(
        cfg,
        limit=args.limit,
        llm_mode=args.llm_mode,
        retry_failed=args.retry_failed,
        model_path=args.model_path,
        user_feedback=args.user_feedback,
        split=args.split,
        resume=args.resume or args.resume_stage05,
        parallel_workers=args.workers,
    )
    summary = result.get("summary", {})
    print(
        f"Agent runs: {len(result['runs'])}, split={summary.get('split', args.split)}, "
        f"traces: {len(result['traces'])}, eval rows: {len(result['eval_report'])}, "
        f"pass_rate: {summary.get('sample_pass_rate', 0.0)}, "
        f"case_pass_rate: {summary.get('case_pass_rate', 0.0)}"
    )
