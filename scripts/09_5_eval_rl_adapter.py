from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage09_5_eval_rl_adapter


if __name__ == "__main__":
    parser = base_parser("Evaluate the Stage09 RL LoRA adapter as a tool agent and score the final submitted code")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="test",
        help="Stage05_5 split to evaluate. Defaults to test.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Override the base local Hugging Face causal LM path.",
    )
    parser.add_argument(
        "--rl_adapter",
        default=None,
        help="Override the Stage09 RL LoRA adapter path. Defaults to the newest adapter under experiments/<dataset>/tool_rl_grpo_lora.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Rerun retryable LLM/local generation failures saved in the Stage09_5 output directory.",
    )
    parser.add_argument(
        "--user_feedback",
        action="store_true",
        help="Append a user continuation message after public submit_final_code preflight failures.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage09_5_eval_rl_adapter(
        cfg,
        limit=args.limit,
        split=args.split,
        llm_mode=args.llm_mode,
        model_path=args.model_path,
        rl_adapter=args.rl_adapter,
        retry_failed=args.retry_failed,
        user_feedback=args.user_feedback,
        resume=args.resume or args.resume_stage05,
        parallel_workers=args.workers,
    )
    summary = result.get("summary", {})
    print(
        f"Stage09_5 RL eval: split={summary.get('split')}, runs={len(result['runs'])}, "
        f"pass_rate={summary.get('sample_pass_rate', 0.0)}, "
        f"case_pass_rate={summary.get('case_pass_rate', 0.0)}, "
        f"adapter={summary.get('rl_adapter')}"
    )
