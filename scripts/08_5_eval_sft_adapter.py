from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage08_5_eval_sft


if __name__ == "__main__":
    parser = base_parser("Evaluate the Stage08 SFT LoRA adapter with Stage06-style tool-agent oracle checks")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="dev",
        help="Stage05_5 split to evaluate. Defaults to dev.",
    )
    parser.add_argument(
        "--train_config",
        default=None,
        help="SFT training config used to locate the base model and adapter. Defaults to configs/<dataset>/train_sft.yaml.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Override the base local Hugging Face causal LM path.",
    )
    parser.add_argument(
        "--sft_adapter",
        default=None,
        help="Override the Stage08 SFT LoRA adapter path.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Rerun retryable LLM/local generation failures saved in the Stage08_5 output directory.",
    )
    parser.add_argument(
        "--user_feedback",
        action="store_true",
        help="Append a user continuation message after public submit_final_code preflight failures.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage08_5_eval_sft(
        cfg,
        limit=args.limit,
        split=args.split,
        train_config=args.train_config,
        llm_mode=args.llm_mode,
        model_path=args.model_path,
        sft_adapter=args.sft_adapter,
        retry_failed=args.retry_failed,
        user_feedback=args.user_feedback,
        resume=args.resume or args.resume_stage05,
        parallel_workers=args.workers,
    )
    summary = result.get("summary", {})
    print(
        f"Stage08_5 SFT eval: split={summary.get('split')}, runs={len(result['runs'])}, "
        f"pass_rate={summary.get('sample_pass_rate', 0.0)}, "
        f"case_pass_rate={summary.get('case_pass_rate', 0.0)}, "
        f"adapter={summary.get('sft_adapter')}"
    )
