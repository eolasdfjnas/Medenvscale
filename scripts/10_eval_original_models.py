from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage10_original_model_eval


if __name__ == "__main__":
    parser = base_parser("Evaluate base and trained adapters on original Stage00 executable tasks")
    parser.add_argument("--model_path", default=None, help="Base local Hugging Face causal LM path.")
    parser.add_argument("--sft_adapter", default=None, help="Stage08 SFT LoRA adapter path.")
    parser.add_argument("--rl_adapter", default=None, help="Stage09 RL LoRA adapter path.")
    parser.add_argument("--no_sft", action="store_true", help="Do not evaluate the SFT adapter.")
    parser.add_argument("--no_rl", action="store_true", help="Do not evaluate the RL adapter.")
    parser.add_argument("--retry_failed", action="store_true", help="Rerun retryable LLM/local generation failures.")
    parser.add_argument("--user_feedback", action="store_true", help="Append repair feedback after public final-code preflight failures.")
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage10_original_model_eval(
        cfg,
        limit=args.limit,
        llm_mode=args.llm_mode,
        model_path=args.model_path,
        sft_adapter=args.sft_adapter,
        rl_adapter=args.rl_adapter,
        eval_sft=not args.no_sft,
        eval_rl=not args.no_rl,
        retry_failed=args.retry_failed,
        user_feedback=args.user_feedback,
        resume=args.resume or args.resume_stage05,
        parallel_workers=args.workers,
    )
    comparison = result["comparison"]
    print(f"Stage10 original model eval: envs={comparison['num_original_envs']}")
    for row in comparison["models"]:
        print(
            f"  {row['model_label']}: sample_pass_rate={row['sample_pass_rate']}, "
            f"case_pass_rate={row['case_pass_rate']}, output_dir={row['output_dir']}"
        )
