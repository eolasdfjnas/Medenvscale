from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage09_rlvr_grpo


if __name__ == "__main__":
    parser = base_parser("Run Stage09 RLVR rollouts and optional GRPO LoRA training")
    parser.add_argument(
        "--rollout_only",
        action="store_true",
        help="Only collect RLVR rollouts and oracle rewards; do not train.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run TRL GRPOTrainer LoRA training. Does not collect Stage06-style rollouts unless --collect_rollouts is set.",
    )
    parser.add_argument(
        "--collect_rollouts",
        action="store_true",
        help="When used with --train, also collect Stage06-style tool-agent rollouts before TRL GRPO training.",
    )
    parser.add_argument(
        "--use_existing_rollouts",
        action="store_true",
        help="Reuse existing rl_rollouts.jsonl and reward_report.jsonl instead of collecting new rollouts.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="train",
        help="Stage05_5 split to use for Stage09. Use all only for debug or full-dataset analysis.",
    )
    parser.add_argument(
        "--eval_split",
        choices=["dev", "test", "all"],
        default=None,
        help="Optional Stage05_5 split used as GRPOTrainer eval_dataset during --train.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run GRPOTrainer eval every N training steps when --eval_split is set.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Base local Hugging Face causal LM path. Defaults to stage09_rlvr_grpo.base_model.",
    )
    parser.add_argument(
        "--sft_adapter",
        default=None,
        help="Stage08 SFT LoRA adapter path. Defaults to stage09_rlvr_grpo.sft_adapter.",
    )
    parser.add_argument(
        "--user_feedback",
        action="store_true",
        help="Append user repair feedback after public submit_final_code preflight failures.",
    )
    parser.add_argument(
        "--use_rubric_reward",
        action="store_true",
        help="Enable oracle-backed rubric_score as part of Stage09 reward.",
    )
    parser.add_argument(
        "--disable_rubric_reward",
        action="store_true",
        help="Disable rubric_score reward shaping even if enabled in config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rollout_only = bool(args.rollout_only or not args.train)
    result = stage09_rlvr_grpo(
        cfg,
        limit=args.limit,
        rollout_only=rollout_only,
        train=args.train,
        collect_rollouts=args.collect_rollouts,
        split=args.split,
        eval_split=args.eval_split,
        eval_steps=args.eval_steps,
        llm_mode=args.llm_mode,
        model_path=args.model_path,
        sft_adapter=args.sft_adapter,
        user_feedback=args.user_feedback,
        use_rubric_reward=False if args.disable_rubric_reward else (True if args.use_rubric_reward else None),
        resume=args.resume or args.resume_stage05,
        use_existing_rollouts=args.use_existing_rollouts,
    )
    summary = result["summary"]
    print(
        f"Stage09 RLVR/GRPO: rollouts={summary['num_rollouts']}, "
        f"mean_reward={summary['mean_reward']}, "
        f"sample_pass_rate={summary['sample_pass_rate']}, "
        f"case_pass_rate={summary['case_pass_rate']}, "
        f"policy={summary['policy_slug']}"
    )
