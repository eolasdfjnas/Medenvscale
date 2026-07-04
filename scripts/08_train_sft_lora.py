from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage08_train_sft


if __name__ == "__main__":
    parser = base_parser("Train Stage08 tool SFT LoRA adapter")
    parser.add_argument("--train_config", default=None, help="Training config path, defaults to configs/<dataset>/train_sft.yaml.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--teacher_slug", default=None, help="Stage07 teacher output slug, e.g. xopqwen35v35b.")
    parser.add_argument("--dry_run", action="store_true", help="Prepare data and manifest without loading model weights.")
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage08_train_sft(
        cfg,
        train_config=args.train_config,
        max_steps=args.max_steps,
        model_name_or_path=args.model_name_or_path,
        teacher_slug=args.teacher_slug,
        dry_run=args.dry_run,
        resume=args.resume or args.resume_stage05,
    )
    print(
        f"Stage08 SFT: status={result['status']}, train={result['num_train_samples']}, "
        f"eval={result['num_eval_samples']}, output_dir={result['output_dir']}"
    )
