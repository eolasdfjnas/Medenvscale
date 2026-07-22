from __future__ import annotations

import argparse

from _common import ROOT
from medenvscale.train.train_sft_lora import run_train_sft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT training manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None, help="Dataset subdirectory name, for example: biocoder")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--teacher_slug", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    result = run_train_sft(
        str((ROOT / args.config).resolve()) if not args.config.startswith(("C:\\", "/")) else args.config,
        args.max_steps,
        dataset=args.dataset,
        model_name_or_path=args.model_name_or_path,
        teacher_slug=args.teacher_slug,
        dry_run=args.dry_run,
    )
    print(result)
