from __future__ import annotations

import argparse

from _common import ROOT
from medenvscale.train.train_dpo_lora import run_train_dpo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DPO training manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None, help="Dataset subdirectory name, for example: biocoder")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    result = run_train_dpo(
        str((ROOT / args.config).resolve()) if not args.config.startswith(("C:\\", "/")) else args.config,
        args.max_steps,
        dataset=args.dataset,
    )
    print(result)
