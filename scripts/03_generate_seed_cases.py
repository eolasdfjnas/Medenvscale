from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage03_seed


if __name__ == "__main__":
    args = base_parser("Generate seed cases").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage03_seed(cfg, limit=args.limit, resume=args.resume or args.resume_stage05)
    print(f"Seed cases: {len(rows)}")
