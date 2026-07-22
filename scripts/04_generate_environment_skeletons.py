from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage04_skeleton


if __name__ == "__main__":
    args = base_parser("Generate environment skeletons").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage04_skeleton(cfg, limit=args.limit, llm_mode=args.llm_mode, resume=args.resume or args.resume_stage05)
    print(f"Environment skeletons: {len(rows)}")
