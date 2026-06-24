from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage05_scale


if __name__ == "__main__":
    args = base_parser("Apply scaling operators").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage05_scale(cfg, limit=args.limit, llm_mode=args.llm_mode, sample_seed=args.sample_seed)
    print(f"Scaled environments: {len(rows)}")
