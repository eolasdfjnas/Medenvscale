from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage01_normalize


if __name__ == "__main__":
    args = base_parser("Normalize MedQA English data").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage01_normalize(cfg, limit=args.limit)
    print(f"Normalized rows: {len(rows)}")
