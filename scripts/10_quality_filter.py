from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage10_quality_filter


if __name__ == "__main__":
    args = base_parser("Quality filter").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage10_quality_filter(cfg)
    print(f"Filtering report rows: {len(rows)}")
