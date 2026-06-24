from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage08_safety


if __name__ == "__main__":
    args = base_parser("Build safety gates").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage08_safety(cfg)
    print(f"Safety gates: {len(rows)}")
