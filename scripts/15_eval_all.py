from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage15_eval


if __name__ == "__main__":
    args = base_parser("Evaluate exported data").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    metrics = stage15_eval(cfg)
    print(metrics)
