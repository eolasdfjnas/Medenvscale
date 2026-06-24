from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage11_make_splits


if __name__ == "__main__":
    args = base_parser("Make train/dev/test splits").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage11_make_splits(cfg)
    print({k: len(v) for k, v in result.items()})
