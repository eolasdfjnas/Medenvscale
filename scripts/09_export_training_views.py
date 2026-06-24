from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage09_export


if __name__ == "__main__":
    args = base_parser("Export training views").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage09_export(cfg)
    print(
        f"SFT: {len(result['sft'])}, preference: {len(result['preference'])}, "
        f"PRM: {len(result['prm'])}, RLVR: {len(result['rlvr'])}"
    )
