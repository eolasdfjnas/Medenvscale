from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage05_5_assign_splits


if __name__ == "__main__":
    args = base_parser("Assign train/dev/test splits to Stage05 clean environments").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage05_5_assign_splits(cfg, resume=args.resume or args.resume_stage05)
    manifest = result["manifest"]
    print(
        "Stage05_5 splits assigned: "
        f"envs={manifest['num_envs']}, groups={manifest['num_groups']}, "
        f"env_counts={manifest['env_counts']}, output={result['output_path']}"
    )
