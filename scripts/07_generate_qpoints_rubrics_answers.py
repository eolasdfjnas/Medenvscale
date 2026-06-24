from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage07_qpoints_rubrics


if __name__ == "__main__":
    args = base_parser("Generate question points, rubrics, and answers").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage07_qpoints_rubrics(cfg, limit=args.limit, llm_mode=args.llm_mode)
    print(
        f"Updated envs: {len(result['environments'])}, "
        f"question points: {len(result['question_points'])}, "
        f"rubrics: {len(result['rubrics'])}"
    )
