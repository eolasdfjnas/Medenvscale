from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage02_route


if __name__ == "__main__":
    args = base_parser("Route MedAgentGym tasks into domain and capability labels").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    rows = stage02_route(cfg, limit=args.limit, llm_mode=args.llm_mode)
    print(f"Routed rows: {len(rows)}")
