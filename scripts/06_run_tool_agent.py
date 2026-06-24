from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage06_tool_agent


if __name__ == "__main__":
    args = base_parser("Run tool-calling coding agent on scaled environments").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    result = stage06_tool_agent(cfg, limit=args.limit, llm_mode=args.llm_mode)
    summary = result.get("summary", {})
    print(
        f"Agent runs: {len(result['runs'])}, "
        f"traces: {len(result['traces'])}, eval rows: {len(result['eval_report'])}, "
        f"pass_rate: {summary.get('sample_pass_rate', 0.0)}, "
        f"case_pass_rate: {summary.get('case_pass_rate', 0.0)}"
    )
