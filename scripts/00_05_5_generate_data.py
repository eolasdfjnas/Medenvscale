from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from _common import base_parser
from medenvscale.pipeline_ops import (
    load_config,
    stage00_download,
    stage01_normalize,
    stage02_route,
    stage03_seed,
    stage04_skeleton,
    stage05_5_assign_splits,
    stage05_scale,
)


StageFn = Callable[..., Any]


STAGES: list[tuple[str, str, StageFn]] = [
    ("00", "prepare raw MedAgentGym rows", stage00_download),
    ("01", "normalize tasks", stage01_normalize),
    ("02", "route tasks", stage02_route),
    ("03", "build seed environments", stage03_seed),
    ("04", "publish environment skeletons", stage04_skeleton),
    ("05", "apply scaling operators", stage05_scale),
    ("05_5", "assign train/dev/test splits", stage05_5_assign_splits),
]


def main() -> None:
    parser = base_parser("Run MedEnvScale data generation stages 00 through 05_5")
    parser.add_argument(
        "--stop_stage",
        choices=[stage_id for stage_id, _, _ in STAGES],
        default="05_5",
        help="Stop after this stage. Defaults to 05_5.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)

    print("Data generation pipeline: Stage00 -> Stage05_5")
    print(f"config={args.config} | dataset={args.dataset or cfg.dataset_name or 'default'} | stop_stage={args.stop_stage}")
    if args.limit is not None:
        print(f"limit={args.limit}")
    if args.sample_seed is not None:
        print(f"sample_seed={args.sample_seed}")
    if args.llm_mode is not None:
        print(f"llm_mode={args.llm_mode}")
    if args.workers is not None:
        print(f"workers={args.workers} (applies to Stage00, Stage02, and Stage05)")
    if args.resume:
        print("resume=True")
    if args.resume_stage05:
        print("resume_stage05=True (deprecated; prefer --resume)")

    summaries = []
    pipeline_start = time.time()
    for stage_id, description, stage_fn in STAGES:
        stage_label = f"Stage{stage_id}"
        print(f"\n[START] {stage_label}: {description}", flush=True)
        start = time.time()
        result = _run_stage(stage_fn, cfg, args)
        elapsed = time.time() - start
        summary = _summarize_result(result)
        summaries.append((stage_label, summary, elapsed))
        print(f"[DONE]  {stage_label}: {summary} | elapsed={_format_seconds(elapsed)}", flush=True)
        if stage_id == args.stop_stage:
            break

    print("\nData generation summary:")
    for stage_label, summary, elapsed in summaries:
        print(f"- {stage_label}: {summary} | elapsed={_format_seconds(elapsed)}")
    print(f"Completed through {summaries[-1][0]} | total_elapsed={_format_seconds(time.time() - pipeline_start)}")


def _run_stage(stage_fn: StageFn, cfg: Any, args: Any) -> Any:
    kwargs: dict[str, Any] = {}
    arg_names = stage_fn.__code__.co_varnames
    if "limit" in arg_names:
        kwargs["limit"] = args.limit
    if "llm_mode" in arg_names:
        kwargs["llm_mode"] = args.llm_mode
    if "sample_seed" in arg_names:
        kwargs["sample_seed"] = args.sample_seed
    if "parallel_workers" in arg_names:
        kwargs["parallel_workers"] = args.workers
    if "resume" in arg_names:
        kwargs["resume"] = args.resume or args.resume_stage05
    return stage_fn(cfg, **kwargs)


def _summarize_result(result: Any) -> str:
    if isinstance(result, list):
        return f"rows={len(result)}"
    if isinstance(result, dict):
        if "manifest" in result and isinstance(result["manifest"], dict):
            manifest = result["manifest"]
            parts = []
            for key in ("num_envs", "num_groups", "env_counts", "splits"):
                if key in manifest:
                    parts.append(f"{key}={manifest[key]}")
            if parts:
                return ", ".join(parts)
        parts = []
        for key in ("rows_written", "num_samples", "num_rejected", "output_path"):
            if key in result:
                parts.append(f"{key}={result[key]}")
        return ", ".join(parts) if parts else f"keys={len(result)}"
    return "done"


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
