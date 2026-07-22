from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage00_download


if __name__ == "__main__":
    args = base_parser("Prepare MedAgentGym task files").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    report = stage00_download(
        cfg,
        limit=args.limit,
        llm_mode=args.llm_mode,
        parallel_workers=args.workers,
        resume=args.resume or args.resume_stage05,
    )
    print(
        "Prepared raw rows: "
        f"{report['rows_written']} | dataset={report['dataset_name']} | "
        f"method={report['download_method']} | splits={','.join(report['splits_requested'])}"
    )
