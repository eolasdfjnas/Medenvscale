from __future__ import annotations

from _common import base_parser
from medenvscale.pipeline_ops import load_config, stage00_download


if __name__ == "__main__":
    args = base_parser("Download or prepare MedQA English data").parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    report = stage00_download(cfg, limit=args.limit)
    print(
        "Prepared raw rows: "
        f"{report['rows_written']} | dataset={report['dataset_name']} | "
        f"method={report['download_method']} | splits={','.join(report['splits_requested'])}"
    )
