from __future__ import annotations

import json
from pathlib import Path

from medenvscale.config import load_training_config


def run_train_dpo(config_path: str, max_steps: int | None = None, dataset: str | None = None) -> dict:
    root, cfg = load_training_config(config_path, dataset=dataset)
    result = {
        "trainer": cfg["trainer"],
        "dataset_path": cfg["dataset_path"],
        "output_dir": str(root / cfg["output_dir"]),
        "max_steps": max_steps or cfg["max_steps"],
        "status": "configured_only",
    }
    output_dir = root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("train_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
