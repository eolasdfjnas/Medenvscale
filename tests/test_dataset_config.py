from __future__ import annotations

import unittest
from pathlib import Path

from medenvscale.config import load_app_config, load_training_config


class DatasetConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent.parent

    def test_load_app_config_scopes_paths_by_dataset(self) -> None:
        cfg = load_app_config(self.root / "configs" / "biocoder" / "medagentgym_pilot.yaml", dataset="biocoder")

        self.assertEqual(cfg.dataset_name, "biocoder")
        self.assertEqual(cfg.values["dataset"]["active_dataset"], "biocoder")
        self.assertEqual(cfg.values["dataset"]["task_files"]["train"], "/archive/zengjiaqi/dataset/medagentgym/biocoder/train_tasks.jsonl")
        self.assertEqual(cfg.values["dataset"]["local_raw_path"], "data/biocoder/raw/train_tasks_raw.jsonl")
        self.assertEqual(cfg.values["dataset"]["local_raw_paths"]["train"], "data/biocoder/raw/train_tasks_raw.jsonl")
        self.assertEqual(cfg.values["dataset"]["local_raw_paths"]["test"], "data/biocoder/raw/test_tasks_raw.jsonl")
        self.assertEqual(cfg.output_dirs["raw"], self.root / "data" / "biocoder" / "raw")
        self.assertEqual(cfg.output_dirs["interim"], self.root / "data" / "biocoder" / "interim")
        self.assertEqual(cfg.output_dirs["processed"], self.root / "data" / "biocoder" / "processed")
        self.assertEqual(cfg.output_dirs["splits"], self.root / "data" / "biocoder" / "splits")
        self.assertEqual(cfg.output_dirs["result"], self.root / "result" / "biocoder")
        self.assertEqual(cfg.output_dirs["experiments"], self.root / "experiments" / "biocoder")
        self.assertEqual(cfg.llm_values["cache"]["dir"], ".cache/llm/biocoder")
        self.assertEqual(cfg.llm_values["trace"]["path"], "data/biocoder/processed/generation_trace.jsonl")

    def test_nested_dataset_config_uses_default_dataset(self) -> None:
        cfg = load_app_config(self.root / "configs" / "biocoder" / "medagentgym_pilot.yaml")

        self.assertEqual(cfg.dataset_name, "biocoder")
        self.assertEqual(cfg.values["dataset"]["task_files"]["test"], "/archive/zengjiaqi/dataset/medagentgym/biocoder/test_tasks.jsonl")
        self.assertEqual(cfg.values["dataset"]["local_raw_paths"]["test"], "data/biocoder/raw/test_tasks_raw.jsonl")
        self.assertEqual(cfg.output_dirs["result"], self.root / "result" / "biocoder")

    def test_training_config_scopes_dataset_paths(self) -> None:
        root, cfg = load_training_config(self.root / "configs" / "biocoder" / "train_sft.yaml", dataset="biocoder")

        self.assertEqual(root, self.root)
        self.assertEqual(cfg["dataset_path"], "data/biocoder/splits/tool_sft_train.jsonl")
        self.assertEqual(cfg["eval_path"], "data/biocoder/splits/tool_sft_dev.jsonl")
        self.assertEqual(cfg["output_dir"], "experiments/biocoder/tool_sft_lora")


if __name__ == "__main__":
    unittest.main()
