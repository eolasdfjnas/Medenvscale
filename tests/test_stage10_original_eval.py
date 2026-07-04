from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from medenvscale.config import AppConfig
from medenvscale.pipeline_ops import stage10_original_model_eval
from medenvscale.utils import ensure_dir, load_yaml, write_jsonl


class Stage10OriginalEvalTests(unittest.TestCase):
    def test_stage10_evaluates_original_stage00_rows_with_mock_model(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cfg = self._build_temp_config(root)
        raw_row = {
            "task_id": "demo_task",
            "source_split": "train",
            "idx": "1",
            "problem": "Write solve() that returns five.",
            "context": "def solve():\n    pass\n",
            "signature": "def solve()",
            "solution": "def solve():\n    return 5",
            "code": "def solve():\n    return 5\n",
            "ground_truth_output_signature": {
                "return_value": 5,
                "return_type": "int",
                "stdout_contains": ["5"],
            },
            "seed_execution_case": {
                "case_id": "seed_case_main",
                "setup_code": "",
                "call_code": "result = solve()\nprint(result)",
                "expected_output_signature": {
                    "return_value": 5,
                    "return_type": "int",
                    "stdout_contains": ["5"],
                },
            },
            "execution_status": "pass",
        }
        write_jsonl(cfg.root / cfg.values["dataset"]["local_raw_paths"]["test"], [raw_row])

        result = stage10_original_model_eval(
            cfg,
            llm_mode="mock",
            eval_sft=False,
            eval_rl=False,
        )

        comparison = result["comparison"]
        self.assertEqual(comparison["num_original_envs"], 1)
        self.assertEqual(comparison["ground_truth_source"], "stage00_seed_execution_case_expected_output_signature")
        self.assertEqual(comparison["models"][0]["model_label"], "base")
        self.assertEqual(comparison["models"][0]["sample_pass_rate"], 1.0)
        self.assertTrue((cfg.output_dirs["result"] / "10" / "original_model_eval" / "comparison_summary.json").exists())

    def _build_temp_config(self, root: Path) -> AppConfig:
        values = copy.deepcopy(load_yaml(root / "configs" / "biocoder" / "medagentgym_pilot.yaml"))
        temp_root = Path(tempfile.mkdtemp(prefix="medenvscale-stage10-"))
        llm_values = copy.deepcopy(load_yaml(root / "configs" / "llm.yaml"))
        values["dataset"]["local_raw_path"] = str(temp_root / "raw" / "train_tasks_raw.jsonl")
        values["dataset"]["local_raw_paths"] = {
            "train": str(temp_root / "raw" / "train_tasks_raw.jsonl"),
            "test": str(temp_root / "raw" / "test_tasks_raw.jsonl"),
        }
        values["dataset"]["metadata_path"] = str(temp_root / "raw" / "prepare_meta.json")
        values["output"]["raw_dir"] = str(temp_root / "raw")
        values["output"]["interim_dir"] = str(temp_root / "interim")
        values["output"]["processed_dir"] = str(temp_root / "processed")
        values["output"]["split_dir"] = str(temp_root / "splits")
        values["output"]["result_dir"] = str(temp_root / "result")
        llm_values["cache"]["dir"] = str(temp_root / "cache" / "llm")
        llm_values["trace"]["path"] = str(temp_root / "processed" / "generation_trace.jsonl")
        cfg = AppConfig(root=root, values=values, llm_values=llm_values, dataset_name="biocoder")
        for path in cfg.output_dirs.values():
            ensure_dir(path)
        ensure_dir(cfg.root / cfg.llm_values["cache"]["dir"])
        return cfg


if __name__ == "__main__":
    unittest.main()
