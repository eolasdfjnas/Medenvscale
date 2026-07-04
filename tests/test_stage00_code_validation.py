from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medenvscale.config import AppConfig
from medenvscale.ingest.code_execution import _compare_seed_case_to_ground_truth
from medenvscale.pipeline_ops import raw_path, raw_rejected_path, raw_test_path, stage00_download
from medenvscale.utils import ensure_dir, load_yaml, read_jsonl, write_jsonl


class Stage00CodeValidationTests(unittest.TestCase):
    def _build_temp_config(self, root: Path, train_path: Path) -> AppConfig:
        values = copy.deepcopy(load_yaml(root / "configs" / "biocoder" / "medagentgym_pilot.yaml"))
        llm_values = copy.deepcopy(load_yaml(root / "configs" / "llm.yaml"))
        temp_root = Path(tempfile.mkdtemp(prefix="medenvscale-stage00-"))
        values["dataset"]["task_files"] = {"train": str(train_path)}
        values["dataset"]["local_raw_path"] = str(temp_root / "raw" / "medagentgym_tasks_raw.jsonl")
        values["dataset"]["metadata_path"] = str(temp_root / "raw" / "prepare_meta.json")
        values["dataset"]["code_execution"]["backend"] = "local"
        values["output"]["raw_dir"] = str(temp_root / "raw")
        values["output"]["interim_dir"] = str(temp_root / "interim")
        values["output"]["processed_dir"] = str(temp_root / "processed")
        values["output"]["split_dir"] = str(temp_root / "splits")
        values["output"]["result_dir"] = str(temp_root / "result")
        values["output"]["experiment_dir"] = str(temp_root / "experiments")
        llm_values["cache"]["dir"] = str(temp_root / "cache" / "llm")
        llm_values["trace"]["path"] = str(temp_root / "processed" / "generation_trace.jsonl")
        cfg = AppConfig(root=root, values=values, llm_values=llm_values, dataset_name="biocoder")
        for path in cfg.output_dirs.values():
            ensure_dir(path)
        ensure_dir(cfg.root / cfg.llm_values["cache"]["dir"])
        return cfg

    def test_stage00_writes_train_and_test_raw_outputs(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="medenvscale-stage00-src-"))
        train_path = temp_dir / "train.jsonl"
        test_path = temp_dir / "test.jsonl"
        row = {
            "idx": "split_case",
            "problem": "Write add.",
            "solution": "def add(a, b):\n    return a + b",
            "context": "<<insert solution here>>",
            "signature": "def add(a, b)",
            "code": "def add(a, b):\n    return a + b\n\nanswer = add(2, 3)\nprint(answer)\n",
        }
        write_jsonl(train_path, [{**row, "idx": "train_case"}])
        write_jsonl(test_path, [{**row, "idx": "test_case"}])
        cfg = self._build_temp_config(root, train_path)
        cfg.values["dataset"]["task_files"]["test"] = str(test_path)
        cfg.values["dataset"]["local_raw_paths"] = {
            "train": str(Path(cfg.values["dataset"]["local_raw_path"])),
            "test": str(Path(cfg.values["output"]["raw_dir"]) / "test_tasks_raw.jsonl"),
        }

        report = stage00_download(cfg, llm_mode="mock")

        self.assertEqual(report["rows_written"], 2)
        self.assertEqual(len(read_jsonl(raw_path(cfg))), 1)
        self.assertEqual(len(read_jsonl(raw_test_path(cfg))), 1)
        self.assertEqual(read_jsonl(raw_path(cfg))[0]["source_split"], "train")
        self.assertEqual(read_jsonl(raw_test_path(cfg))[0]["source_split"], "test")

    def test_stage00_keeps_executable_code_and_ground_truth_signature(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="medenvscale-stage00-src-"))
        train_path = temp_dir / "train.jsonl"
        rows = [
            {
                "idx": "good_1",
                "problem": "Write add.",
                "solution": "def add(a, b):\n    return a + b",
                "context": "<<insert solution here>>",
                "signature": "def add(a, b)",
                "code": "def add(a, b):\n    return a + b\n\nanswer = add(2, 3)\nprint(answer)\n",
            }
        ]
        write_jsonl(train_path, rows)
        cfg = self._build_temp_config(root, train_path)

        report = stage00_download(cfg, llm_mode="mock")

        self.assertEqual(report["rows_written"], 1)
        prepared = read_jsonl(raw_path(cfg))
        self.assertEqual(len(prepared), 1)
        row = prepared[0]
        self.assertEqual(row["execution_status"], "pass")
        self.assertEqual(row["ground_truth_output_signature"]["stdout"], "5")
        self.assertEqual(row["ground_truth_output_signature"]["return_value"], 5)
        self.assertEqual(row["seed_case_audit"]["status"], "pass")
        self.assertTrue(row["seed_execution_case"]["validated_against_ground_truth"])
        self.assertEqual(row["seed_execution_case"]["expected_output_signature"]["stdout_contains"], ["5"])
        self.assertIn("result =", row["seed_execution_case"]["call_code"])
        self.assertEqual(row["repair_attempts"], 0)
        self.assertFalse(row["repair_succeeded"])
        self.assertEqual(read_jsonl(raw_rejected_path(cfg)), [])

    def test_stage00_repairs_broken_code_and_preserves_wrong_code(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="medenvscale-stage00-src-"))
        train_path = temp_dir / "train.jsonl"
        broken_code = "def add(a, b):\n    return a + b\n\nanswer = add(2, 3)\nprin(answer)\n"
        rows = [
            {
                "idx": "broken_1",
                "problem": "Write add.",
                "solution": "def add(a, b):\n    return a + b",
                "context": "<<insert solution here>>",
                "signature": "def add(a, b)",
                "code": broken_code,
            }
        ]
        write_jsonl(train_path, rows)
        cfg = self._build_temp_config(root, train_path)

        report = stage00_download(cfg, llm_mode="mock")

        self.assertEqual(report["rows_written"], 1)
        prepared = read_jsonl(raw_path(cfg))
        self.assertEqual(len(prepared), 1)
        row = prepared[0]
        self.assertEqual(row["wrong_code"], broken_code.strip())
        self.assertIn("print(answer)", row["code"])
        self.assertEqual(row["repair_attempts"], 1)
        self.assertTrue(row["repair_succeeded"])
        self.assertEqual(row["ground_truth_output_signature"]["stdout"], "5")
        self.assertEqual(row["seed_case_audit"]["status"], "pass")
        self.assertTrue(row["seed_execution_case"]["validated_against_ground_truth"])
        self.assertEqual(read_jsonl(raw_rejected_path(cfg)), [])

    def test_stage00_builds_seed_case_from_main_flow(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="medenvscale-stage00-src-"))
        train_path = temp_dir / "train.jsonl"
        rows = [
            {
                "idx": "main_case_1",
                "problem": "Compute reaction charge and print two example outputs.",
                "solution": "def reaction_charge(reaction, compound_charge):\n    return 0.0",
                "context": "<<insert solution here>>",
                "signature": "def reaction_charge(reaction, compound_charge)",
                "code": "class Compound:\n    def __init__(self, name):\n        self.name = name\n\nclass Reaction:\n    def __init__(self, compounds):\n        self.compounds = compounds\n\ndef reaction_charge(reaction, compound_charge):\n    charge_sum = 0.0\n    for compound, value in reaction.compounds:\n        charge_sum += compound_charge[compound.name] * float(value)\n    return charge_sum\n\ndef main():\n    h_plus = Compound('H+')\n    oh_minus = Compound('OH-')\n    na_plus = Compound('Na+')\n    reaction1 = Reaction([(h_plus, 2), (oh_minus, 1)])\n    reaction2 = Reaction([(na_plus, 1), (oh_minus, 1)])\n    compound_charge = {'H+': 1.0, 'OH-': -1.0, 'Na+': 1.0}\n    net_charge1 = reaction_charge(reaction1, compound_charge)\n    print(f'Net charge of reaction1: {net_charge1}')\n    net_charge2 = reaction_charge(reaction2, compound_charge)\n    print(f'Net charge of reaction2: {net_charge2}')\n\nif __name__ == '__main__':\n    main()\n",
            }
        ]
        write_jsonl(train_path, rows)
        cfg = self._build_temp_config(root, train_path)

        stage00_download(cfg, llm_mode="mock")

        prepared = read_jsonl(raw_path(cfg))
        row = prepared[0]
        self.assertEqual(row["seed_case_audit"]["status"], "pass")
        self.assertIn("compound_charge", row["seed_execution_case"]["setup_code"])
        self.assertIn("reaction_charge", row["seed_execution_case"]["call_code"])
        self.assertIn("result =", row["seed_execution_case"]["call_code"])
        self.assertEqual(
            row["seed_execution_case"]["expected_output_signature"]["stdout_contains"],
            ["Net charge of reaction1: 1.0", "Net charge of reaction2: 0.0"],
        )

    def test_stage00_rejects_seed_case_ground_truth_mismatch(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="medenvscale-stage00-src-"))
        train_path = temp_dir / "train.jsonl"
        rows = [
            {
                "idx": "seed_mismatch",
                "problem": "Write add.",
                "solution": "def add(a, b):\n    return a + b",
                "context": "<<insert solution here>>",
                "signature": "def add(a, b)",
                "code": "def add(a, b):\n    return a + b\n\nanswer = add(2, 3)\nprint(answer)\n",
            }
        ]
        write_jsonl(train_path, rows)
        cfg = self._build_temp_config(root, train_path)

        with patch(
            "medenvscale.ingest.code_execution.build_seed_execution_case",
            return_value=(
                None,
                {
                    "status": "soft_fail",
                    "failure_reason": "seed_case_ground_truth_mismatch",
                    "mismatch_reasons": ["stdout_mismatch"],
                    "validated_against_ground_truth": False,
                },
            ),
        ):
            report = stage00_download(cfg, llm_mode="mock")

        self.assertEqual(report["rows_written"], 0)
        self.assertEqual(report["accepted_rows"], 0)
        self.assertEqual(report["rejected_rows"], 1)
        self.assertEqual(read_jsonl(raw_path(cfg)), [])
        rejected = read_jsonl(raw_rejected_path(cfg))
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "seed_case_admission_failed")
        self.assertEqual(rejected[0]["seed_case_failure_reason"], "seed_case_ground_truth_mismatch")
        self.assertEqual(rejected[0]["seed_case_mismatch_reasons"], ["stdout_mismatch"])

    def test_seed_case_stdout_compare_ignores_literal_dict_order(self) -> None:
        matched, reasons = _compare_seed_case_to_ground_truth(
            {
                "return_value": None,
                "stdout": "Extended model reactions: []\nWeights: {'c1_db_rxn': 10, 'c2_db_rxn': 5, 'c1_e_tp': 3, 'e_ex': 2}",
                "file_artifacts": [],
            },
            {
                "return_value": None,
                "stdout": "Extended model reactions: []\nWeights: {'c2_db_rxn': 5, 'c1_db_rxn': 10, 'c1_e_tp': 3, 'e_ex': 2}",
                "file_artifacts": [],
            },
        )
        self.assertTrue(matched)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
