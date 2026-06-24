from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from medenvscale.config import AppConfig
from medenvscale.demo_data import DEMO_MEDAGENTGYM_ROWS
from medenvscale.pipeline_ops import (
    artifact_admission_report_output_path,
    hidden_tests_clean_output_path,
    hidden_tests_quality_report_output_path,
    operator_realization_report_output_path,
    scaled_clean_output_path,
    scaled_raw_output_path,
    scaled_task_consistency_report_output_path,
    stage05_quality_report_output_path,
    stage01_normalize,
    stage02_route,
    stage03_seed,
    stage05_scale,
    stage06_qpoints_rubrics,
    stage07_safety,
    stage08_export,
)
from medenvscale.scaling.quality_filter import is_semantic_hidden_test
from medenvscale.utils import ensure_dir, load_yaml, read_jsonl, write_jsonl


class PipelineSmokeTests(unittest.TestCase):
    def _build_temp_config(self, root: Path) -> AppConfig:
        values = copy.deepcopy(load_yaml(root / "configs" / "biocoder" / "medagentgym_pilot.yaml"))
        temp_root = Path(tempfile.mkdtemp(prefix="medenvscale-smoke-"))
        llm_values = copy.deepcopy(load_yaml(root / "configs" / "llm.yaml"))
        values["dataset"]["local_raw_path"] = str(temp_root / "raw" / "medagentgym_tasks_raw.jsonl")
        values["dataset"]["metadata_path"] = str(temp_root / "raw" / "prepare_meta.json")
        values["output"]["raw_dir"] = str(temp_root / "raw")
        values["output"]["interim_dir"] = str(temp_root / "interim")
        values["output"]["processed_dir"] = str(temp_root / "processed")
        values["output"]["split_dir"] = str(temp_root / "splits")
        llm_values["cache"]["dir"] = str(temp_root / "cache" / "llm")
        llm_values["trace"]["path"] = str(temp_root / "processed" / "generation_trace.jsonl")
        cfg = AppConfig(root=root, values=values, llm_values=llm_values, dataset_name="biocoder")
        for path in cfg.output_dirs.values():
            ensure_dir(path)
        ensure_dir(cfg.root / cfg.llm_values["cache"]["dir"])
        return cfg

    def test_mock_pipeline_generates_scaled_envs(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cfg = self._build_temp_config(root)
        write_jsonl(cfg.root / cfg.values["dataset"]["local_raw_path"], DEMO_MEDAGENTGYM_ROWS[:5])
        normalized = stage01_normalize(cfg, limit=5)
        self.assertEqual(len(normalized), 5)
        routed = stage02_route(cfg, limit=5, llm_mode="mock")
        self.assertEqual(len(routed), 5)
        seeds = stage03_seed(cfg, limit=5)
        self.assertEqual(len(seeds), 5)
        environments = stage05_scale(cfg, limit=5, llm_mode="mock")
        self.assertEqual(len(environments), 20)
        stage06_qpoints_rubrics(cfg, limit=5, llm_mode="mock")
        report = stage07_safety(cfg)
        exports = stage08_export(cfg)
        self.assertEqual(len(report), 20)
        self.assertEqual(len(read_jsonl(scaled_raw_output_path(cfg))), 20)
        self.assertTrue((operator_realization_report_output_path(cfg)).exists())
        self.assertTrue(hidden_tests_quality_report_output_path(cfg).exists())
        self.assertTrue(scaled_task_consistency_report_output_path(cfg).exists())
        self.assertTrue(artifact_admission_report_output_path(cfg).exists())
        self.assertTrue(stage05_quality_report_output_path(cfg).exists())
        clean_envs = read_jsonl(scaled_clean_output_path(cfg))
        self.assertEqual(len(exports["sft"]), len(clean_envs))
        semantic_count = sum(
            1 for env in clean_envs for test in env.get("hidden_tests", []) if isinstance(test, dict) and is_semantic_hidden_test(test)
        )
        self.assertEqual(len(read_jsonl(hidden_tests_clean_output_path(cfg))), semantic_count)

    def test_stage05_limit_can_use_seeded_random_sampling(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cfg = self._build_temp_config(root)
        write_jsonl(cfg.root / cfg.values["dataset"]["local_raw_path"], DEMO_MEDAGENTGYM_ROWS[:5])
        stage01_normalize(cfg, limit=5)
        stage02_route(cfg, limit=5, llm_mode="mock")
        stage03_seed(cfg, limit=5)
        envs_a = stage05_scale(cfg, limit=1, llm_mode="mock", sample_seed=7)
        envs_b = stage05_scale(cfg, limit=1, llm_mode="mock", sample_seed=7)
        envs_c = stage05_scale(cfg, limit=1, llm_mode="mock", sample_seed=8)
        self.assertEqual(len(envs_a), 4)
        self.assertEqual([env.original_task_id for env in envs_a], [env.original_task_id for env in envs_b])
        self.assertNotEqual([env.original_task_id for env in envs_a], [env.original_task_id for env in envs_c])
