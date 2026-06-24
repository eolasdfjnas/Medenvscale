from __future__ import annotations

import copy
import io
from contextlib import redirect_stderr, redirect_stdout
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from medenvscale.config import AppConfig
from medenvscale.pipeline_ops import normalize_output_path, routing_output_path
from medenvscale.scaling.hidden_test_runner import run_hidden_test_execution_check
from medenvscale.scaling.quality_filter import split_clean_and_rejected
from medenvscale.schemas import DifficultyProfile, ExecutableEnvSpec
from medenvscale.utils import ensure_dir, load_yaml, write_jsonl


class Stage05QualityGuardTests(unittest.TestCase):
    def _build_temp_config(self, root: Path) -> AppConfig:
        values = copy.deepcopy(load_yaml(root / "configs" / "biocoder" / "medagentgym_pilot.yaml"))
        temp_root = Path(tempfile.mkdtemp(prefix="medenvscale-stage05-"))
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

    def _base_env(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="env_demo_M1",
            original_task_id="demo",
            split="train",
            problem="Write a function solve(x) that returns x + 1.",
            context="def solve(x):\n    <<insert solution here>>\n",
            signature="def solve(x):",
            solution_form="function_body",
            primary_domain="scientific_software_engineering",
            primary_task_type="validation_and_code_utility",
            gold_solution="return x + 1",
            visible_state={"placeholder_token": "<<insert solution here>>"},
            difficulty=DifficultyProfile(global_level="M1", H=0, D=0, R=0, I=0, E=0, C=0, A=0, V=0, selected_axes=[], total_intensity=0),
            hidden_tests=[],
            quality_flags=[],
        )

    def test_hidden_test_execution_guard_catches_duplicates_and_failures(self) -> None:
        env = self._base_env().model_copy(
            update={
                "hidden_tests": [
                    {"test_id": "t1", "name": "t1", "code": "assert solve(1) == 2", "source": "llm", "test_tier": "semantic", "counts_as_hidden_test": True, "eligible_for_clean_export": True},
                    {"test_id": "t1", "name": "t1_dup", "code": "assert solve(2) == 3", "source": "llm", "test_tier": "semantic", "counts_as_hidden_test": True, "eligible_for_clean_export": True},
                    {"test_id": "t2", "name": "t2", "code": "assert solve(3) == 99", "source": "llm", "test_tier": "semantic", "counts_as_hidden_test": True, "eligible_for_clean_export": True},
                ]
            }
        )
        result = run_hidden_test_execution_check(env)
        self.assertEqual(result.status, "fail")
        self.assertTrue(any("duplicate_hidden_test_id" in error for error in result.errors))
        self.assertTrue(any("hidden_test_execution_error" in error for error in result.errors))

    def test_hidden_test_execution_guard_suppresses_subprocess_output(self) -> None:
        env = self._base_env().model_copy(
            update={
                "context": (
                    "import subprocess\nimport sys\n"
                    "def solve(x):\n    <<insert solution here>>\n"
                    "if __name__ == '__main__':\n"
                    "    subprocess.run([sys.executable, '-c', \"import sys; print('loud stdout'); print('loud stderr', file=sys.stderr)\"])\n"
                ),
            }
        )
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            result = run_hidden_test_execution_check(env)
        self.assertEqual(result.status, "pass")
        self.assertEqual(captured_stdout.getvalue(), "")
        self.assertEqual(captured_stderr.getvalue(), "")

    def test_split_clean_and_rejected_rejects_semantic_equivalent_and_fallback_hidden_tests(self) -> None:
        base_env = self._base_env()
        m2_env = base_env.model_copy(
            update={
                "env_id": "env_demo_M2",
                "difficulty": DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=1, A=0, V=0, selected_axes=["C"], total_intensity=1),
                "hidden_tests": [
                    {
                        "test_id": "fallback_test",
                        "name": "fallback_test",
                        "code": "assert solve(1) == 2",
                        "source": "fallback",
                        "test_tier": "smoke",
                        "counts_as_hidden_test": False,
                        "eligible_for_clean_export": False,
                    }
                ],
                "operator_instances": [
                    {
                        "operator_id": "op1",
                        "state_updates": {
                            "task_state_patch": {"extra_constraints": ["handle edge cases"]},
                            "data_state_patch": {},
                            "tool_state_patch": {},
                            "visible_state_patch": {},
                            "gold_state_patch": {},
                            "verifier_state_patch": {},
                            "test_state_patch": {},
                            "turn_state_patch": {},
                        }
                    }
                ],
            }
        )
        clean, rejected = split_clean_and_rejected([base_env, m2_env])
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0].env_id, "env_demo_M1")
        self.assertEqual(len(rejected), 1)
        blocking = rejected[0].blocking_quality_flags
        self.assertIn("semantic_equivalent_to_base_m1", blocking)
        self.assertTrue(any("fallback_hidden_test_present" in item for item in blocking))
        self.assertTrue(any("operator_missing_semantic_delta" in item for item in blocking))
