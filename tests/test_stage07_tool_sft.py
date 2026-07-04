from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medenvscale.config import AppConfig
from medenvscale.llm.client import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import DifficultyProfile, ExecutableEnvSpec
from medenvscale.sft.tool_sft import generate_tool_sft_data


class Stage07ToolSftTests(unittest.TestCase):
    def _env(self, env_id: str = "env_demo_M2", original_task_id: str = "demo") -> ExecutableEnvSpec:
        code = "def solve(x):\n    return x + 1\n"
        return ExecutableEnvSpec(
            env_id=env_id,
            original_task_id=original_task_id,
            split="train",
            problem="Implement solve(x) and return x + 1.",
            context="Use only the Python standard library.",
            signature="def solve(x):",
            solution_form="python_function",
            primary_domain="scientific_software_engineering",
            primary_task_type="code_validation_and_utility",
            gold_solution=code,
            scaled_gold_solution=code,
            scaled_executable_gold_code=code,
            validated_oracle_cases=[
                {
                    "case_id": "case_1",
                    "case_kind": "publicly_described_behavior",
                    "setup_code": "",
                    "call_code": "result = solve(1)",
                    "expected_output_signature": {"return_value": 2, "return_type": "int", "stdout": "", "file_artifacts": []},
                }
            ],
            difficulty=DifficultyProfile(global_level="M2", D=0, C=1, A=0, V=0, selected_axes=["C"], total_intensity=1),
        )

    def test_generate_tool_sft_data_uses_stage06_tool_protocol(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="stage07-tool-sft-"))
        cfg = AppConfig(
            root=root,
            values={
                "stage06": {"tool_agent": {}},
                "splits": {"train": 0.7, "dev": 0.1, "test": 0.2, "seed": 1337},
                "output": {
                    "raw_dir": str(temp_dir / "raw"),
                    "interim_dir": str(temp_dir / "interim"),
                    "processed_dir": str(temp_dir / "processed"),
                    "split_dir": str(temp_dir / "splits"),
                    "result_dir": str(temp_dir / "result"),
                    "experiment_dir": str(temp_dir / "experiments"),
                },
            },
            llm_values={},
            dataset_name="biocoder",
        )
        llm = LLMClient(config={"api": {"model": "mock-teacher"}}, mode="mock", cache_dir=str(temp_dir / "cache"))
        output_paths = {
            "trajectories": temp_dir / "tool_sft_trajectories.jsonl",
            "quality_report": temp_dir / "tool_sft_quality_report.jsonl",
            "manifest": temp_dir / "manifest.json",
            "split_train": temp_dir / "tool_sft_train.jsonl",
            "split_dev": temp_dir / "tool_sft_dev.jsonl",
            "split_test": temp_dir / "tool_sft_test.jsonl",
        }

        result = generate_tool_sft_data(
            cfg=cfg,
            environments=[self._env()],
            llm_client=llm,
            prompt_runner=PromptRunner(root / "prompts"),
            output_paths=output_paths,
        )

        rows = result["trajectories"]
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row["quality"]["hidden_oracle_passed"] for row in rows))
        self.assertTrue(any(row["trajectory_type"] == "oracle_gold_tool_trajectory" for row in rows))
        self.assertTrue(any(row["trajectory_type"] == "teacher_agent_trajectory" for row in rows))
        tool_names = [
            call["function"]["name"]
            for row in rows
            for message in row["messages"]
            for call in message.get("tool_calls", [])
        ]
        self.assertIn("get_task_context", tool_names)
        self.assertIn("validate_candidate_code", tool_names)
        self.assertIn("run_custom_test", tool_names)
        self.assertIn("submit_final_code", tool_names)
        oracle_rows = [row for row in rows if row["trajectory_type"] == "oracle_gold_tool_trajectory"]
        self.assertTrue(oracle_rows)
        for row in oracle_rows:
            oracle_tool_names = [
                call["function"]["name"]
                for message in row["messages"]
                for call in message.get("tool_calls", [])
            ]
            self.assertIn("run_custom_test", oracle_tool_names)
        for row in rows:
            for message in row["messages"]:
                if message.get("role") == "tool":
                    self.assertNotIn("case_reports", message.get("content", ""))
                    self.assertNotIn("validated_oracle_cases", message.get("content", ""))
        self.assertTrue(output_paths["trajectories"].exists())
        self.assertTrue(output_paths["manifest"].exists())
        self.assertEqual(result["manifest"]["split_policy"], "legacy_stage07_group_split")
        self.assertEqual(
            result["manifest"]["trajectory_recipe"],
            "oracle_gold_tool_trajectory_plus_autonomous_teacher_agent_trajectory",
        )

    def test_split_happens_before_generation_by_original_task(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="stage07-tool-sft-split-"))
        cfg = AppConfig(
            root=root,
            values={
                "stage06": {"tool_agent": {}},
                "splits": {"train": 0.34, "dev": 0.33, "test": 0.33, "seed": 7},
                "output": {
                    "raw_dir": str(temp_dir / "raw"),
                    "interim_dir": str(temp_dir / "interim"),
                    "processed_dir": str(temp_dir / "processed"),
                    "split_dir": str(temp_dir / "splits"),
                    "result_dir": str(temp_dir / "result"),
                    "experiment_dir": str(temp_dir / "experiments"),
                },
            },
            llm_values={},
            dataset_name="biocoder",
        )
        llm = LLMClient(config={"api": {"model": "mock-teacher"}}, mode="mock", cache_dir=str(temp_dir / "cache"))
        output_paths = {
            "trajectories": temp_dir / "tool_sft_trajectories.jsonl",
            "quality_report": temp_dir / "tool_sft_quality_report.jsonl",
            "manifest": temp_dir / "manifest.json",
            "split_train": temp_dir / "tool_sft_train.jsonl",
            "split_dev": temp_dir / "tool_sft_dev.jsonl",
            "split_test": temp_dir / "tool_sft_test.jsonl",
        }

        result = generate_tool_sft_data(
            cfg=cfg,
            environments=[
                self._env(env_id="env_same_M1", original_task_id="same_seed"),
                self._env(env_id="env_same_M2", original_task_id="same_seed"),
                self._env(env_id="env_other_M2", original_task_id="other_seed"),
            ],
            llm_client=llm,
            prompt_runner=PromptRunner(root / "prompts"),
            output_paths=output_paths,
        )

        split_by_seed: dict[str, set[str]] = {}
        for split_name, rows in result["splits"].items():
            for row in rows:
                split_by_seed.setdefault(row["original_task_id"], set()).add(split_name)
                self.assertEqual(row["split"], split_name)
                self.assertTrue(row["metadata"]["split_assigned_before_generation"])
        self.assertTrue(split_by_seed)
        self.assertTrue(all(len(split_names) == 1 for split_names in split_by_seed.values()))

    def test_resume_rebuilds_outputs_from_checkpoint(self) -> None:
        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="stage07-tool-sft-resume-"))
        cfg = AppConfig(
            root=root,
            values={
                "stage06": {"tool_agent": {}},
                "splits": {"train": 0.7, "dev": 0.1, "test": 0.2, "seed": 1337},
                "output": {
                    "raw_dir": str(temp_dir / "raw"),
                    "interim_dir": str(temp_dir / "interim"),
                    "processed_dir": str(temp_dir / "processed"),
                    "split_dir": str(temp_dir / "splits"),
                    "result_dir": str(temp_dir / "result"),
                    "experiment_dir": str(temp_dir / "experiments"),
                },
            },
            llm_values={},
            dataset_name="biocoder",
        )
        llm = LLMClient(config={"api": {"model": "mock-teacher"}}, mode="mock", cache_dir=str(temp_dir / "cache"))
        output_paths = {
            "trajectories": temp_dir / "tool_sft_trajectories.jsonl",
            "quality_report": temp_dir / "tool_sft_quality_report.jsonl",
            "manifest": temp_dir / "manifest.json",
            "split_train": temp_dir / "tool_sft_train.jsonl",
            "split_dev": temp_dir / "tool_sft_dev.jsonl",
            "split_test": temp_dir / "tool_sft_test.jsonl",
        }
        checkpoint = temp_dir / "checkpoint.jsonl"
        first = generate_tool_sft_data(
            cfg=cfg,
            environments=[self._env()],
            llm_client=llm,
            prompt_runner=PromptRunner(root / "prompts"),
            output_paths=output_paths,
            resume=True,
            checkpoint_path=checkpoint,
        )
        for path in output_paths.values():
            path.unlink()

        resumed = generate_tool_sft_data(
            cfg=cfg,
            environments=[self._env()],
            llm_client=llm,
            prompt_runner=PromptRunner(root / "prompts"),
            output_paths=output_paths,
            resume=True,
            checkpoint_path=checkpoint,
        )

        self.assertEqual(len(resumed["trajectories"]), len(first["trajectories"]))
        self.assertTrue(output_paths["trajectories"].exists())
        self.assertTrue(output_paths["manifest"].exists())


if __name__ == "__main__":
    unittest.main()
