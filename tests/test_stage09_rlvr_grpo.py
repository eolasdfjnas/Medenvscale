from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medenvscale.rl.rlvr_grpo import (
    _build_grpo_config,
    _collect_rollouts,
    _dedupe_rollout_rows,
    _extract_final_code_from_completion,
    _prepare_trl_grpo_prompt_rows,
    attach_group_advantages,
    compute_rlvr_reward,
)
from medenvscale.utils import read_jsonl, write_jsonl
from medenvscale.pipeline_ops import _select_stage09_environments_from_stage05_5_split
from medenvscale.schemas import ExecutableEnvSpec


class Stage09RlvrGrpoTests(unittest.TestCase):
    def test_compute_reward_prefers_case_pass_and_submit(self) -> None:
        reward = compute_rlvr_reward(
            eval_row={
                "compile_passed": True,
                "execution_passed": False,
                "passed_cases": 1,
                "total_cases": 2,
                "failure_reasons": [],
            },
            run={"notes": []},
            reward_cfg={
                "sample_pass_weight": 1.0,
                "case_pass_rate_weight": 0.5,
                "valid_submit_weight": 0.1,
            },
        )

        self.assertFalse(reward["sample_pass"])
        self.assertEqual(reward["case_pass_rate"], 0.5)
        self.assertAlmostEqual(reward["reward"], 0.35)

    def test_attach_group_advantages_normalizes_within_env(self) -> None:
        rows = attach_group_advantages(
            [
                {"group_id": "env1", "reward": 0.0},
                {"group_id": "env1", "reward": 1.0},
                {"group_id": "env2", "reward": 0.5},
                {"group_id": "env2", "reward": 0.5},
            ]
        )

        self.assertLess(rows[0]["advantage"], 0)
        self.assertGreater(rows[1]["advantage"], 0)
        self.assertEqual(rows[2]["advantage"], 0.0)
        self.assertEqual(rows[3]["advantage"], 0.0)

    def test_compute_reward_can_include_rubric_score(self) -> None:
        reward = compute_rlvr_reward(
            eval_row={
                "compile_passed": True,
                "execution_passed": False,
                "passed_cases": 1,
                "total_cases": 2,
                "failure_reasons": [],
            },
            run={"notes": []},
            reward_cfg={
                "case_pass_rate_weight": 0.5,
                "valid_submit_weight": 0.1,
                "rubric_score_weight": 0.4,
            },
            rubric_score=0.75,
            use_rubric_reward=True,
        )

        self.assertAlmostEqual(reward["reward"], 0.65)
        self.assertEqual(reward["rubric_score"], 0.75)
        self.assertTrue(reward["use_rubric_reward"])

    def test_compute_reward_adds_bonus_for_success_within_budget(self) -> None:
        reward = compute_rlvr_reward(
            eval_row={
                "compile_passed": True,
                "execution_passed": True,
                "passed_cases": 2,
                "total_cases": 2,
                "failure_reasons": [],
            },
            run={"notes": []},
            trace={"tool_trace": []},
            reward_cfg={
                "sample_pass_weight": 1.0,
                "case_pass_rate_weight": 0.5,
                "valid_submit_weight": 0.1,
                "within_budget_success_bonus": 0.1,
            },
        )

        self.assertTrue(reward["sample_pass"])
        self.assertTrue(reward["budget_ok"])
        self.assertFalse(reward["tool_budget_violation"])
        self.assertAlmostEqual(reward["reward"], 1.7)

    def test_compute_reward_penalizes_budget_exhaustion_in_trace(self) -> None:
        reward = compute_rlvr_reward(
            eval_row={
                "compile_passed": True,
                "execution_passed": True,
                "passed_cases": 2,
                "total_cases": 2,
                "failure_reasons": [],
            },
            run={"notes": []},
            trace={
                "tool_trace": [
                    {"tool_name": "get_task_context", "result": {"ok": False, "error": "tool_budget_exhausted:get_task_context"}},
                    {"tool_name": "validate_candidate_code", "result": {"ok": False, "error": "tool_budget_exhausted:validate_candidate_code"}},
                ]
            },
            reward_cfg={
                "sample_pass_weight": 1.0,
                "case_pass_rate_weight": 0.5,
                "valid_submit_weight": 0.1,
                "within_budget_success_bonus": 0.1,
                "tool_budget_penalty": 0.1,
                "tool_budget_penalty_cap": 3,
            },
        )

        self.assertFalse(reward["budget_ok"])
        self.assertTrue(reward["tool_budget_violation"])
        self.assertEqual(reward["tool_budget_violation_count"], 2)
        self.assertAlmostEqual(reward["reward"], 1.4)

    def test_compute_reward_penalizes_soft_budget_violation_marker(self) -> None:
        reward = compute_rlvr_reward(
            eval_row={
                "compile_passed": True,
                "execution_passed": True,
                "passed_cases": 2,
                "total_cases": 2,
                "failure_reasons": [],
            },
            run={"notes": []},
            trace={
                "tool_trace": [
                    {
                        "tool_name": "get_task_context",
                        "budget_violation": True,
                        "budget_error": "tool_budget_exceeded:get_task_context",
                        "result": {"ok": True, "budget_violation": True},
                    }
                ]
            },
            reward_cfg={
                "sample_pass_weight": 1.0,
                "case_pass_rate_weight": 0.5,
                "valid_submit_weight": 0.1,
                "within_budget_success_bonus": 0.1,
                "tool_budget_penalty": 0.1,
                "tool_budget_penalty_cap": 3,
            },
        )

        self.assertFalse(reward["budget_ok"])
        self.assertEqual(reward["tool_budget_violation_count"], 1)
        self.assertAlmostEqual(reward["reward"], 1.5)

    def test_extract_final_code_from_json_completion(self) -> None:
        code = _extract_final_code_from_completion('{"final_code": "def solve():\\n    return 1", "notes": []}')

        self.assertEqual(code, "def solve():\n    return 1")

    def test_extract_final_code_from_markdown_fence(self) -> None:
        code = _extract_final_code_from_completion("```python\ndef solve():\n    return 2\n```")

        self.assertIn("return 2", code)

    def test_prepare_trl_grpo_prompt_rows_skips_zero_case_envs(self) -> None:
        env_with_case = ExecutableEnvSpec(
            env_id="env_with_case",
            original_task_id="task1",
            split="train",
            problem="Return one.",
            context="def solve():\n    pass",
            solution_form="function",
            primary_domain="scientific_software_engineering",
            primary_task_type="code_generation",
            gold_solution="def solve():\n    return 1",
            seed_execution_case={"case_id": "seed", "call_code": "result = solve()", "expected_output_signature": {}},
        )
        env_without_case = ExecutableEnvSpec(
            env_id="env_without_case",
            original_task_id="task2",
            split="train",
            problem="No case.",
            context="",
            solution_form="function",
            primary_domain="scientific_software_engineering",
            primary_task_type="code_generation",
            gold_solution="",
        )

        rows = _prepare_trl_grpo_prompt_rows([env_with_case, env_without_case], self._tmp_path("prepared_grpo_train.jsonl"))

        self.assertEqual([row["env_id"] for row in rows], ["env_with_case"])
        self.assertIn("final_code", "\n".join(message.get("content", "") for message in rows[0]["prompt"]))

    def test_stage09_split_filter_uses_stage05_5_labels(self) -> None:
        env_train = self._env("env_a_M1", "task_a", split="train")
        env_dev = self._env("env_b_M1", "task_b", split="dev")

        selected, metadata = _select_stage09_environments_from_stage05_5_split(
            environments=[env_train, env_dev],
            split="dev",
        )

        self.assertEqual([env.env_id for env in selected], ["env_b_M1"])
        self.assertEqual(metadata["split_source"], "stage05_5_assigned_split")

    def test_build_grpo_config_adds_step_eval_when_eval_split_is_set(self) -> None:
        class DummyConfig:
            def __init__(
                self,
                output_dir: str,
                eval_strategy: str | None = None,
                eval_steps: int | None = None,
                do_eval: bool | None = None,
            ) -> None:
                self.output_dir = output_dir
                self.eval_strategy = eval_strategy
                self.eval_steps = eval_steps
                self.do_eval = do_eval

        cfg = _build_grpo_config(
            DummyConfig,
            stage_cfg={"eval_split": "dev", "eval_steps": 25},
            output_dir=self._tmp_path("grpo"),
        )

        self.assertEqual(cfg.eval_strategy, "steps")
        self.assertEqual(cfg.eval_steps, 25)
        self.assertTrue(cfg.do_eval)

    def test_dedupe_rollout_rows_keeps_latest_for_id(self) -> None:
        rows = _dedupe_rollout_rows(
            [
                {"rollout_id": "env::r0", "value": 1},
                {"rollout_id": "env::r1", "value": 2},
                {"rollout_id": "env::r0", "value": 3},
            ]
        )

        self.assertEqual([row["rollout_id"] for row in rows], ["env::r0", "env::r1"])
        self.assertEqual(rows[0]["value"], 3)

    def test_collect_rollouts_resume_skips_completed_ids(self) -> None:
        class FakeConfig:
            values = {"stage06": {"tool_agent": {}}}

            def dataset_config_path(self, name: str):
                return name

            def dataset_config_path_with_fallback(self, primary: str, fallback: str):
                return primary

        env = self._env("env_a", "task_a")
        existing = {
            "rollout_id": "env_a::r0",
            "group_id": "env_a",
            "env_id": "env_a",
            "rollout_index": 0,
            "level": "M1",
            "run": {"passed": True},
            "trace": {"tool_trace": []},
            "eval": {"compile_passed": True, "execution_passed": True, "passed_cases": 1, "total_cases": 1},
            "rubrics": [],
        }
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = self._tmp_path_in(temp_dir, "rl_rollouts.jsonl")
            write_jsonl(checkpoint_path, [existing])
            with (
                patch("medenvscale.rl.rlvr_grpo.load_yaml", return_value={}),
                patch("medenvscale.rl.rlvr_grpo.run_tool_agent_for_env") as run_agent,
            ):
                run_agent.return_value = {
                    "run": {"passed": True},
                    "trace": {"tool_trace": []},
                    "eval": {"compile_passed": True, "execution_passed": True, "passed_cases": 1, "total_cases": 1},
                }

                rows = _collect_rollouts(
                    cfg=FakeConfig(),
                    environments=[env],
                    llm_client=object(),
                    stage_cfg={},
                    rollouts_per_env=2,
                    user_feedback=False,
                    existing_rows=read_jsonl(checkpoint_path),
                    checkpoint_path=checkpoint_path,
                )
                checkpoint_rows = read_jsonl(checkpoint_path)

                self.assertEqual([row["rollout_id"] for row in rows], ["env_a::r0", "env_a::r1"])
                run_agent.assert_called_once()
                self.assertEqual([row["rollout_id"] for row in checkpoint_rows], ["env_a::r0", "env_a::r1"])

    def _env(self, env_id: str, original_task_id: str, split: str = "train") -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id=env_id,
            original_task_id=original_task_id,
            split=split,
            problem="Return one.",
            context="",
            solution_form="function",
            primary_domain="scientific_software_engineering",
            primary_task_type="code_generation",
            gold_solution="",
            metadata={"dataset_split": split, "split_stage": "05_5"},
        )

    def _tmp_path(self, name: str):
        from tempfile import TemporaryDirectory

        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return __import__("pathlib").Path(temp.name) / name

    def _tmp_path_in(self, dirname: str, name: str):
        return __import__("pathlib").Path(dirname) / name


if __name__ == "__main__":
    unittest.main()
