from __future__ import annotations

import unittest

from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.split_assignment import assign_dataset_splits, split_envs_by_assigned_split


class Stage055SplitTests(unittest.TestCase):
    def test_assign_dataset_splits_groups_by_original_task(self) -> None:
        envs = [
            self._env("env_a_M1", "task_a"),
            self._env("env_a_M2", "task_a"),
            self._env("env_b_M1", "task_b"),
            self._env("env_c_M1", "task_c"),
        ]

        assigned, manifest = assign_dataset_splits(
            envs,
            {"seed": 7, "split_ratios": {"train": 0.7, "test": 0.3, "dev": 0.1}},
        )
        by_split = split_envs_by_assigned_split(assigned)
        task_a_splits = {env.split for env in assigned if env.original_task_id == "task_a"}

        self.assertEqual(len(task_a_splits), 1)
        self.assertEqual(manifest["raw_ratios"], {"train": 0.7, "dev": 0.1, "test": 0.3})
        self.assertAlmostEqual(sum(manifest["normalized_ratios"].values()), 1.0)
        self.assertEqual(sum(len(items) for items in by_split.values()), 4)
        self.assertTrue(all((env.metadata or {}).get("split_stage") == "05_5" for env in assigned))

    def _env(self, env_id: str, original_task_id: str) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id=env_id,
            original_task_id=original_task_id,
            split="source_train",
            problem="Return one.",
            context="",
            solution_form="function",
            primary_domain="scientific_software_engineering",
            primary_task_type="code_generation",
            gold_solution="",
        )


if __name__ == "__main__":
    unittest.main()
