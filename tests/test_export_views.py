from __future__ import annotations

import unittest

from medenvscale.export.export_prm import export_prm_samples
from medenvscale.export.export_rlvr import export_rlvr_stub
from medenvscale.export.export_sft import export_sft_sample
from medenvscale.schemas import DifficultyProfile, DomainHint, ExecutableEnvSpec, QuestionPoint, RubricCriterion


class ExportViewTests(unittest.TestCase):
    def _env(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="env_demo_M2",
            original_task_id="demo",
            split="train",
            problem="Filter rows",
            context="def run(path):\n    <<insert solution here>>\n",
            signature="def run(path):",
            solution_form="function_body",
            primary_domain="biomedical_data_analysis",
            secondary_domains=[DomainHint(domain="omics_measurement_analysis", relevance=0.8)],
            primary_task_type="tabular_data_transformation",
            verifier_type_hint="dataframe_equal",
            gold_solution="return []",
            visible_state={"placeholder_token": "<<insert solution here>>"},
            gold_state={"operator_mode": "gold_compatible"},
            difficulty=DifficultyProfile(
                global_level="M2",
                H=0,
                R=0,
                I=1,
                E=1,
                C=1,
                A=0,
                V=1,
                selected_axes=["E", "V"],
                total_intensity=2,
            ),
            tool_config={"allowed_tools": [{"tool_name": "get_context"}], "tool_budget": {"max_total_tool_calls": 3}},
            verifier_spec={"verifier_id": "verifier_env_demo_M2"},
            operator_instances=[{"operator_id": "env_demo_M2_v_01"}],
            user_prompt="Task prompt",
        )

    def test_sft_and_rlvr_schema(self) -> None:
        env = self._env()
        rubrics = [RubricCriterion(env_id=env.env_id, rubric_id="r1", source_point_id="q1", criterion="preserve contract", score_type="binary", weight=1, category="C")]
        sft = export_sft_sample(env, rubrics, system_prompt="sys")
        rlvr = export_rlvr_stub(env)
        self.assertEqual(sft.verifier_id, "verifier_env_demo_M2")
        self.assertEqual(sft.secondary_domains[0]["domain"], "omics_measurement_analysis")
        self.assertIn("submit_answer", rlvr.action_space)
        self.assertEqual(rlvr.tool_config["tool_budget"]["max_total_tool_calls"], 3)
        self.assertEqual(rlvr.secondary_task_types, env.secondary_task_types)

    def test_prm_export_schema(self) -> None:
        env = self._env()
        points = [QuestionPoint(env_id=env.env_id, point_id="q1", title="Contract", description="Preserve contract", related_axes=["C"])]
        prm = export_prm_samples(env, points)
        self.assertEqual(prm[0].env_id, env.env_id)
        self.assertEqual(prm[0].related_axes, ["C"])
