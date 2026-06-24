from __future__ import annotations

import tempfile
import unittest

from medenvscale.config import resolve_dataset_config_path, resolve_dataset_config_path_with_fallback
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.axis_weight_planner import fallback_rank_weights, plan_axis_weights
from medenvscale.scaling.scaling_plan import build_scaling_plan
from medenvscale.schemas import AxisWeightPlannerResult
from medenvscale.utils import load_yaml


class ScalingPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = __import__("pathlib").Path(__file__).resolve().parent.parent
        self.axis_cfg = load_yaml(resolve_dataset_config_path(self.root, "task_axis_priority.yaml", dataset="biocoder"))
        self.budgets_cfg = load_yaml(
            resolve_dataset_config_path_with_fallback(
                self.root,
                "m_level_budgets_4axis.yaml",
                "m_level_budgets_7axis.yaml",
                dataset="biocoder",
            )
        )
        self.fusion_cfg = load_yaml(resolve_dataset_config_path_with_fallback(self.root, "axis_weight_fusion.yaml", "axis_weight_fusion.yaml", dataset="biocoder"))
        self.client = LLMClient(
            config={"api": {"model": "unused", "api_key_env": "UNUSED", "base_url": "https://example.com"}},
            mode="mock",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-axis-"),
        )
        self.prompt_runner = PromptRunner(self.root / "prompts")

    def _weights(self, task_type: str) -> AxisWeightPlannerResult:
        axis_priority = self.axis_cfg["task_axis_priority"][task_type]["axis_priority"]
        return AxisWeightPlannerResult(primary_axis_weight_hint=fallback_rank_weights(axis_priority))

    def test_axis_weight_planner_mock_uses_llm_main_path(self) -> None:
        weights, source, trace = plan_axis_weights(
            task_type="numerical_computation",
            secondary_task_types=["validation_and_code_utility"],
            task_axis_priority_cfg=self.axis_cfg,
            problem="Write code that computes a tolerance-aware free energy estimate and raises on invalid input.",
            context_summary="Use numpy arrays and validate numeric outputs.",
            signature="def solve(values):",
            verifier_type_hint="numeric_tolerance",
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            domain="systems_molecular_modeling",
            solution_form="function_body",
        )
        self.assertEqual(source, "llm")
        self.assertGreaterEqual(weights.primary_axis_weight_hint["C"], 5)
        self.assertGreaterEqual(weights.primary_axis_weight_hint["V"], 4)
        self.assertEqual(trace, [])

    def test_m1_has_no_axes(self) -> None:
        plan = build_scaling_plan(
            env_id="env_demo_M1",
            global_level="M1",
            task_type="structured_data_processing",
            secondary_task_types=[],
            domain="biomedical_data_analysis",
            solution_form="function_body",
            axis_priority_cfg=self.axis_cfg,
            budgets_cfg=self.budgets_cfg,
            fusion_cfg=self.fusion_cfg,
            axis_weights=self._weights("structured_data_processing"),
            axis_weight_source="fallback",
        )
        self.assertEqual(plan.selected_axes, [])
        self.assertEqual(plan.total_intensity, 0)

    def test_m4_selects_three_to_four_4axis_axes(self) -> None:
        plan = build_scaling_plan(
            env_id="env_demo_M4",
            global_level="M4",
            task_type="code_validation_and_utility",
            secondary_task_types=[],
            domain="scientific_software_engineering",
            solution_form="patch_or_bugfix",
            axis_priority_cfg=self.axis_cfg,
            budgets_cfg=self.budgets_cfg,
            fusion_cfg=self.fusion_cfg,
            axis_weights=self._weights("code_validation_and_utility"),
            axis_weight_source="fallback",
        )
        self.assertGreaterEqual(len(plan.selected_axes), 3)
        self.assertLessEqual(len(plan.selected_axes), 4)
        self.assertTrue(set(plan.selected_axes).issubset({"D", "C", "A", "V"}))
        self.assertTrue({"C", "A"}.issubset(set(plan.selected_axes)))
        self.assertEqual(sum(plan.axis_intensity.values()), plan.total_intensity)
