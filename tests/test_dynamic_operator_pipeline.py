from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.dynamic_verifiable_operator_planner import synthesize_dynamic_operator_instances
from medenvscale.scaling.generic_operator_validator import repair_missing_state_updates, validate_dynamic_operator_instances
from medenvscale.scaling.verifier_delta_validator import validate_verifier_delta
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.verifier.verifier_builder import build_verifier_spec


class DynamicOperatorPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.client = LLMClient(
            config={"api": {"model": "unused", "api_key_env": "UNUSED", "base_url": "https://example.com"}},
            mode="mock",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-operator-"),
        )
        self.prompt_runner = PromptRunner(root / "prompts")

    def test_operator_axis_must_be_selected_and_v_axis_adds_tests(self) -> None:
        scaling_plan = {
            "selected_axes": ["C", "V"],
            "axis_intensity": {"D": 0, "C": 2, "A": 0, "V": 1},
        }
        tool_config = {"allowed_tools": [{"tool_name": "run_custom_test"}]}
        operators = synthesize_dynamic_operator_instances(
            env_id="env_demo_M3",
            task_id="demo",
            task_type="validation_and_code_utility",
            domain="scientific_software_engineering",
            secondary_domains=[{"domain": "bioinformatics_sequence_structure", "relevance": 0.6}],
            solution_form="patch_or_bugfix",
            scaling_plan=scaling_plan,
            tool_config=tool_config,
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo", "problem": "Patch code and preserve hidden verification."},
            intensity_rubric={"axis_definitions": {"C": "Constraint complexity", "V": "Verifier complexity"}},
        )
        errors = validate_dynamic_operator_instances(operators, scaling_plan)
        self.assertEqual(errors, [])
        v_ops = [op for op in operators if op.axis == "V"]
        self.assertTrue(v_ops)
        self.assertEqual(validate_verifier_delta(v_ops[0]), [])
        self.assertIn("bioinformatics_sequence_structure", v_ops[0].transformation_goal)
        self.assertEqual(sum(op.operator_intensity for op in operators if op.axis == "C"), 2)

    def test_verifier_builder_sanitizes_bad_hidden_tests(self) -> None:
        scaling_plan = {
            "selected_axes": ["V"],
            "axis_intensity": {"D": 0, "C": 0, "A": 0, "V": 1},
        }
        operators = synthesize_dynamic_operator_instances(
            env_id="env_demo_bad_V",
            task_id="demo_bad",
            task_type="validation_and_code_utility",
            domain="scientific_software_engineering",
            secondary_domains=[],
            solution_form="function_body",
            scaling_plan=scaling_plan,
            tool_config={"allowed_tools": [{"tool_name": "run_custom_test"}]},
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo_bad", "problem": "Validate and return code."},
            intensity_rubric={"axis_definitions": {"V": "Verifier complexity"}},
        )
        operators[0].verifier_delta.new_hidden_tests = ["bad_hidden_test", {"name": "ok", "assertion_code": "assert True"}]
        self.assertEqual(validate_verifier_delta(operators[0]), [])
        env = ExecutableEnvSpec(
            env_id="env_demo_bad_V",
            original_task_id="demo_bad",
            split="train",
            problem="Validate and return code.",
            context="def solve():\n    <<insert solution here>>\n",
            signature="def solve():",
            solution_form="function_body",
            primary_domain="scientific_software_engineering",
            primary_task_type="validation_and_code_utility",
            gold_solution="return 1",
        )
        verifier = build_verifier_spec(env, operators)
        self.assertEqual(verifier.hidden_tests, [])

    def test_missing_state_updates_are_repaired(self) -> None:
        scaling_plan = {
            "selected_axes": ["A"],
            "axis_intensity": {"D": 0, "C": 0, "A": 1, "V": 0},
        }
        operators = synthesize_dynamic_operator_instances(
            env_id="env_demo_bad_A",
            task_id="demo_bad_a",
            task_type="validation_and_code_utility",
            domain="scientific_software_engineering",
            secondary_domains=[],
            solution_form="function_body",
            scaling_plan=scaling_plan,
            tool_config={"allowed_tools": [{"tool_name": "run_custom_test"}]},
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo_bad_a", "problem": "Validate and return code."},
            intensity_rubric={"axis_definitions": {"A": "Adversarial"}},
        )
        broken = operators[0].model_copy(
            update={
                "state_updates": {
                    "task_state_patch": {},
                    "data_state_patch": {},
                    "tool_state_patch": {},
                    "visible_state_patch": {},
                    "gold_state_patch": {},
                    "verifier_state_patch": {},
                    "test_state_patch": {},
                    "turn_state_patch": {},
                }
            }
        )
        repaired = repair_missing_state_updates(broken)
        self.assertTrue(repaired.state_updates.visible_state_patch.get("robustness_trap"))
        self.assertTrue(repaired.state_updates.visible_state_patch.get("must_not_assume"))

    def test_axis_specific_repairs_fill_semantic_patches(self) -> None:
        scaling_plan = {
            "selected_axes": ["D"],
            "axis_intensity": {"D": 1, "C": 0, "A": 0, "V": 0},
        }
        operators = synthesize_dynamic_operator_instances(
            env_id="env_demo_bad_D",
            task_id="demo_bad_d",
            task_type="validation_and_code_utility",
            domain="scientific_software_engineering",
            secondary_domains=[],
            solution_form="function_body",
            scaling_plan=scaling_plan,
            tool_config={"allowed_tools": [{"tool_name": "run_custom_test"}]},
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo_bad_d", "problem": "Validate and return code."},
            intensity_rubric={"axis_definitions": {"D": "Data complexity"}},
        )
        weakened = operators[0].model_copy(
            update={
                "state_updates": {
                    "task_state_patch": {"axis_constraints": ["D"]},
                    "data_state_patch": {},
                    "tool_state_patch": {},
                    "visible_state_patch": {},
                    "gold_state_patch": {},
                    "verifier_state_patch": {},
                    "test_state_patch": {},
                    "turn_state_patch": {},
                }
            }
        )
        repaired = repair_missing_state_updates(weakened)
        self.assertTrue(repaired.state_updates.data_state_patch.get("resource_variants"))
        self.assertTrue(repaired.state_updates.visible_state_patch.get("input_description"))
        self.assertTrue(repaired.state_updates.gold_state_patch.get("gold_changed"))

    def test_repair_missing_state_updates_tolerates_none_patch_values(self) -> None:
        scaling_plan = {
            "selected_axes": ["V"],
            "axis_intensity": {"D": 0, "C": 0, "A": 0, "V": 1},
        }
        operators = synthesize_dynamic_operator_instances(
            env_id="env_demo_bad_V_none",
            task_id="demo_bad_v_none",
            task_type="validation_and_code_utility",
            domain="scientific_software_engineering",
            secondary_domains=[],
            solution_form="function_body",
            scaling_plan=scaling_plan,
            tool_config={"allowed_tools": [{"tool_name": "run_custom_test"}]},
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo_bad_v_none", "problem": "Validate and return code."},
            intensity_rubric={"axis_definitions": {"V": "Verification complexity"}},
        )
        broken = operators[0].model_copy(
            update={
                "state_updates": {
                    "task_state_patch": {},
                    "data_state_patch": {},
                    "tool_state_patch": {},
                    "visible_state_patch": {},
                    "gold_state_patch": None,
                    "verifier_state_patch": None,
                    "test_state_patch": {},
                    "turn_state_patch": {},
                }
            }
        )
        repaired = repair_missing_state_updates(broken)
        self.assertTrue(repaired.state_updates.gold_state_patch.get("answer_invariant"))
        self.assertTrue(repaired.state_updates.verifier_state_patch.get("stronger_checks"))

    def test_operator_planner_prompt_requires_output_constraints(self) -> None:
        prompt = self.prompt_runner.render(
            "dynamic_verifiable_operator_planner.jinja",
            seed_task="{}",
            base_environment="{}",
            domain="scientific_software_engineering",
            task_type="validation_and_code_utility",
            solution_form="function_body",
            domain_concepts="[]",
            scaling_plan="{}",
            tool_config="{}",
            intensity_rubric="{}",
        )
        self.assertIn("output_requirements", prompt)
        self.assertIn("output_constraint_spec", prompt)
        self.assertIn("Do not invent brittle checks.", prompt)
