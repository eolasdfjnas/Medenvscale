from __future__ import annotations

import unittest

from medenvscale.schemas import DifficultyProfile, ExecutableEnvSpec
from medenvscale.validation.operator_realization_checker import check_operator_realization


class OperatorRealizationCheckerTests(unittest.TestCase):
    def _seed_env(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="seed_demo",
            original_task_id="demo",
            split="train",
            problem="Return x + 1.",
            context="def solve(x):\n    <<insert solution here>>\n",
            signature="def solve(x):",
            solution_form="function_body",
            primary_domain="scientific_software_engineering",
            primary_task_type="validation_and_code_utility",
            gold_solution="return x + 1",
            visible_state={"placeholder_token": "<<insert solution here>>"},
            difficulty=DifficultyProfile(global_level="M1", H=0, D=0, R=0, I=0, E=0, C=0, A=0, V=0, selected_axes=[], total_intensity=0),
        )

    def test_semantic_operator_reusing_seed_gold_hard_fails(self) -> None:
        seed = self._seed_env()
        scaled = seed.model_copy(
            update={
                "env_id": "env_demo_M2",
                "user_prompt": (
                    "Additional requirements:\n"
                    "- handle edge cases\n"
                    "- preserve output contract\n"
                ),
                "gold_state": {
                    "gold_changed": True,
                    "answer_invariant": False,
                    "gold_change_reason": "C-axis operator adds explicit constraints.",
                    "seed_gold_compatible_with_scaled_task": False,
                },
            }
        )
        operator = {
            "operator_id": "env_demo_M2_c_01",
            "axis": "C",
            "operator_intensity": 1,
            "state_updates": {
                "task_state_patch": {"extra_constraints": ["handle edge cases", "preserve output contract"]},
                "visible_state_patch": {"constraint_hints": ["handle edge cases", "preserve output contract"]},
                "verifier_state_patch": {"constraint_checks": ["handle edge cases"]},
                "test_state_patch": {"constraint_hidden_tests": ["verify added constraint"]},
            },
            "verifier_delta": {
                "new_hidden_tests": [
                    {
                        "test_id": "test_extra_constraint_format",
                        "targets_operator_id": "env_demo_M2_c_01",
                        "axis": "C",
                        "code": "assert solve(1) == 2",
                        "test_tier": "semantic",
                        "counts_as_hidden_test": True,
                        "eligible_for_clean_export": True,
                        "expected_failure_mode": "original_solution_ignores_extra_constraint",
                    }
                ],
                "new_checks": [],
                "static_checks": [],
                "expected_failure_modes": ["original_solution_ignores_extra_constraint"],
            },
        }
        report = check_operator_realization(
            seed_task=seed,
            scaled_task=scaled.model_copy(update={"operator_instances": [operator], "hidden_tests": operator["verifier_delta"]["new_hidden_tests"]}),
            operator_instance=operator,
            hidden_tests=operator["verifier_delta"]["new_hidden_tests"],
            verifier_specs={"hidden_tests": operator["verifier_delta"]["new_hidden_tests"]},
        )
        self.assertEqual(report["severity"], "hard_fail")
        self.assertIn("SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION", report["failure_reasons"])

    def test_v_axis_with_answer_invariant_and_linked_oracle_case_passes(self) -> None:
        seed = self._seed_env()
        v_case = {
            "case_id": "case_v_semantic",
            "targets_operator_id": "env_demo_M2_v_01",
            "axis": "V",
            "description": "Check stronger oracle coverage for x=1.",
            "semantic_intent": "Verify stronger oracle coverage while preserving answer invariant behavior.",
            "target_constraint": "Solutions must remain correct under stronger oracle validation.",
            "expected_failure_mode": "weak verifier misses this executable oracle case",
            "setup_code": "x = 1",
            "call_code": "result = solve(x)",
            "expected_output_signature": {"return_type": "int", "return_value": 2},
            "covered_requirements": ["Solutions must remain correct under stronger oracle validation."],
        }
        scaled = seed.model_copy(
            update={
                "env_id": "env_demo_M2_v",
                "user_prompt": "Task prompt with oracle validation.\nSolutions must remain correct under stronger oracle validation.",
                "gold_state": {
                    "gold_changed": False,
                    "answer_invariant": True,
                    "gold_change_reason": "V-axis only; task semantics unchanged.",
                    "seed_gold_compatible_with_scaled_task": True,
                },
                "validated_oracle_cases": [v_case],
                "scaled_oracle_cases": [v_case],
                "scaled_gold_case_execution_report": [
                    {
                        "case_id": "case_v_semantic",
                        "passed": True,
                        "failure_reasons": [],
                        "observed_output_signature": {"return_type": "int", "return_value": 2, "stdout": "", "file_artifacts": []},
                        "expected_output_signature": v_case["expected_output_signature"],
                    }
                ],
            }
        )
        operator = {
            "operator_id": "env_demo_M2_v_01",
            "axis": "V",
            "operator_intensity": 1,
            "state_updates": {
                "visible_state_patch": {"execution_requirements": ["Solutions must remain correct under stronger oracle validation."]},
                "verifier_state_patch": {"verifier_hardening": {"hidden_test_layers": 1}, "stronger_checks": ["semantic verifier coverage level 1"]},
                "test_state_patch": {"oracle_cases": ["case_v_semantic"]},
                "gold_state_patch": {
                    "gold_changed": False,
                    "answer_invariant": True,
                    "gold_change_reason": "V-axis only; task semantics unchanged.",
                    "seed_gold_compatible_with_scaled_task": True,
                },
            },
            "verifier_delta": {
                "new_hidden_tests": [],
                "new_checks": [],
                "static_checks": [],
                "expected_failure_modes": ["semantic_verifier_finds_uncovered_edge_case"],
            },
        }
        report = check_operator_realization(
            seed_task=seed,
            scaled_task=scaled.model_copy(update={"operator_instances": [operator]}),
            operator_instance=operator,
        )
        self.assertEqual(report["severity"], "pass")
        self.assertTrue(report["passed"])

    def test_v_axis_without_linked_oracle_case_hard_fails_even_with_verifier_patch(self) -> None:
        seed = self._seed_env()
        scaled = seed.model_copy(
            update={
                "env_id": "env_demo_M2_v",
                "user_prompt": "Solutions must remain correct under stronger oracle validation.",
                "gold_state": {
                    "gold_changed": False,
                    "answer_invariant": True,
                    "gold_change_reason": "V-axis only; task semantics unchanged.",
                    "seed_gold_compatible_with_scaled_task": True,
                },
            }
        )
        operator = {
            "operator_id": "env_demo_M2_v_01",
            "axis": "V",
            "operator_intensity": 1,
            "state_updates": {
                "visible_state_patch": {"execution_requirements": ["Solutions must remain correct under stronger oracle validation."]},
                "verifier_state_patch": {"stronger_checks": ["semantic verifier coverage level 1"]},
                "test_state_patch": {"oracle_cases": ["case_v_semantic"]},
                "gold_state_patch": {
                    "gold_changed": False,
                    "answer_invariant": True,
                    "gold_change_reason": "V-axis only; task semantics unchanged.",
                    "seed_gold_compatible_with_scaled_task": True,
                },
            },
            "verifier_delta": {"new_hidden_tests": [], "new_checks": [], "static_checks": []},
        }
        report = check_operator_realization(seed_task=seed, scaled_task=scaled, operator_instance=operator)
        self.assertEqual(report["severity"], "hard_fail")
        self.assertIn("V_NOT_ENOUGH_CASES", report["failure_reasons"])
        self.assertIn("V_NO_GOLD_PASS_SIGNAL", report["failure_reasons"])
