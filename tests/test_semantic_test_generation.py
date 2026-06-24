from __future__ import annotations

import unittest
from pathlib import Path

from medenvscale.scaling.hidden_test_runner import run_hidden_test_execution_check
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.scaled_gold_solver_generator import (
    _build_hidden_tests_from_scaled_oracle_cases,
    _build_scaled_gold_prompt,
    detect_semantic_change,
    generate_scaled_gold_solution_if_needed,
)
from medenvscale.schemas import DifficultyProfile, ExecutableEnvSpec
from medenvscale.verifier.semantic_test_materializer import materialize_semantic_test_spec


class SemanticTestGenerationTests(unittest.TestCase):
    def _env(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="env_semantic_demo",
            original_task_id="semantic_demo",
            split="train",
            problem="Write solve(x) that returns x + 1.",
            context="def solve(x):\n    <<insert solution here>>\n",
            signature="def solve(x):",
            solution_form="function_body",
            primary_domain="scientific_software_engineering",
            primary_task_type="validation_and_code_utility",
            code="def solve(x):\n    return x + 1\n",
            gold_solution="return x + 1",
            seed_gold_solution="return x + 1",
            scaled_gold_solution="return x + 1",
            visible_state={"placeholder_token": "<<insert solution here>>"},
            difficulty=DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=1, A=0, V=1, selected_axes=["C", "V"], total_intensity=2),
        )

    def test_detect_semantic_change_distinguishes_v_only(self) -> None:
        v_only = detect_semantic_change(
            [
                {
                    "operator_id": "op_v_1",
                    "axis": "V",
                    "semantic_change": False,
                    "state_updates": {
                        "visible_state_patch": {},
                        "task_state_patch": {},
                        "gold_state_patch": {},
                        "data_state_patch": {},
                        "test_state_patch": {},
                        "verifier_state_patch": {"stronger_checks": ["x"]},
                    },
                }
            ]
        )
        semantic = detect_semantic_change(
            [
                {
                    "operator_id": "op_c_1",
                    "axis": "C",
                    "semantic_change": True,
                    "state_updates": {
                        "visible_state_patch": {"output_constraints": ["empty input returns None"]},
                        "task_state_patch": {},
                        "gold_state_patch": {"gold_changed": True},
                        "data_state_patch": {},
                        "test_state_patch": {},
                        "verifier_state_patch": {},
                    },
                }
            ]
        )
        self.assertFalse(v_only["semantic_change"])
        self.assertTrue(semantic["semantic_change"])

    def test_materialize_semantic_test_spec_builds_executable_test(self) -> None:
        spec = {
            "spec_id": "spec_v_001",
            "targets_operator_id": "op_V_001",
            "axis": "V",
            "semantic_intent": "Check lower-bound behavior.",
            "target_constraint": "The solution should remain correct on a non-happy-path integer input.",
            "expected_failure_mode": "weak_solution_only_handles_visible_happy_path",
            "test_template_type": "semantic_coverage_case",
            "input_variant": {"kind": "coverage_variant", "value": "boundary_case"},
            "expected_behavior": {"kind": "oracle_output_match"},
        }
        env = self._env()
        result = materialize_semantic_test_spec(
            spec,
            {
                "problem": env.problem,
                "context": env.context,
                "signature": env.signature,
                "solution_form": env.solution_form,
                "scaled_gold_solution": env.scaled_gold_solution,
                "gold_solution": env.gold_solution,
                "placeholder_token": env.visible_state["placeholder_token"],
            },
        )
        self.assertEqual(result["materialization_status"], "success")
        self.assertIn("assert", result["code"])
        checked_env = env.model_copy(update={"hidden_tests": [result], "gold_solution": env.scaled_gold_solution})
        run_result = run_hidden_test_execution_check(checked_env)
        self.assertEqual(run_result.status, "pass")

    def test_generate_scaled_gold_solution_reuses_seed_for_v_only(self) -> None:
        env = self._env().model_copy(
            update={"difficulty": DifficultyProfile(global_level="M1", H=0, D=0, R=0, I=0, E=0, C=0, A=0, V=1, selected_axes=["V"], total_intensity=1)}
        )
        result = generate_scaled_gold_solution_if_needed(
            env=env,
            operator_instances=[
                {
                    "operator_id": "op_v_1",
                    "axis": "V",
                    "semantic_change": False,
                    "state_updates": {
                        "visible_state_patch": {},
                        "task_state_patch": {},
                        "gold_state_patch": {},
                        "data_state_patch": {},
                        "test_state_patch": {},
                        "verifier_state_patch": {"stronger_checks": ["x"]},
                    },
                    "gold_update_policy": {"answer_invariant": True},
                }
            ],
            semantic_test_specs=[],
            output_constraint_spec={},
            llm_client=None,
            prompt_runner=None,
        )
        self.assertTrue(result["answer_invariant"])
        self.assertEqual(result["scaled_executable_gold_code"].strip(), env.code.strip())

    def test_generate_scaled_gold_solution_requires_oracle_case_for_non_m1_v_only(self) -> None:
        env = self._env().model_copy(
            update={
                "difficulty": DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=0, A=0, V=1, selected_axes=["V"], total_intensity=1),
                "user_prompt": (
                    "Return x + 1.\n\n"
                    "Additional requirements:\n"
                    "- The oracle case must call solve with x and return int 2.\n"
                ),
                "output_requirements": ["The oracle case must call solve with x and return int 2."],
                "seed_case_audit": {"status": "pass"},
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "description": "seed",
                    "setup_code": "x = 1",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 2},
                },
                "seed_ground_truth_output_signature": {"return_type": "int", "return_value": 2},
            }
        )
        result = generate_scaled_gold_solution_if_needed(
            env=env,
            operator_instances=[
                {
                    "operator_id": "op_v_1",
                    "axis": "V",
                    "semantic_change": False,
                    "output_requirements": ["The oracle case must call solve with x and return int 2."],
                    "state_updates": {
                        "visible_state_patch": {"execution_requirements": ["The oracle case must call solve with x and return int 2."]},
                        "task_state_patch": {},
                        "gold_state_patch": {
                            "gold_changed": False,
                            "answer_invariant": True,
                            "seed_gold_compatible_with_scaled_task": True,
                        },
                        "data_state_patch": {},
                        "test_state_patch": {"oracle_cases": ["seed_case_main"]},
                        "verifier_state_patch": {},
                    },
                    "gold_update_policy": {"answer_invariant": True},
                }
            ],
            semantic_test_specs=[],
            output_constraint_spec={},
            llm_client=None,
            prompt_runner=None,
        )
        self.assertTrue(result["answer_invariant"])
        self.assertEqual(result["scaled_executable_gold_code"].strip(), env.code.strip())
        self.assertGreaterEqual(len(result["validated_oracle_cases"]), 1)
        self.assertTrue(result["scaled_gold_case_execution_report"][0]["passed"])
        self.assertNotIn("NO_VALIDATED_ORACLE_CASES", result["failure_reasons"])

    def test_scaled_oracle_case_compiles_into_closed_loop_hidden_test(self) -> None:
        hidden_tests, failures = _build_hidden_tests_from_scaled_oracle_cases(
            [
                {
                    "case_id": "case_return_and_stdout",
                    "description": "Return value and stdout are both checked.",
                    "targets_operator_id": "op_c_1",
                    "axis": "C",
                    "semantic_intent": "Verify empty-safe path.",
                    "target_constraint": "Return 2 and print done.",
                    "expected_failure_mode": "returns wrong value",
                    "setup_code": "x = 1",
                    "call_code": "result = solve(x)\nprint('done')",
                    "expected_output_signature": {
                        "return_type": "int",
                        "return_value": 2,
                        "stdout_contains": ["done"],
                    },
                }
            ]
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(hidden_tests), 1)
        code = hidden_tests[0]["code"]
        self.assertIn("result = solve(x)", code)
        self.assertIn("stdout = _scaled_oracle_stdout.getvalue()", code)
        self.assertIn("assert result == 2", code)

    def test_generate_scaled_gold_solution_flags_unjustified_seed_reuse(self) -> None:
        env = self._env().model_copy(
            update={
                "difficulty": DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=1, A=0, V=0, selected_axes=["C"], total_intensity=1),
                "user_prompt": "Return None on empty input; otherwise return x + 1.",
            }
        )
        result = generate_scaled_gold_solution_if_needed(
            env=env,
            operator_instances=[
                {
                    "operator_id": "op_c_1",
                    "axis": "C",
                    "semantic_change": True,
                    "state_updates": {
                        "visible_state_patch": {"output_constraints": ["Return None on empty input."]},
                        "task_state_patch": {"extra_constraints": ["Return None on empty input."]},
                        "gold_state_patch": {"gold_changed": True},
                        "data_state_patch": {},
                        "test_state_patch": {},
                        "verifier_state_patch": {},
                    },
                    "gold_update_policy": {"requires_scaled_gold_solution_generation": True, "answer_invariant": False},
                }
            ],
            semantic_test_specs=[
                {
                    "spec_id": "spec_c_empty_input",
                    "targets_operator_id": "op_c_1",
                    "axis": "C",
                    "semantic_intent": "Check the empty-input constraint.",
                    "target_constraint": "Return None on empty input.",
                    "expected_failure_mode": "original_solution_raises_error_on_empty_input",
                    "test_template_type": "constraint_boundary_case",
                    "input_variant": {"kind": "empty_input"},
                    "expected_behavior": {"kind": "oracle_output_match"},
                }
            ],
            output_constraint_spec={
                "checks": [
                    {
                        "check_id": "stdout_non_empty",
                        "kind": "stdout",
                        "rule": "non_empty",
                        "params": {},
                        "severity": "soft",
                    }
                ]
            },
            llm_client=None,
            prompt_runner=None,
        )
        self.assertIn("SEED_GOLD_REUSED_WITHOUT_JUSTIFICATION", result["failure_reasons"])
        self.assertEqual(result["scaled_gold_solution"], result["scaled_executable_gold_code"])

    def test_scaled_gold_generate_prompt_uses_jinja_template(self) -> None:
        env = self._env().model_copy(update={"user_prompt": "Return x + 1 while preserving semantics."})
        prompt_runner = PromptRunner(Path(__file__).resolve().parent.parent / "prompts")
        prompt = _build_scaled_gold_prompt(
            env=env,
            operator_instances=[],
            semantic_test_specs=[],
            scaled_oracle_cases=[],
            hidden_tests=[],
            previous_solution=None,
            previous_errors=[],
            prompt_runner=prompt_runner,
        )
        self.assertIn("You are an expert Python reference-solution generator", prompt)
        self.assertIn("[SEED PROBLEM]", prompt)
        self.assertIn("[SEED EXECUTABLE CODE]", prompt)
        self.assertIn("[SCALED FINAL USER PROMPT]", prompt)
        self.assertIn("[SCALED ORACLE CASES]", prompt)
        self.assertNotIn("[SEED GOLD SOLUTION]", prompt)
        self.assertNotIn("solution_form", prompt)

    def test_scaled_gold_repair_prompt_uses_jinja_template(self) -> None:
        env = self._env().model_copy(update={"user_prompt": "Return x + 1 while preserving semantics."})
        prompt_runner = PromptRunner(Path(__file__).resolve().parent.parent / "prompts")
        prompt = _build_scaled_gold_prompt(
            env=env,
            operator_instances=[],
            semantic_test_specs=[],
            scaled_oracle_cases=[],
            hidden_tests=[],
            previous_solution="return x",
            previous_errors=["compile_failed"],
            prompt_runner=prompt_runner,
            repair_context={
                "compile_result": {"compile_passed": False},
                "execution_result": {"execution_passed": False},
                "visible_test_result": {"status": "not_separately_executed"},
                "hidden_test_result": {"passed": False, "errors": ["x"]},
                "failure_summary": ["compile_failed"],
            },
        )
        self.assertIn("You are an expert Python debugging assistant", prompt)
        self.assertIn("[EXISTING SCALED ORACLE CASES]", prompt)
        self.assertIn("[PREVIOUS SCALED GOLD SOLUTION]", prompt)


if __name__ == "__main__":
    unittest.main()
