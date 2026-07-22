from __future__ import annotations

import unittest

from medenvscale.scaling.case_execution import run_scaled_gold_on_validated_oracle_cases
from medenvscale.scaling.output_constraints import check_output_constraints, output_constraints_from_scaled_oracle_cases
from medenvscale.scaling.path_safety import analyze_relative_path
from medenvscale.scaling.scaled_gold_solver_generator import (
    _build_fallback_oracle_case,
    _collect_failed_case_diffs,
    _derive_scaled_gold_repair_rules,
    generate_scaled_gold_solution_if_needed,
    validate_and_repair_oracle_cases,
)
from medenvscale.scaling.seed_case_clarifier import add_seed_behavior_requirements_to_env
from medenvscale.scaling.oracle_case_validator import validate_scaled_oracle_cases
from medenvscale.scaling.output_signature import materialize_executable_gold_code
from medenvscale.schemas import DifficultyProfile, ExecutableEnvSpec


class Stage05CaseFirstTests(unittest.TestCase):
    def _env(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="env_case_first",
            original_task_id="task_case_first",
            split="train",
            problem="Write solve(x) to return 0 if x == 0 else x + 1.",
            context="def solve(x):\n    <<insert solution here>>\n",
            signature="def solve(x):",
            solution_form="function_body",
            primary_domain="scientific_software_engineering",
            primary_task_type="validation_and_code_utility",
            gold_solution="return 0 if x == 0 else x + 1",
            user_prompt="Return 0 when x is 0; otherwise return x + 1.",
            output_requirements=["Return 0 when x is 0."],
            visible_state={"placeholder_token": "<<insert solution here>>", "output_constraints": ["Return 0 when x is 0."]},
            task_state={"extra_constraints": ["Return 0 when x is 0."]},
            difficulty=DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=1, A=0, V=0, selected_axes=["C"], total_intensity=1),
            operator_instances=[
                {
                    "operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_change": True,
                    "state_updates": {
                        "visible_state_patch": {"output_constraints": ["Return 0 when x is 0."]},
                        "task_state_patch": {"extra_constraints": ["Return 0 when x is 0."]},
                    },
                }
            ],
        )

    def test_validator_rejects_empty_expected_output_signature(self) -> None:
        env = self._env()
        _, _, report, _ = validate_scaled_oracle_cases(
            env,
            env.operator_instances,
            [
                {
                    "case_id": "case_1",
                    "description": "desc",
                    "targets_operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_intent": "intent",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "x = 0",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {},
                    "covers_requirements": ["Return 0 when x is 0."],
                }
            ],
        )
        self.assertFalse(report[0]["valid"])
        self.assertIn("EMPTY_EXPECTED_OUTPUT_SIGNATURE", report[0]["failure_reasons"])

    def test_validator_rejects_unknown_operator_target(self) -> None:
        env = self._env()
        _, _, report, _ = validate_scaled_oracle_cases(
            env,
            env.operator_instances,
            [
                {
                    "case_id": "case_1",
                    "description": "desc",
                    "targets_operator_id": "op_UNKNOWN",
                    "axis": "C",
                    "semantic_intent": "intent",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "x = 0",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 0},
                    "covers_requirements": ["Return 0 when x is 0."],
                }
            ],
        )
        self.assertFalse(report[0]["valid"])
        self.assertIn("UNKNOWN_TARGET_OPERATOR_ID:op_UNKNOWN", report[0]["failure_reasons"])

    def test_validated_cases_drive_scaled_gold_execution(self) -> None:
        env = self._env()
        cases = [
            {
                "case_id": "case_zero",
                "description": "desc",
                "targets_operator_id": "op_C_001",
                "axis": "C",
                "semantic_intent": "intent",
                "target_constraint": "Return 0 when x is 0.",
                "expected_failure_mode": "mode",
                "setup_code": "x = 0",
                "call_code": "result = solve(x)",
                "expected_output_signature": {"return_type": "int", "return_value": 0},
                "covers_requirements": ["Return 0 when x is 0."],
            }
        ]
        execution = run_scaled_gold_on_validated_oracle_cases(
            env,
            "def solve(x):\n    return 0 if x == 0 else x + 1\n",
            cases,
        )
        self.assertTrue(execution["compile_passed"])
        self.assertTrue(execution["execution_passed"])
        self.assertTrue(execution["case_reports"][0]["passed"])

    def test_scaled_gold_mismatch_is_reported(self) -> None:
        env = self._env()
        cases = [
            {
                "case_id": "case_zero",
                "description": "desc",
                "targets_operator_id": "op_C_001",
                "axis": "C",
                "semantic_intent": "intent",
                "target_constraint": "Return 0 when x is 0.",
                "expected_failure_mode": "mode",
                "setup_code": "x = 0",
                "call_code": "result = solve(x)",
                "expected_output_signature": {"return_type": "int", "return_value": 0},
                "covers_requirements": ["Return 0 when x is 0."],
            }
        ]
        execution = run_scaled_gold_on_validated_oracle_cases(
            env,
            "def solve(x):\n    return x + 1\n",
            cases,
        )
        self.assertFalse(execution["execution_passed"])
        self.assertFalse(execution["case_reports"][0]["passed"])

    def test_materialize_executable_gold_code_uses_complete_program_directly(self) -> None:
        env = self._env().model_copy(
            update={
                "context": "def solve(x):\n    <<insert solution here>>\n",
                "code": (
                    "def helper(x):\n"
                    "    return x + 41\n\n"
                    "def solve(x):\n"
                    "    return helper(x)\n\n"
                    "if __name__ == '__main__':\n"
                    "    print(solve(1))\n"
                ),
            }
        )
        materialized = materialize_executable_gold_code(
            env,
            (
                "def solve(x):\n"
                "    return x + 100\n\n"
                "if __name__ == '__main__':\n"
                "    result = solve(1)\n"
            ),
        )
        self.assertIn("def solve(x):", materialized)
        self.assertIn("return x + 100", materialized)
        self.assertNotIn("def helper(x):", materialized)
        self.assertNotIn("return helper(x)", materialized)

    def test_case_rule_repair_removes_setup_stdout_and_fixture_artifacts(self) -> None:
        env = self._env().model_copy(
            update={
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "setup_code": "",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 0},
                }
            }
        )
        result = validate_and_repair_oracle_cases(
            env=env,
            operator_instances=env.operator_instances,
            oracle_case_candidates=[
                {
                    "case_id": "case_rule_repair",
                    "description": "desc",
                    "targets_operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_intent": "intent",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "artifact_path = 'fixture.txt'\nwith open(artifact_path, 'w') as f:\n    f.write('x')\nprint('setup token')",
                    "call_code": "result = {'ok': True, 'stdout': 'real token'}",
                    "expected_output_signature": {
                        "return_type": "tuple",
                        "return_value": {"ok": True, "stdout": "real token"},
                        "stdout_contains": ["setup token", "STDOUT:", "real token"],
                        "file_artifacts": [{"path": "fixture.txt"}, {"path": "output.txt"}],
                    },
                    "covers_requirements": ["Return 0 when x is 0."],
                }
            ],
            llm_client=None,
            prompt_runner=None,
            config={"stage05_cfg": {"oracle_case_repair": {"enabled": False, "max_rounds": 0}}},
        )
        repaired_case = result["scaled_oracle_cases"][0]
        self.assertEqual(repaired_case["expected_output_signature"]["return_type"], "dict")
        self.assertEqual(repaired_case["expected_output_signature"]["stdout_contains"], [])
        self.assertEqual(repaired_case["expected_output_signature"]["file_artifacts"], [{"path": "output.txt"}])
        self.assertTrue(result["oracle_case_rule_repair_report"][0]["changed"])

    def test_case_rule_repair_appends_result_assignment(self) -> None:
        env = self._env()
        result = validate_and_repair_oracle_cases(
            env=env,
            operator_instances=env.operator_instances,
            oracle_case_candidates=[
                {
                    "case_id": "case_missing_result",
                    "description": "desc",
                    "targets_operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_intent": "intent",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "x = 0",
                    "call_code": "solve(x)",
                    "expected_output_signature": {"stdout_contains": ["token"]},
                    "covers_requirements": ["Return 0 when x is 0."],
                }
            ],
            llm_client=None,
            prompt_runner=None,
            config={"stage05_cfg": {"oracle_case_repair": {"enabled": False, "max_rounds": 0}}},
        )
        repaired_case = result["scaled_oracle_cases"][0]
        self.assertIn("result = None", repaired_case["call_code"])

    def test_path_safety_allows_normalized_file_but_rejects_escape(self) -> None:
        normalized = analyze_relative_path("outputs/../result.txt", artifact=True)
        self.assertTrue(normalized.safe)
        self.assertEqual(normalized.normalized_path, "result.txt")

        escape = analyze_relative_path("../result.txt", artifact=True)
        self.assertFalse(escape.safe)
        self.assertEqual(escape.reason, "path_escapes_workdir")

        directory = analyze_relative_path("outputs/", artifact=True)
        self.assertFalse(directory.safe)
        self.assertEqual(directory.reason, "directory_file_artifact_path")

    def test_validator_rejects_case_code_path_that_normalizes_to_workdir(self) -> None:
        env = self._env()
        _, _, report, _ = validate_scaled_oracle_cases(
            env,
            env.operator_instances,
            [
                {
                    "case_id": "case_bad_path",
                    "description": "desc",
                    "case_kind": "scaled",
                    "targets_operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_intent": "Return 0 when x is 0.",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "import os\nos.makedirs('scripts/..')",
                    "call_code": "result = solve(0)",
                    "expected_output_signature": {"return_type": "int", "return_value": 0},
                    "covered_requirements": ["Return 0 when x is 0."],
                }
            ],
        )
        self.assertFalse(report[0]["valid"])
        self.assertIn("UNSAFE_CASE_PATH:setup_code:scripts/..:normalizes_to_workdir", report[0]["failure_reasons"])

    def test_validator_rejects_unstable_object_memory_address_expected_value(self) -> None:
        env = self._env()
        _, _, report, _ = validate_scaled_oracle_cases(
            env,
            env.operator_instances,
            [
                {
                    "case_id": "case_object_repr",
                    "description": "desc",
                    "case_kind": "scaled",
                    "targets_operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_intent": "Return 0 when x is 0.",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "x = 0",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {
                        "return_type": "dict",
                        "return_value_contains": {
                            "prob": "<__seed_case_runtime__.Problem object at 0x7f75c207ff40>",
                            "stable": 0,
                        },
                    },
                    "covered_requirements": ["Return 0 when x is 0."],
                }
            ],
        )

        self.assertFalse(report[0]["valid"])
        self.assertIn("UNSTABLE_EXPECTED_OUTPUT_OBJECT_MEMORY_ADDRESS", report[0]["failure_reasons"])

    def test_case_rule_repair_normalizes_and_removes_unsafe_file_artifacts(self) -> None:
        env = self._env()
        result = validate_and_repair_oracle_cases(
            env=env,
            operator_instances=env.operator_instances,
            oracle_case_candidates=[
                {
                    "case_id": "case_artifacts",
                    "description": "desc",
                    "case_kind": "scaled",
                    "targets_operator_id": "op_C_001",
                    "axis": "C",
                    "semantic_intent": "Return 0 when x is 0.",
                    "target_constraint": "Return 0 when x is 0.",
                    "expected_failure_mode": "mode",
                    "setup_code": "from pathlib import Path\nPath('fixtures/tmp.txt').write_text('x')",
                    "call_code": "result = solve(0)",
                    "expected_output_signature": {
                        "return_type": "int",
                        "return_value": 0,
                        "file_artifacts": [
                            {"path": "outputs/../result.txt"},
                            {"path": "fixtures/tmp.txt"},
                            {"path": "deblur.log"},
                            {"path": "../escape.txt"},
                            {"path": "outputs/"},
                        ],
                    },
                    "covered_requirements": ["Return 0 when x is 0."],
                }
            ],
            llm_client=None,
            prompt_runner=None,
            config={"stage05_cfg": {"oracle_case_repair": {"enabled": False, "max_rounds": 0}}},
        )
        repaired_case = result["scaled_oracle_cases"][0]
        self.assertEqual(repaired_case["expected_output_signature"]["file_artifacts"], [{"path": "result.txt"}])
        actions = result["oracle_case_rule_repair_report"][0]["actions"]
        self.assertTrue(any(action.startswith("NORMALIZED_FILE_ARTIFACT_PATHS:") for action in actions))
        self.assertTrue(any(action.startswith("REMOVED_UNSAFE_FILE_ARTIFACT_PATHS:") for action in actions))
        self.assertTrue(any(action.startswith("REMOVED_SETUP_ARTIFACT_EXPECTATIONS:") for action in actions))
        self.assertTrue(any(action.startswith("REMOVED_LOG_ARTIFACT_EXPECTATIONS:") for action in actions))

    def test_scaled_gold_repair_diff_extracts_targeted_rules(self) -> None:
        cases = [
            {
                "case_id": "case_failed",
                "setup_code": "x = 1",
                "call_code": "result = solve(x)",
                "target_constraint": "Write the expected file and print status.",
                "covered_requirements": ["Write the expected file and print status."],
                "expected_output_signature": {
                    "return_type": "dict",
                    "return_value": {"ok": True},
                    "stdout_contains": ["done"],
                    "file_artifacts": [{"path": "{out_dir}/result.txt"}],
                },
            }
        ]
        reports = [
            {
                "case_id": "case_failed",
                "passed": False,
                "expected_output_signature": cases[0]["expected_output_signature"],
                "observed_output_signature": {
                    "return_value": None,
                    "return_type": "NoneType",
                    "stdout": "",
                    "stderr": "",
                    "file_artifacts": [],
                },
                "failure_reasons": [
                    "CASE_EXECUTION_ERROR:ModuleNotFoundError:No module named 'reaction_utils'",
                    "case_failed_stdout_contains_1:stdout missing substring: done",
                    "case_failed_artifact_exists_1:missing file artifact: {out_dir}/result.txt",
                ],
            }
        ]
        diffs = _collect_failed_case_diffs(cases, reports)
        self.assertEqual(diffs[0]["missing_stdout_tokens"], ["done"])
        self.assertEqual(diffs[0]["missing_file_artifacts"], ["{out_dir}/result.txt"])
        self.assertIn("module_not_found", diffs[0]["failure_types"])
        self.assertIn("unresolved_artifact_template", diffs[0]["failure_types"])

        rules = "\n".join(_derive_scaled_gold_repair_rules(diffs))
        self.assertIn("ModuleNotFoundError repair", rules)
        self.assertIn("Template-path artifact repair", rules)
        self.assertIn("Stdout repair", rules)

    def test_fallback_oracle_case_from_seed_can_validate_against_additional_requirement(self) -> None:
        env = self._env().model_copy(
            update={
                "user_prompt": (
                    "Return 0 when x is 0; otherwise return x + 1.\n\n"
                    "Additional requirements:\n"
                    "- The function must return zero for the boundary input 0.\n"
                    "Output requirement:\n"
                ),
                "seed_execution_case": {
                    "case_id": "seed_zero",
                    "description": "seed",
                    "setup_code": "x = 0",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 0},
                },
            }
        )
        operators = [
            {
                "operator_id": "op_C_001",
                "axis": "C",
                "semantic_change": True,
                "output_requirements": ["The function must return zero for the boundary input 0."],
                "state_updates": {
                    "visible_state_patch": {"output_constraints": ["The function must return zero for the boundary input 0."]},
                    "task_state_patch": {},
                },
            }
        ]
        fallback_case = _build_fallback_oracle_case(env, operators, [])
        self.assertIsNotNone(fallback_case)
        validated, _, report, _ = validate_scaled_oracle_cases(env, operators, [fallback_case])
        self.assertEqual(len(validated), 1)
        self.assertTrue(report[0]["valid"])

    def test_m1_seed_execution_case_becomes_validated_oracle_case_and_executes(self) -> None:
        env = self._env().model_copy(
            update={
                "env_id": "env_m1_seed_case",
                "difficulty": DifficultyProfile(
                    global_level="M1",
                    H=0,
                    D=0,
                    R=0,
                    I=0,
                    E=0,
                    C=0,
                    A=0,
                    V=0,
                    selected_axes=[],
                    total_intensity=0,
                ),
                "operator_instances": [],
                "semantic_test_specs": [],
                "output_requirements": [],
                "problem": "Write solve(x) to return x + 1.",
                "user_prompt": "Write solve(x) to return x + 1.",
                "gold_solution": "def solve(x):\n    return x + 1\n",
                "seed_gold_solution": "def solve(x):\n    return x + 1\n",
                "code": "def solve(x):\n    return x + 1\n",
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "description": "Original seed case.",
                    "setup_code": "x = 2",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 3},
                },
                "seed_ground_truth_output_signature": {"return_type": "int", "return_value": 3},
                "seed_case_audit": {
                    "status": "pass",
                    "failure_reason": "",
                    "mismatch_reasons": [],
                },
            }
        )

        result = generate_scaled_gold_solution_if_needed(
            env=env,
            operator_instances=[],
            semantic_test_specs=[],
            output_constraint_spec={},
            llm_client=None,
            prompt_runner=None,
            config={"stage05_cfg": {"oracle_case_repair": {"enabled": False, "max_rounds": 0}}},
        )

        self.assertEqual(len(result["scaled_oracle_cases"]), 1)
        self.assertEqual(len(result["validated_oracle_cases"]), 1)
        self.assertEqual(result["validated_oracle_cases"][0]["case_kind"], "seed_baseline")
        self.assertEqual(result["validated_oracle_cases"][0]["case_id"], "seed_case_main")
        self.assertEqual(result["validated_oracle_cases"][0]["expected_output_signature"]["return_value"], 3)
        self.assertTrue(result["compile_passed"])
        self.assertTrue(result["visible_tests_passed"])
        self.assertTrue(result["scaled_gold_case_execution_report"][0]["passed"])
        self.assertNotIn("NO_VALIDATED_ORACLE_CASES", result["failure_reasons"])

    def test_m1_seed_baseline_allows_callable_name_as_target(self) -> None:
        env = self._env().model_copy(
            update={
                "difficulty": DifficultyProfile(
                    global_level="M1",
                    H=0,
                    D=0,
                    R=0,
                    I=0,
                    E=0,
                    C=0,
                    A=0,
                    V=0,
                    selected_axes=[],
                    total_intensity=0,
                )
            }
        )
        _, _, report, _ = validate_scaled_oracle_cases(
            env,
            [],
            [
                {
                    "case_id": "seed_case_main",
                    "description": "seed",
                    "case_kind": "seed_baseline",
                    "targets_operator_id": "solve",
                    "axis": "M1",
                    "semantic_intent": "Preserve seed behavior.",
                    "target_constraint": "Preserve seed behavior.",
                    "expected_failure_mode": "breaks seed behavior",
                    "setup_code": "x = 2",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 3},
                    "covered_requirements": ["Preserve seed behavior."],
                }
            ],
        )

        self.assertTrue(report[0]["valid"])
        self.assertNotIn("UNKNOWN_TARGET_OPERATOR_ID:solve", report[0]["failure_reasons"])

    def test_context_created_files_are_removed_from_seed_artifacts(self) -> None:
        env = self._env().model_copy(
            update={
                "context": (
                    "def write_sample_files():\n"
                    "    with open('sample_reactions.tsv', 'w') as f:\n"
                    "        f.write('x')\n"
                    "write_sample_files()\n"
                    "<<insert solution here>>\n"
                ),
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "setup_code": "",
                    "call_code": "result = solve(0)",
                    "expected_output_signature": {
                        "return_type": "int",
                        "return_value": 0,
                        "file_artifacts": [{"path": "sample_reactions.tsv"}],
                    },
                },
            }
        )

        result = validate_and_repair_oracle_cases(
            env=env,
            operator_instances=[],
            oracle_case_candidates=[
                {
                    "case_id": "seed_case_main",
                    "description": "seed",
                    "case_kind": "seed_baseline",
                    "targets_operator_id": "solve",
                    "axis": "M1",
                    "semantic_intent": "Preserve seed behavior.",
                    "target_constraint": "Preserve seed behavior.",
                    "expected_failure_mode": "breaks seed behavior",
                    "setup_code": "",
                    "call_code": "result = solve(0)",
                    "expected_output_signature": {
                        "return_type": "int",
                        "return_value": 0,
                        "file_artifacts": [{"path": "sample_reactions.tsv"}],
                    },
                    "covered_requirements": ["Preserve seed behavior."],
                }
            ],
            llm_client=None,
            prompt_runner=None,
            config={"stage05_cfg": {"oracle_case_repair": {"enabled": False, "max_rounds": 0}}},
        )

        repaired = result["validated_oracle_cases"][0]
        self.assertEqual(repaired["expected_output_signature"]["file_artifacts"], [])
        actions = result["oracle_case_rule_repair_report"][0]["actions"]
        self.assertTrue(any(action.startswith("REMOVED_SETUP_ARTIFACT_EXPECTATIONS:") for action in actions))

    def test_object_repr_memory_addresses_are_stabilized_for_comparison(self) -> None:
        case = {
            "case_id": "case_object",
            "targets_operator_id": "seed_regression",
            "expected_output_signature": {
                "return_value": {
                    "prob": "<__seed_case_runtime__.Problem object at 0x7f75c207ff40>",
                    "compounds": ["A", "B"],
                }
            },
        }
        spec = output_constraints_from_scaled_oracle_cases([case])
        observed = {
            "return_value": {
                "prob": "<__case_runtime__.Problem object at 0x7f3f102b9c00>",
                "compounds": ["A", "B"],
            }
        }

        result = check_output_constraints(observed, spec)

        self.assertTrue(result["passed"], result.get("failed_checks"))

    def test_seed_behavior_requirements_are_added_to_visible_state(self) -> None:
        env = self._env().model_copy(
            update={
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "description": "Original seed case.",
                    "setup_code": "x = 2",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 3},
                },
                "seed_ground_truth_output_signature": {"return_type": "int", "return_value": 3},
            }
        )

        clarified = add_seed_behavior_requirements_to_env(env)
        requirements = clarified.visible_state["seed_behavior_requirements"]

        self.assertTrue(any("Preserve the original seed behavior for solve" in item for item in requirements))
        self.assertTrue(any("result = solve(x)" in item for item in requirements))
        self.assertTrue(any("return value must be 3" in item for item in requirements))
        self.assertTrue(any("return type must be 'int'" in item for item in requirements))
        self.assertIn(requirements[0], clarified.visible_state["execution_requirements"])
        self.assertIn(requirements[0], clarified.output_requirements)
        self.assertTrue(any("result = solve(x)" in item for item in clarified.output_requirements))

    def test_seed_behavior_requirements_preserve_existing_output_requirements(self) -> None:
        env = self._env().model_copy(
            update={
                "output_requirements": ["Existing scaled requirement."],
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "description": "Original seed case.",
                    "setup_code": "x = 2",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_value": 3},
                },
                "seed_ground_truth_output_signature": {"return_value": 3},
            }
        )

        clarified = add_seed_behavior_requirements_to_env(env)

        self.assertEqual(clarified.output_requirements[0], "Existing scaled requirement.")
        self.assertTrue(any("Preserve the original seed behavior for solve" in item for item in clarified.output_requirements))

    def test_m2_v_only_env_gets_seed_regression_case_and_gold_gate_executes_it(self) -> None:
        env = self._env().model_copy(
            update={
                "env_id": "env_m2_seed_regression",
                "difficulty": DifficultyProfile(
                    global_level="M2",
                    H=0,
                    D=0,
                    R=0,
                    I=0,
                    E=0,
                    C=0,
                    A=0,
                    V=1,
                    selected_axes=["V"],
                    total_intensity=1,
                ),
                "operator_instances": [
                    {
                        "operator_id": "op_V_001",
                        "axis": "V",
                        "semantic_change": False,
                        "state_updates": {},
                    }
                ],
                "semantic_test_specs": [],
                "output_requirements": [],
                "problem": "Write solve(x) to return x + 1.",
                "user_prompt": "Write solve(x) to return x + 1.",
                "gold_solution": "def solve(x):\n    return x + 1\n",
                "seed_gold_solution": "def solve(x):\n    return x + 1\n",
                "code": "def solve(x):\n    return x + 1\n",
                "seed_execution_case": {
                    "case_id": "seed_case_main",
                    "description": "Original seed case.",
                    "setup_code": "x = 2",
                    "call_code": "result = solve(x)",
                    "expected_output_signature": {"return_type": "int", "return_value": 3},
                },
                "seed_ground_truth_output_signature": {"return_type": "int", "return_value": 3},
                "seed_case_audit": {
                    "status": "pass",
                    "failure_reason": "",
                    "mismatch_reasons": [],
                },
            }
        )

        result = generate_scaled_gold_solution_if_needed(
            env=env,
            operator_instances=env.operator_instances,
            semantic_test_specs=[],
            output_constraint_spec={},
            llm_client=None,
            prompt_runner=None,
            config={"stage05_cfg": {"oracle_case_repair": {"enabled": False, "max_rounds": 0}}},
        )

        self.assertEqual(result["validated_oracle_cases"][0]["case_kind"], "seed_regression")
        self.assertEqual(result["validated_oracle_cases"][0]["case_id"], "regression_seed_case_main")
        self.assertEqual(result["validated_oracle_cases"][0]["expected_output_signature"]["return_value"], 3)
        self.assertTrue(result["compile_passed"])
        self.assertTrue(result["visible_tests_passed"])
        self.assertTrue(result["scaled_gold_case_execution_report"][0]["passed"])
