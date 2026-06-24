from __future__ import annotations

import unittest
from pathlib import Path

from medenvscale.scaling.oracle_case_validator import validate_scaled_oracle_cases
from medenvscale.schemas import DifficultyProfile, ExecutableEnvSpec, ToolBudget, ToolConfig
from medenvscale.config import resolve_dataset_config_path_with_fallback
from medenvscale.utils import load_yaml
from medenvscale.validation.artifact_admission_gate import run_pipeline_artifact_admission_gate
from medenvscale.validation.gold_case_execution_gate import run_gold_case_execution_gate
from medenvscale.validation.oracle_case_quality_gate import run_oracle_case_quality_gate
from medenvscale.validation.stage05_gate_runner import run_stage05_gates


class Stage05CaseFirstGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent.parent
        cls.tool_pool_cfg = load_yaml(root / "configs" / "biocoder" / "tool_pool.yaml")
        cls.budgets_cfg = load_yaml(
            resolve_dataset_config_path_with_fallback(
                root,
                "m_level_budgets_4axis.yaml",
                "m_level_budgets_7axis.yaml",
                dataset="biocoder",
            )
        )
        cls.stage05_cfg = load_yaml(root / "configs" / "biocoder" / "medagentgym_pilot.yaml").get("stage05", {})

    def _valid_tool_config(self, env_id: str = "env_demo_M2") -> dict:
        available_tools = [
            {
                "tool_name": "get_task_context",
                "description": "desc",
                "input_schema": {"window": "integer"},
                "output_schema": {"ok": "boolean"},
                "when_to_use": "Use to summarize the task before coding.",
            }
        ]
        return ToolConfig(
            env_id=env_id,
            global_level="M2",
            planning_source="llm",
            allowed_tools=available_tools,
            tool_budget=ToolBudget(max_total_tool_calls=3, max_calls_per_tool={"get_task_context": 1}, max_validation_calls=1),
            output_requirement={"output_format": "code", "required_fields": [], "forbidden_fields": [], "strict": True},
            tool_choice_reason="valid",
            budget_reason="valid",
            related_axes=["C"],
            validation_trace=[],
        ).model_dump()

    def _seed_env(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="seed_demo",
            original_task_id="demo_task",
            split="train",
            problem="Write solve(x) to return x + 1.",
            context="def solve(x):\n    <<insert solution here>>\n",
            signature="def solve(x):",
            solution_form="function_body",
            primary_domain="scientific_software_engineering",
            primary_task_type="validation_and_code_utility",
            gold_solution="return x + 1",
            visible_state={"placeholder_token": "<<insert solution here>>"},
            verifier_state={"verifier_type_hint": "unit_test"},
            difficulty=DifficultyProfile(global_level="M1", H=0, D=0, R=0, I=0, E=0, C=0, A=0, V=0, selected_axes=[], total_intensity=0),
            tool_config={},
            scaling={"axis_intensity": {"H": 0, "R": 0, "I": 0, "E": 0, "C": 0, "A": 0, "V": 0}},
            scaling_plan={"axis_intensity": {"H": 0, "R": 0, "I": 0, "E": 0, "C": 0, "A": 0, "V": 0}},
            verifier_spec={"verifier_id": "verifier_seed_demo", "env_id": "seed_demo", "verifier_type": "unit_test", "solution_form": "function_body", "checks": [], "hidden_tests": [], "static_checks": [], "generated_from_operator_ids": []},
        )

    def _operator_instances(self) -> list[dict]:
        return [
            {
                "operator_id": "op_C_001",
                "axis": "C",
                "semantic_change": True,
                "state_updates": {
                    "visible_state_patch": {"output_constraints": ["Return 0 when x is 0."]},
                    "task_state_patch": {"extra_constraints": ["Return 0 when x is 0."]},
                    "gold_state_patch": {"gold_changed": True},
                    "verifier_state_patch": {"constraint_checks": ["zero-case"]},
                    "test_state_patch": {"oracle_cases": ["case_zero"]},
                },
            }
        ]

    def _scaled_env(self) -> ExecutableEnvSpec:
        seed = self._seed_env()
        valid_cases = [
            {
                "case_id": "case_zero",
                "description": "Check zero input special case.",
                "targets_operator_id": "op_C_001",
                "axis": "C",
                "semantic_intent": "Verify the x_equals_zero special branch produces integer_zero_result.",
                "target_constraint": "For x_equals_zero, solve must use zero_branch handling and produce integer_zero_result.",
                "expected_failure_mode": "seed behavior returns 1 instead of integer_zero_result for x_equals_zero",
                "setup_code": "x = 0",
                "call_code": "result = solve(x)",
                "assertion_code": "",
                "covers_requirements": ["Return 0 when x is 0."],
                "expected_output_signature": {"return_type": "int", "return_value": 0},
            },
            {
                "case_id": "case_zero_float",
                "description": "Check zero-like float edge constraint case.",
                "targets_operator_id": "op_C_001",
                "axis": "C",
                "semantic_intent": "Verify the float_zero variant follows the zero_branch and yields integer_zero_result.",
                "target_constraint": "For float_zero input, solve must use zero_branch handling and produce integer_zero_result.",
                "expected_failure_mode": "seed behavior returns 1.0 instead of integer_zero_result for float_zero",
                "setup_code": "x = 0.0",
                "call_code": "result = solve(x)",
                "assertion_code": "",
                "covers_requirements": ["Return 0 when x is 0."],
                "expected_output_signature": {"return_type": "int", "return_value": 0},
            },
            {
                "case_id": "case_zero_bool",
                "description": "Check boolean false edge constraint case.",
                "targets_operator_id": "op_C_001",
                "axis": "C",
                "semantic_intent": "Verify the falsey_zero boolean variant follows the zero_branch and yields integer_zero_result.",
                "target_constraint": "For falsey_zero input, solve must use zero_branch handling and produce integer_zero_result.",
                "expected_failure_mode": "seed behavior returns 1 instead of integer_zero_result for falsey_zero",
                "setup_code": "x = False",
                "call_code": "result = solve(x)",
                "assertion_code": "",
                "covers_requirements": ["Return 0 when x is 0."],
                "expected_output_signature": {"return_type": "int", "return_value": 0},
            },
        ]
        validation_report = validate_scaled_oracle_cases(
            env=seed.model_copy(
                update={
                    "env_id": "env_demo_M2",
                    "difficulty": DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=1, A=0, V=0, selected_axes=["C"], total_intensity=1),
                    "user_prompt": "Return 0 when x is 0; otherwise return x + 1.",
                    "output_requirements": ["Return 0 when x is 0."],
                    "tool_config": self._valid_tool_config(),
                    "scaling_plan": {"global_level": "M2", "axis_intensity": {"H": 0, "R": 0, "I": 0, "E": 0, "C": 1, "A": 0, "V": 0}},
                    "operator_instances": self._operator_instances(),
                }
            ),
            operator_instances=self._operator_instances(),
            cases=valid_cases,
        )[2]
        return seed.model_copy(
            update={
                "env_id": "env_demo_M2",
                "user_prompt": "Return 0 when x is 0; otherwise return x + 1.",
                "seed_gold_solution": "return x + 1",
                "scaled_gold_solution": "def solve(x):\n    return 0 if x == 0 else x + 1\n",
                "scaled_executable_gold_code": "def solve(x):\n    return 0 if x == 0 else x + 1\n",
                "gold_solution": "def solve(x):\n    return 0 if x == 0 else x + 1\n",
                "difficulty": DifficultyProfile(global_level="M2", H=0, D=0, R=0, I=0, E=0, C=1, A=0, V=0, selected_axes=["C"], total_intensity=1),
                "tool_config": self._valid_tool_config(),
                "visible_state": {"placeholder_token": "<<insert solution here>>", "output_constraints": ["Return 0 when x is 0."]},
                "task_state": {"extra_constraints": ["Return 0 when x is 0."], "required_capabilities": ["python"]},
                "gold_state": {"gold_changed": True, "answer_invariant": False, "gold_change_reason": "new zero-input branch", "seed_gold_compatible_with_scaled_task": False},
                "scaled_oracle_cases": valid_cases,
                "validated_oracle_cases": valid_cases,
                "oracle_case_validation_report": validation_report,
                "scaled_gold_case_execution_report": [
                    {
                        "env_id": "env_demo_M2",
                        "level": "M2",
                        "case_id": case["case_id"],
                        "passed": True,
                        "observed_output_signature": {
                            "return_value": 0,
                            "return_type": "int",
                            "stdout": "",
                            "file_artifacts": [],
                        },
                        "expected_output_signature": case["expected_output_signature"],
                        "failure_reasons": [],
                    }
                    for case in valid_cases
                ],
                "output_requirements": ["Return 0 when x is 0."],
                "scaling_plan": {"global_level": "M2", "axis_intensity": {"H": 0, "R": 0, "I": 0, "E": 0, "C": 1, "A": 0, "V": 0}},
                "operator_instances": self._operator_instances(),
                "hidden_tests": [],
                "hidden_tests_mode": "disabled_in_case_first_stage05",
                "stage05_quality_report": {"final_decision": "clean"},
                "verifier_spec": {"verifier_id": "verifier_env_demo_M2", "env_id": "env_demo_M2", "verifier_type": "unit_test", "solution_form": "function_body", "checks": [], "hidden_tests": [], "static_checks": [], "generated_from_operator_ids": ["op_C_001"]},
            }
        )

    def test_oracle_case_quality_gate_rejects_missing_result_assignment(self) -> None:
        env = self._scaled_env().model_copy(
            update={
                "scaled_oracle_cases": [
                    {
                        "case_id": "bad_case",
                        "description": "bad",
                        "targets_operator_id": "op_C_001",
                        "axis": "C",
                        "semantic_intent": "intent",
                        "target_constraint": "Return 0 when x is 0.",
                        "expected_failure_mode": "bad",
                        "setup_code": "x = 0",
                        "call_code": "solve(x)",
                        "assertion_code": "",
                        "covers_requirements": ["Return 0 when x is 0."],
                        "expected_output_signature": {"return_type": "int", "return_value": 0},
                    }
                ],
                "validated_oracle_cases": [],
                "oracle_case_validation_report": [
                    {
                        "env_id": "env_demo_M2",
                        "level": "M2",
                        "case_id": "bad_case",
                        "valid": False,
                        "failure_reasons": ["CALL_CODE_MUST_ASSIGN_RESULT"],
                        "targets_operator_id": "op_C_001",
                        "covers_requirements": ["Return 0 when x is 0."],
                        "axis": "C",
                    }
                ],
            }
        )
        result = run_oracle_case_quality_gate({"scaled_task": env}, config={"stage05_cfg": self.stage05_cfg})
        self.assertEqual(result["severity"], "hard_fail")
        self.assertIn("NO_VALIDATED_ORACLE_CASES", result["failure_reasons"])

    def test_oracle_case_quality_gate_skips_generic_output_requirements(self) -> None:
        env = self._scaled_env().model_copy(
            update={
                "output_requirements": [
                    "Scaled executable cases must expose the new requirement introduced by this operator.",
                ],
            }
        )

        result = run_oracle_case_quality_gate({"scaled_task": env}, config={"stage05_cfg": self.stage05_cfg})

        self.assertTrue(result["passed"])
        self.assertEqual(result["severity"], "pass_with_warning")
        self.assertIn("GENERIC_OUTPUT_REQUIREMENTS_SKIPPED", result["warnings"])
        self.assertNotIn("NO_CASE_COVERS_NEW_REQUIREMENT", result["failure_reasons"])
        self.assertEqual(result["evidence"]["coverage_targets"], [])

    def test_oracle_case_quality_gate_matches_specific_requirement_tokens(self) -> None:
        concrete_cases = []
        for case in self._scaled_env().validated_oracle_cases:
            concrete_case = dict(case)
            concrete_case["covers_requirements"] = [
                "Parse read identifiers with a multi-character separator while preserving extra separator chunks for barcode and UMI extraction.",
            ]
            concrete_cases.append(concrete_case)
        env = self._scaled_env().model_copy(
            update={
                "validated_oracle_cases": concrete_cases,
                "output_requirements": [
                    "Read ids using a multi-character separator must still parse the barcode and UMI when extra separators are present.",
                ],
            }
        )

        result = run_oracle_case_quality_gate({"scaled_task": env}, config={"stage05_cfg": self.stage05_cfg})

        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"]["requirement_coverage_passed"], True)
        self.assertEqual(
            result["evidence"]["coverage_targets"],
            ["Read ids using a multi-character separator must still parse the barcode and UMI when extra separators are present."],
        )

    def test_oracle_case_quality_gate_matches_case_semantic_evidence_with_synonyms(self) -> None:
        concrete_cases = []
        for case in self._scaled_env().validated_oracle_cases:
            concrete_case = dict(case)
            concrete_case["covered_requirements"] = [
                "Add constraint that the function must not modify the input row list (this_row).",
            ]
            concrete_case["target_constraint"] = "The function must not modify the input row list."
            concrete_cases.append(concrete_case)
        env = self._scaled_env().model_copy(
            update={
                "validated_oracle_cases": concrete_cases,
                "output_requirements": [
                    "Scaled executable cases must prove no mutation of input row.",
                ],
            }
        )

        result = run_oracle_case_quality_gate({"scaled_task": env}, config={"stage05_cfg": self.stage05_cfg})

        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"]["requirement_coverage_passed"], True)
        self.assertNotIn("NO_CASE_COVERS_NEW_REQUIREMENT", result["failure_reasons"])

    def test_gold_case_execution_gate_rejects_failed_case(self) -> None:
        env = self._scaled_env().model_copy(
            update={
                "scaled_gold_case_execution_report": [
                    {
                        "env_id": "env_demo_M2",
                        "level": "M2",
                        "case_id": "case_zero",
                        "passed": False,
                        "observed_output_signature": {"return_value": 1, "return_type": "int", "stdout": "", "file_artifacts": []},
                        "expected_output_signature": {"return_type": "int", "return_value": 0},
                        "failure_reasons": ["case_zero_return_value_equals:expected return value 0, got 1"],
                    }
                ]
            }
        )
        result = run_gold_case_execution_gate({"scaled_task": env})
        self.assertEqual(result["severity"], "hard_fail")
        self.assertIn("SCALED_GOLD_CASE_EXECUTION_FAILED:case_zero:case_zero_return_value_equals:expected return value 0, got 1", result["failure_reasons"])

    def test_artifact_integrity_gate_rejects_missing_validated_cases(self) -> None:
        env = self._scaled_env().model_copy(update={"validated_oracle_cases": []})
        result = run_pipeline_artifact_admission_gate(
            {
                "seed_task": self._seed_env(),
                "scaled_task": env,
                "operator_realization_report": [{"operator_id": "op_C_001", "severity": "pass", "failure_reasons": [], "warnings": []}],
                "prior_gate_results": {"oracle_case_quality_gate": {"severity": "pass"}, "gold_case_execution_gate": {"severity": "pass"}},
            },
            config={"tool_pool_cfg": self.tool_pool_cfg, "budgets_cfg": self.budgets_cfg},
        )
        self.assertEqual(result["severity"], "hard_fail")
        self.assertIn("EMPTY_VALIDATED_ORACLE_CASES", result["failure_reasons"])

    def test_stage05_gate_runner_allows_clean_case_first_sample(self) -> None:
        env = self._scaled_env()
        result = run_stage05_gates(
            {"seed_task": self._seed_env(), "scaled_task": env, "operator_realization_report": [{"operator_id": "op_C_001", "severity": "pass", "failure_reasons": [], "warnings": []}]},
            config={"tool_pool_cfg": self.tool_pool_cfg, "budgets_cfg": self.budgets_cfg, "stage05_cfg": self.stage05_cfg},
        )
        self.assertTrue(result["stage05_passed"])
        self.assertEqual(result["final_decision"], "clean")
        self.assertEqual(result["gate_results"]["oracle_case_quality_gate"]["severity"], "pass")

    def test_stage05_gate_runner_rejects_m1_baseline_without_cases(self) -> None:
        env = self._seed_env()
        result = run_stage05_gates({"seed_task": env, "scaled_task": env}, config={"stage05_cfg": self.stage05_cfg})
        self.assertFalse(result["stage05_passed"])
        self.assertEqual(result["final_decision"], "rejected")
        self.assertIn("NO_VALIDATED_ORACLE_CASES", result["rejection_reasons"])

    def test_stage05_gate_runner_allows_m1_validated_seed_baseline(self) -> None:
        case = {
            "case_id": "seed_case_main",
            "description": "M1 baseline oracle case derived from the seed execution case.",
            "case_kind": "seed_baseline",
            "targets_operator_id": "",
            "axis": "M1",
            "semantic_intent": "Validate original seed behavior for solve.",
            "target_constraint": "Validate original seed behavior for solve.",
            "expected_failure_mode": "candidate does not satisfy the original seed task behavior",
            "setup_code": "x = 2",
            "call_code": "result = solve(x)",
            "assertion_code": "",
            "covered_requirements": ["Validate original seed behavior for solve."],
            "covers_requirements": ["Validate original seed behavior for solve."],
            "expected_output_signature": {"return_type": "int", "return_value": 3},
        }
        env = self._seed_env().model_copy(
            update={
                "gold_solution": "def solve(x):\n    return x + 1\n",
                "scaled_gold_solution": "def solve(x):\n    return x + 1\n",
                "scaled_executable_gold_code": "def solve(x):\n    return x + 1\n",
                "scaled_oracle_cases": [case],
                "validated_oracle_cases": [case],
                "oracle_case_validation_report": validate_scaled_oracle_cases(self._seed_env(), [], [case])[2],
                "scaled_gold_case_execution_report": [
                    {
                        "env_id": "seed_demo",
                        "level": "M1",
                        "case_id": "seed_case_main",
                        "passed": True,
                        "observed_output_signature": {
                            "return_value": 3,
                            "return_type": "int",
                            "stdout": "",
                            "file_artifacts": [],
                        },
                        "expected_output_signature": case["expected_output_signature"],
                        "failure_reasons": [],
                    }
                ],
            }
        )

        result = run_stage05_gates({"seed_task": self._seed_env(), "scaled_task": env}, config={"stage05_cfg": self.stage05_cfg})

        self.assertTrue(result["stage05_passed"])
        self.assertEqual(result["final_decision"], "clean")
        self.assertEqual(result["gate_results"]["oracle_case_quality_gate"]["severity"], "pass")
        self.assertEqual(result["gate_results"]["gold_case_execution_gate"]["severity"], "pass")
