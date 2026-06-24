from __future__ import annotations

import unittest

from medenvscale.scaling.output_constraints import check_output_constraints, output_constraints_from_scaled_oracle_cases


class OutputConstraintTests(unittest.TestCase):
    def test_float_dict_contains_uses_tolerance(self) -> None:
        result = check_output_constraints(
            {
                "return_value": {
                    "Delta_f": 1.0089163936550536,
                    "dDelta_f": 0.008336174802093614,
                },
                "stdout": "",
                "file_artifacts": [],
            },
            {
                "checks": [
                    {
                        "check_id": "bar_values",
                        "kind": "return_value",
                        "rule": "dict_contains",
                        "params": {
                            "value": {
                                "Delta_f": 1.0089163936551835,
                                "dDelta_f": 0.008336174802093615,
                            }
                        },
                        "severity": "hard",
                    }
                ]
            },
        )

        self.assertTrue(result["passed"])

    def test_structured_numeric_return_text_uses_numeric_tolerance(self) -> None:
        result = check_output_constraints(
            {
                "return_value": "[[20. 21. 22.]\n [25. 26. 27.]]",
                "return_type": "ndarray",
                "stdout": "",
                "file_artifacts": [],
            },
            {
                "checks": [
                    {
                        "check_id": "matrix_value",
                        "kind": "return_value",
                        "rule": "equals",
                        "params": {"value": "array([[20., 21., 22.],\n       [25., 26., 27.]])"},
                        "severity": "hard",
                    }
                ]
            },
        )

        self.assertTrue(result["passed"])

    def test_structured_numeric_string_repr_does_not_force_str_return_type(self) -> None:
        spec = output_constraints_from_scaled_oracle_cases(
            [
                {
                    "case_id": "matrix_case",
                    "expected_output_signature": {
                        "return_type": "str",
                        "return_value": "array([[20., 21., 22.],\n       [25., 26., 27.]])",
                    },
                }
            ]
        )

        result = check_output_constraints(
            {
                "return_value": "[[20. 21. 22.]\n [25. 26. 27.]]",
                "return_type": "ndarray",
                "stdout": "",
                "file_artifacts": [],
            },
            spec,
        )

        self.assertTrue(result["passed"])

    def test_stdout_contains_normalizes_whitespace(self) -> None:
        result = check_output_constraints(
            {
                "return_value": None,
                "stdout": "0          1    700   800    geneE",
                "file_artifacts": [],
            },
            {
                "checks": [
                    {
                        "check_id": "df_line",
                        "kind": "stdout",
                        "rule": "contains",
                        "params": {"text": "0          1    700  800    geneE"},
                        "severity": "soft",
                    }
                ]
            },
        )

        self.assertTrue(result["passed"])

    def test_invalid_stdout_regex_becomes_failed_check_not_exception(self) -> None:
        result = check_output_constraints(
            {"return_value": None, "stdout": "abc", "file_artifacts": []},
            {
                "checks": [
                    {
                        "check_id": "bad_regex",
                        "kind": "stdout",
                        "rule": "regex_match",
                        "params": {"pattern": "[seqs-fp]"},
                        "severity": "soft",
                    }
                ]
            },
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_checks"][0]["check_id"], "bad_regex")
        self.assertIn("invalid stdout regex", result["failed_checks"][0]["reason"])

    def test_file_artifact_path_exists_matches_basename_for_harness_prefix(self) -> None:
        result = check_output_constraints(
            {
                "return_value": None,
                "stdout": "",
                "file_artifacts": [{"path": "scripts/test_chrom_label.svg"}],
            },
            {
                "checks": [
                    {
                        "check_id": "svg_exists",
                        "kind": "file_artifact",
                        "rule": "path_exists",
                        "params": {"path": "test_chrom_label.svg"},
                        "severity": "hard",
                    }
                ]
            },
        )
        self.assertTrue(result["passed"])

    def test_file_artifact_path_exists_keeps_directory_specific_expectations_strict(self) -> None:
        result = check_output_constraints(
            {
                "return_value": None,
                "stdout": "",
                "file_artifacts": [{"path": "scripts/test_chrom_label.svg"}],
            },
            {
                "checks": [
                    {
                        "check_id": "svg_exists",
                        "kind": "file_artifact",
                        "rule": "path_exists",
                        "params": {"path": "expected/test_chrom_label.svg"},
                        "severity": "hard",
                    }
                ]
            },
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
