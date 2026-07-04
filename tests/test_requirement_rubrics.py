from __future__ import annotations

import unittest

from medenvscale.rubrics import build_requirement_rubrics, score_requirement_rubrics
from medenvscale.schemas import ExecutableEnvSpec


class RequirementRubricTests(unittest.TestCase):
    def test_build_requirement_rubrics_links_oracle_cases(self) -> None:
        env = ExecutableEnvSpec(
            env_id="env_rubric",
            original_task_id="task",
            split="train",
            problem="Return zero.",
            context="",
            signature="def solve(x)",
            solution_form="function",
            primary_domain="code",
            primary_task_type="code",
            gold_solution="def solve(x):\n    return 0\n",
            output_requirements=["Return 0 when x is 0."],
            validated_oracle_cases=[
                {
                    "case_id": "case_zero",
                    "covered_requirements": ["Return 0 when x is 0."],
                    "expected_output_signature": {"return_value": 0},
                }
            ],
        )

        rubrics = build_requirement_rubrics(env)

        self.assertEqual(len(rubrics), 1)
        self.assertEqual(rubrics[0]["evidence_type"], "oracle_case")
        self.assertEqual(rubrics[0]["covered_by_cases"], ["case_zero"])

    def test_score_requirement_rubrics_uses_covered_case_pass_rate(self) -> None:
        rubrics = [
            {
                "rubric_id": "r1",
                "requirement": "Return 0.",
                "category": "scaled_requirement",
                "weight": 2.0,
                "covered_by_cases": ["case_a", "case_b"],
            }
        ]

        failed = score_requirement_rubrics(
            rubrics,
            [{"case_id": "case_a", "passed": True}, {"case_id": "case_b", "passed": False}],
        )
        passed = score_requirement_rubrics(
            rubrics,
            [{"case_id": "case_a", "passed": True}, {"case_id": "case_b", "passed": True}],
        )

        self.assertEqual(failed["rubric_score"], 0.5)
        self.assertEqual(failed["rubric_scores"][0]["score"], 0.5)
        self.assertEqual(failed["rubric_scores"][0]["passed_covered_cases"], 1)
        self.assertEqual(failed["rubric_scores"][0]["total_covered_cases"], 2)
        self.assertEqual(failed["rubric_scores"][0]["satisfied"], False)
        self.assertEqual(passed["rubric_score"], 1.0)
        self.assertEqual(passed["rubric_scores"][0]["satisfied"], True)


if __name__ == "__main__":
    unittest.main()
