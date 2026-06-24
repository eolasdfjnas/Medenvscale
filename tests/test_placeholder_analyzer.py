from __future__ import annotations

import unittest

from medenvscale.ingest.placeholder_analyzer import detect_solution_form


class PlaceholderAnalyzerTests(unittest.TestCase):
    def test_detects_expression_completion(self) -> None:
        form = detect_solution_form(
            problem="Return the expression.",
            context="def score(x):\n    return <<insert solution here>>\n",
            signature="def score(x):",
        )
        self.assertEqual(form, "expression_completion")

    def test_detects_function_body(self) -> None:
        form = detect_solution_form(
            problem="Complete the body.",
            context="def score(x):\n    <<insert solution here>>\n",
            signature="def score(x):",
        )
        self.assertEqual(form, "function_body")

    def test_detects_patch_or_bugfix(self) -> None:
        form = detect_solution_form(
            problem="Patch the broken validator.",
            context="def validate(x):\n    <<insert solution here>>\n",
            signature="def validate(x):",
        )
        self.assertEqual(form, "patch_or_bugfix")
