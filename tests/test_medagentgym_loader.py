from __future__ import annotations

import unittest

from medenvscale.ingest.load_medagentgym import collect_resource_files
from medenvscale.ingest.normalize_medagentgym import normalize_row


class MedAgentGymLoaderTests(unittest.TestCase):
    def test_normalize_row_keeps_core_code_fields(self) -> None:
        row = {
            "idx": "demo_001",
            "problem": "Complete the function.",
            "context": "def f(x):\n    <<insert solution here>>\n",
            "signature": "def f(x):",
            "solution": "return x + 1",
            "resources": [{"path": "data/input.csv"}],
        }
        normalized = normalize_row(row, "train", 1)
        self.assertEqual(normalized.problem, row["problem"])
        self.assertEqual(normalized.context, row["context"])
        self.assertEqual(normalized.signature, row["signature"])
        self.assertEqual(normalized.solution, row["solution"])
        self.assertTrue(normalized.has_placeholder)
        self.assertEqual(normalized.resource_files, ["data/input.csv"])

    def test_collect_resource_files(self) -> None:
        row = {"resources": ["a.csv", {"path": "b.db"}], "artifacts": [{"name": "note.md"}]}
        self.assertEqual(collect_resource_files(row), ["a.csv", "b.db", "note.md"])
