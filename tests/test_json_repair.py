from __future__ import annotations

import unittest

from medenvscale.llm.json_repair import parse_json_payload


class JsonRepairTests(unittest.TestCase):
    def test_parse_json_payload_escapes_raw_newlines_inside_strings(self) -> None:
        raw = '{"scaled_executable_gold_code": "line1\nline2", "gold_changed": true}'
        parsed = parse_json_payload(raw)
        self.assertEqual(parsed["scaled_executable_gold_code"], "line1\nline2")
        self.assertTrue(parsed["gold_changed"])

    def test_parse_json_payload_extracts_outer_object(self) -> None:
        raw = 'prefix text {"scaled_oracle_cases": [{"case_id": "c1"}]} suffix text'
        parsed = parse_json_payload(raw)
        self.assertEqual(parsed["scaled_oracle_cases"][0]["case_id"], "c1")

    def test_parse_json_payload_repairs_trailing_commas(self) -> None:
        raw = '{"scaled_oracle_cases": [{"case_id": "c1",}],}'
        parsed = parse_json_payload(raw)
        self.assertEqual(parsed["scaled_oracle_cases"][0]["case_id"], "c1")

    def test_parse_json_payload_raises_json_decode_error_for_unrepairable_pythonish(self) -> None:
        raw = '{"repaired_code": "bad "quoted" string"}'
        with self.assertRaises(Exception) as ctx:
            parse_json_payload(raw)
        self.assertEqual(ctx.exception.__class__.__name__, "JSONDecodeError")


if __name__ == "__main__":
    unittest.main()
