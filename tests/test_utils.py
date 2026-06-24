from __future__ import annotations

import unittest

from medenvscale.utils import stable_hash


class UtilsTests(unittest.TestCase):
    def test_stable_hash_handles_mixed_key_types(self) -> None:
        value = {
            "context": {
                "expected_output_signature": {
                    "return_value": {
                        0: "zero",
                        "1": "one",
                    }
                }
            }
        }

        first = stable_hash(value)
        second = stable_hash(value)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)


if __name__ == "__main__":
    unittest.main()
