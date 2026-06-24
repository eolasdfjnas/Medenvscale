from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["tests"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in args.paths:
        suite.addTests(loader.discover(start_dir=str(root / path)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
