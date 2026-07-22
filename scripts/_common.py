from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None, help="Dataset subdirectory name, for example: biocoder")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=None, help="Randomly sample input tasks before applying --limit.")
    parser.add_argument("--llm_mode", default=None)
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers for supported stages.")
    parser.add_argument("--resume", action="store_true", help="Resume supported stages from complete outputs or checkpoints.")
    parser.add_argument("--resume_stage05", action="store_true", help="Deprecated alias for Stage05 resume; prefer --resume.")
    return parser
