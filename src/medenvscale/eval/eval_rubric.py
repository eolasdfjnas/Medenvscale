from __future__ import annotations

from medenvscale.utils import read_jsonl


def evaluate_rubrics(processed_dir: str) -> dict:
    rubrics = read_jsonl(f"{processed_dir}/rubrics.jsonl")
    return {"num_rubrics": len(rubrics)}
