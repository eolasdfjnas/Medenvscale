from __future__ import annotations

from medenvscale.utils import read_jsonl


def evaluate_prm(processed_dir: str) -> dict:
    rows = read_jsonl(f"{processed_dir}/prm_steps.jsonl")
    steps = sum(len(row["steps"]) for row in rows)
    return {"num_samples": len(rows), "num_steps": steps}
