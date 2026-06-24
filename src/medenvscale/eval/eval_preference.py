from __future__ import annotations

from medenvscale.utils import read_jsonl


def evaluate_preference(processed_dir: str) -> dict:
    rows = read_jsonl(f"{processed_dir}/preference.jsonl")
    margin = sum(row["chosen_score"] - row["rejected_score"] for row in rows) / max(len(rows), 1)
    return {"num_pairs": len(rows), "avg_margin": margin}
