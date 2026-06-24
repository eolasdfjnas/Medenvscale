from __future__ import annotations

from medenvscale.utils import read_jsonl


def evaluate_safety(processed_dir: str) -> dict:
    gates = read_jsonl(f"{processed_dir}/safety_gates.jsonl")
    return {"num_safety_gates": len(gates)}
