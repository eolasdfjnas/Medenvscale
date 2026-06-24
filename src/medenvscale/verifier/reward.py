from __future__ import annotations


def compute_reward(weighted_rubric_score: float, safety_pass: bool, process_score: float = 0.5, tool_state_score: float = 0.0) -> float:
    unsafe_penalty = 0.5 if not safety_pass else 0.0
    return (1.0 if safety_pass else 0.0) * weighted_rubric_score + tool_state_score + process_score - unsafe_penalty
