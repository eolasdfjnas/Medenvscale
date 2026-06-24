from __future__ import annotations

from medenvscale.schemas import SafetyGate
from medenvscale.safety.safety_gate import apply_safety_gate


def verify_safety(answer: str, gate: SafetyGate | None) -> dict:
    if gate is None:
        return {"pass_gate": True, "score_cap": None, "failed_patterns": [], "missing_required_actions": []}
    return apply_safety_gate(answer, gate)
