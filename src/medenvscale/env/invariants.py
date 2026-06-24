from __future__ import annotations

from medenvscale.schemas import ClinicalEnvironment


def validate_environment_invariants(environment: ClinicalEnvironment) -> list[str]:
    errors: list[str] = []
    if not environment.patient_state:
        errors.append("missing_patient_state")
    if not environment.gold_state:
        errors.append("missing_gold_state")
    if not environment.user_prompt:
        errors.append("missing_user_prompt")
    if environment.level == "M4" and environment.difficulty.R >= 4 and not environment.safety_gate_id:
        errors.append("high_risk_missing_safety_gate")
    return errors
