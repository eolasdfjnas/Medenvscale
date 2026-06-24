from __future__ import annotations

from typing import Literal

from .common import StrictBaseModel


class SafetyGate(StrictBaseModel):
    env_id: str
    gate_id: str
    risk_level: Literal["R1", "R2", "R3", "R4", "R5"]
    unsafe_patterns: list[str]
    required_actions: list[str]
    score_cap_if_failed: float
    fatal_if_failed: bool = False
