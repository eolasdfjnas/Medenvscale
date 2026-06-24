from __future__ import annotations

from typing import Literal

from .common import StrictBaseModel


class DifficultyProfile(StrictBaseModel):
    global_level: Literal["M1", "M2", "M3", "M4"]
    D: int
    C: int
    A: int
    V: int
    selected_axes: list[str]
    total_intensity: int = 0
    applied_operators: list[str] = []
