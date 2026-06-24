from __future__ import annotations

from medenvscale.schemas import DifficultyProfile


LEVELS = {
    "M1": {"D": 0, "C": 0, "A": 0, "V": 0},
    "M2": {"D": 1, "C": 1, "A": 0, "V": 1},
    "M3": {"D": 1, "C": 2, "A": 1, "V": 2},
    "M4": {"D": 2, "C": 3, "A": 2, "V": 3},
}


def build_difficulty(level: str, axis_priority: list[str], operators: list[str] | None = None) -> DifficultyProfile:
    scores = LEVELS[level]
    selected_axes = [axis for axis in axis_priority if scores.get(axis, 0) > 0]
    return DifficultyProfile(
        global_level=level,
        D=scores["D"],
        C=scores["C"],
        A=scores["A"],
        V=scores["V"],
        selected_axes=selected_axes,
        total_intensity=sum(scores.values()),
        applied_operators=operators or [],
    )
