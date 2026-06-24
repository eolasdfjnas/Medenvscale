from __future__ import annotations

from typing import Literal

from .common import StrictBaseModel


class RubricCriterion(StrictBaseModel):
    env_id: str
    rubric_id: str
    source_point_id: str
    criterion: str
    score_type: Literal["binary", "scalar"]
    weight: int
    category: str
    fail_cap: float | None = None
