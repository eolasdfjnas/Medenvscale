from __future__ import annotations

from .common import StrictBaseModel


class QuestionPoint(StrictBaseModel):
    env_id: str
    point_id: str
    title: str
    description: str
    related_axes: list[str] = []
    required: bool = True
