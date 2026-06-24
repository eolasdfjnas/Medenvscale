from __future__ import annotations

from typing import Any

from .common import StrictBaseModel


class MedQAItem(StrictBaseModel):
    medqa_id: str
    source_split: str
    question: str
    options: dict[str, str]
    answer_key: str | None = None
    answer_text: str | None = None
    explanation: str | None = None
    raw_subject: str | None = None
    raw_metadata: dict[str, Any] = {}
