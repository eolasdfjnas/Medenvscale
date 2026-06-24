from __future__ import annotations

from typing import Any

from medenvscale.schemas import MedQAItem
from medenvscale.utils import slugify

from .load_medqa import extract_answer_key, extract_answer_text, extract_options


def normalize_row(row: dict[str, Any], source_split: str, index: int) -> MedQAItem:
    options = extract_options(row)
    answer_key = extract_answer_key(row, options)
    answer_text = extract_answer_text(row, options)
    question = str(row.get("question") or row.get("query") or row.get("problem") or "").strip()
    medqa_id = str(row.get("medqa_id") or row.get("id") or f"medqa_en_{source_split}_{index:06d}")
    if not medqa_id.startswith("medqa_en_"):
        medqa_id = f"medqa_en_{source_split}_{slugify(medqa_id)}"
    return MedQAItem(
        medqa_id=medqa_id,
        source_split=source_split,
        question=question,
        options=options,
        answer_key=answer_key,
        answer_text=answer_text,
        explanation=row.get("explanation"),
        raw_subject=row.get("subject") or row.get("topic"),
        raw_metadata={k: v for k, v in row.items() if k not in {"question", "options", "answer", "answer_idx", "explanation", "subject"}},
    )


def normalize_rows(rows: list[dict[str, Any]], source_split: str = "train") -> list[MedQAItem]:
    normalized: list[MedQAItem] = []
    for index, row in enumerate(rows, start=1):
        row_split = str(row.get("_hf_split") or source_split)
        normalized.append(normalize_row(row, row_split, index))
    return normalized
