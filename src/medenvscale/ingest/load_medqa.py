from __future__ import annotations

from pathlib import Path
from typing import Any

from medenvscale.utils import read_jsonl


def load_raw_medqa(raw_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(raw_path)
    if limit is not None:
        return rows[:limit]
    return rows


def extract_options(row: dict[str, Any]) -> dict[str, str]:
    if isinstance(row.get("options"), dict):
        return {str(k): str(v) for k, v in row["options"].items()}
    for key in ("opa", "cop", "choices"):
        value = row.get(key)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, list):
            return {chr(65 + idx): str(item) for idx, item in enumerate(value)}
    option_keys = [key for key in row if len(key) == 1 and key.isalpha() and key.upper() == key]
    if option_keys:
        return {key: str(row[key]) for key in sorted(option_keys)}
    return {}


def extract_answer_key(row: dict[str, Any], options: dict[str, str]) -> str | None:
    for key in ("answer_idx", "answer_key", "label", "correct_option"):
        if row.get(key):
            return str(row[key]).strip()
    answer_text = extract_answer_text(row, options)
    if answer_text:
        for key, value in options.items():
            if value == answer_text:
                return key
    return None


def extract_answer_text(row: dict[str, Any], options: dict[str, str]) -> str | None:
    for key in ("answer", "answer_text", "label_text"):
        if row.get(key):
            return str(row[key]).strip()
    answer_key = row.get("answer_idx") or row.get("answer_key")
    if answer_key and str(answer_key) in options:
        return options[str(answer_key)]
    return None
