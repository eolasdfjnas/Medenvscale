from __future__ import annotations

from pathlib import Path
from typing import Any

from medenvscale.utils import read_jsonl


def load_raw_medagentgym(raw_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(raw_path)
    if limit is not None:
        return rows[:limit]
    return rows


def collect_resource_files(row: dict[str, Any]) -> list[str]:
    candidates = [
        row.get("resource_files"),
        row.get("resources"),
        row.get("resource_paths"),
        row.get("files"),
        row.get("artifacts"),
    ]
    collected: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            values = candidate.values()
        elif isinstance(candidate, list):
            values = candidate
        else:
            continue
        for value in values:
            if isinstance(value, dict):
                value = value.get("path") or value.get("name") or value.get("file")
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in collected:
                collected.append(text)
    return collected
