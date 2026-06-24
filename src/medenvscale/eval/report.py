from __future__ import annotations

import json
from pathlib import Path


def write_report(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
