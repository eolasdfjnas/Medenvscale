from __future__ import annotations

import re
from pathlib import Path


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_trainer_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
