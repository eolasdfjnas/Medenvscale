from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medenvscale.utils import ensure_dir, stable_hash


class DiskCache:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = ensure_dir(base_dir)

    def _path(self, key: dict[str, Any]) -> Path:
        return self.base_dir / f"{stable_hash(key)}.json"

    def get(self, key: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: dict[str, Any], value: dict[str, Any]) -> None:
        path = self._path(key)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
