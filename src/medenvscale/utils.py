from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _simple_yaml_load(text)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    tmp_target = target.with_name(f"{target.name}.tmp")
    with tmp_target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    tmp_target.replace(target)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def stable_hash(value: Any) -> str:
    payload = json.dumps(_normalize_hash_value(value), ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _normalize_hash_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {_normalize_hash_key(key): _normalize_hash_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_hash_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_hash_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize_hash_value(item) for item in value), key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=_json_default))
    return value


def _normalize_hash_key(key: Any) -> str:
    if isinstance(key, str):
        return f"str:{key}"
    if isinstance(key, bool):
        return f"bool:{key}"
    if isinstance(key, int):
        return f"int:{key}"
    if isinstance(key, float):
        return f"float:{key!r}"
    if key is None:
        return "none:null"
    return f"{type(key).__name__}:{str(key)}"


def slugify(text: str, max_length: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:max_length] or "item"


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm_mode", default=None)
    return parser


def project_root_from_config(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent.parent


def seeded_shuffle(items: list[Any], seed: int) -> list[Any]:
    cloned = list(items)
    rng = random.Random(seed)
    rng.shuffle(cloned)
    return cloned


def print_progress(current: int, total: int, label: str = "Progress", width: int = 30) -> None:
    if total <= 0:
        return
    current = max(0, min(current, total))
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    message = f"\r{label}: [{bar}] {current}/{total}"
    if current >= total:
        message += "\n"
    sys.stdout.write(message)
    sys.stdout.flush()


def _simple_yaml_load(text: str) -> dict[str, Any]:
    lines = []
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.lstrip()))

    def parse_scalar(value: str) -> Any:
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        current_indent, current_content = lines[index]
        if current_indent < indent:
            return {}, index

        if current_content.startswith("- "):
            items = []
            while index < len(lines):
                current_indent, current_content = lines[index]
                if current_indent != indent or not current_content.startswith("- "):
                    break
                item_content = current_content[2:].strip()
                if item_content:
                    items.append(parse_scalar(item_content))
                    index += 1
                else:
                    nested, index = parse_block(index + 1, indent + 2)
                    items.append(nested)
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(lines):
            current_indent, current_content = lines[index]
            if current_indent != indent or current_content.startswith("- "):
                break
            key, _, remainder = current_content.partition(":")
            key = key.strip()
            remainder = remainder.strip()
            if remainder:
                mapping[key] = parse_scalar(remainder)
                index += 1
            else:
                nested, index = parse_block(index + 1, indent + 2)
                mapping[key] = nested
        return mapping, index

    parsed, _ = parse_block(0, 0)
    return parsed


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
