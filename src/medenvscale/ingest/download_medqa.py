from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from medenvscale.utils import ensure_dir, write_jsonl


def download_or_prepare_medqa(
    raw_path: str | Path,
    *,
    source_zip_url: str,
    source_zip_path: str | Path,
    extract_dir: str | Path,
    splits: list[str],
    metadata_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_zip_target = Path(source_zip_path)
    extract_target = Path(extract_dir)
    ensure_dir(source_zip_target.parent)
    ensure_dir(extract_target)

    _download_file(source_zip_url, source_zip_target)
    _extract_zip(source_zip_target, extract_target)

    total_seen = 0
    split_counts: dict[str, int] = {}
    split_path_map = {
        "train": extract_target / "data_clean" / "questions" / "US" / "train.jsonl",
        "valid": extract_target / "data_clean" / "questions" / "US" / "dev.jsonl",
        "test": extract_target / "data_clean" / "questions" / "US" / "test.jsonl",
    }
    for split_name in splits:
        if split_name not in split_path_map:
            raise ValueError(
                f"Unsupported split '{split_name}'. Supported splits: {list(split_path_map)}"
            )
        source_path = split_path_map[split_name]
        if not source_path.exists():
            raise FileNotFoundError(f"Expected English split file not found: {source_path}")
        split_counts[split_name] = 0
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload["_hf_split"] = split_name
                rows.append(payload)
                split_counts[split_name] += 1
                total_seen += 1
                if limit is not None and total_seen >= limit:
                    break
        if limit is not None and total_seen >= limit:
            break

    write_jsonl(raw_path, rows)
    metadata = {
        "dataset_name": "bigbio/med_qa",
        "language": "en",
        "download_method": "zip_direct",
        "source_zip_url": source_zip_url,
        "source_zip_path": str(source_zip_target),
        "extract_dir": str(extract_target),
        "splits_requested": splits,
        "splits_available": list(split_path_map),
        "rows_written": len(rows),
        "limit": limit,
        "raw_path": str(Path(raw_path)),
        "split_counts": split_counts,
    }
    metadata_target = Path(metadata_path)
    ensure_dir(metadata_target.parent)
    metadata_target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _download_file(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    expected = extract_dir / "data_clean" / "questions" / "US" / "train.jsonl"
    if expected.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
