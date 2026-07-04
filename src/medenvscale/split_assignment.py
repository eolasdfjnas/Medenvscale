from __future__ import annotations

from collections import defaultdict
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.utils import seeded_shuffle


SPLIT_NAMES = ("train", "dev", "test")


def assign_dataset_splits(
    environments: list[ExecutableEnvSpec],
    split_cfg: dict[str, Any],
) -> tuple[list[ExecutableEnvSpec], dict[str, Any]]:
    grouped = _group_by_original_task(environments)
    split_keys, normalized_ratios = split_group_keys(grouped.keys(), split_cfg)
    assigned: list[ExecutableEnvSpec] = []
    counts = {name: 0 for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for key in split_keys[split_name]:
            for env in grouped[key]:
                previous_split = env.split
                env.split = split_name
                metadata = dict(env.metadata or {})
                metadata.update(
                    {
                        "dataset_split": split_name,
                        "split_stage": "05_5",
                        "split_group_key": key,
                        "source_split_before_stage05_5": metadata.get("source_split_before_stage05_5", previous_split),
                    }
                )
                env.metadata = metadata
                assigned.append(env)
                counts[split_name] += 1
    manifest = {
        "split_stage": "05_5",
        "split_policy": "group_by_original_task_id",
        "split_seed": int(split_cfg.get("seed", 1337)),
        "raw_ratios": _raw_ratios(split_cfg),
        "normalized_ratios": normalized_ratios,
        "num_groups": len(grouped),
        "num_envs": len(environments),
        "env_counts": counts,
        "group_counts": {name: len(split_keys[name]) for name in SPLIT_NAMES},
    }
    return assigned, manifest


def split_group_keys(keys: Any, split_cfg: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, float]]:
    shuffled = seeded_shuffle([str(key) for key in keys], int(split_cfg.get("seed", 1337)))
    n = len(shuffled)
    ratios = _normalized_ratios(split_cfg)
    train_count = int(n * ratios["train"])
    dev_count = int(n * ratios["dev"])
    if n >= 3 and dev_count == 0 and ratios["dev"] > 0:
        dev_count = 1
    test_count = n - train_count - dev_count
    if n >= 3 and test_count == 0 and ratios["test"] > 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        elif dev_count > 1:
            dev_count -= 1
    if n and train_count == 0 and ratios["train"] > 0:
        train_count = 1
        if test_count > 0:
            test_count -= 1
        elif dev_count > 0:
            dev_count -= 1
    train_end = min(max(train_count, 0), n)
    dev_end = min(train_end + max(dev_count, 0), n)
    return (
        {
            "train": shuffled[:train_end],
            "dev": shuffled[train_end:dev_end],
            "test": shuffled[dev_end:],
        },
        ratios,
    )


def split_envs_by_assigned_split(environments: list[ExecutableEnvSpec]) -> dict[str, list[ExecutableEnvSpec]]:
    result = {name: [] for name in SPLIT_NAMES}
    for env in environments:
        split_name = str((env.metadata or {}).get("dataset_split") or env.split or "train").lower()
        if split_name not in result:
            split_name = "train"
        result[split_name].append(env)
    return result


def has_stage05_5_split(environments: list[ExecutableEnvSpec]) -> bool:
    return any((env.metadata or {}).get("split_stage") == "05_5" for env in environments)


def _group_by_original_task(environments: list[ExecutableEnvSpec]) -> dict[str, list[ExecutableEnvSpec]]:
    grouped: dict[str, list[ExecutableEnvSpec]] = defaultdict(list)
    for env in environments:
        grouped[str(env.original_task_id or env.env_id)].append(env)
    return grouped


def _normalized_ratios(split_cfg: dict[str, Any]) -> dict[str, float]:
    raw = _raw_ratios(split_cfg)
    total = sum(raw.values())
    if total <= 0:
        raw = {"train": 0.7, "dev": 0.1, "test": 0.2}
        total = 1.0
    return {name: raw[name] / total for name in SPLIT_NAMES}


def _raw_ratios(split_cfg: dict[str, Any]) -> dict[str, float]:
    ratios = split_cfg.get("split_ratios") if isinstance(split_cfg.get("split_ratios"), dict) else split_cfg
    return {
        "train": max(0.0, float(ratios.get("train", 0.7))),
        "dev": max(0.0, float(ratios.get("dev", 0.1))),
        "test": max(0.0, float(ratios.get("test", 0.2))),
    }
