from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_dir, load_yaml, slugify


@dataclass
class AppConfig:
    root: Path
    values: dict[str, Any]
    llm_values: dict[str, Any]
    config_path: Path | None = None
    dataset_name: str | None = None

    @property
    def output_dirs(self) -> dict[str, Path]:
        output = self.values["output"]
        dataset_cfg = self.values.get("dataset", {})
        experiment_slug = self.dataset_name or dataset_cfg.get("dataset_slug") or slugify(str(dataset_cfg.get("name", "dataset")), max_length=48)
        default_experiment_dir = output.get(
            "experiment_dir",
            str(Path("experiments") / str(experiment_slug)),
        )
        return {
            "raw": self.root / output["raw_dir"],
            "interim": self.root / output["interim_dir"],
            "processed": self.root / output["processed_dir"],
            "splits": self.root / output["split_dir"],
            "result": self.root / output.get("result_dir", "result"),
            "experiments": self.root / default_experiment_dir,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def dataset_config_path(self, filename: str) -> Path:
        if self.dataset_name:
            scoped = self.root / "configs" / self.dataset_name / filename
            if scoped.exists():
                return scoped
        return self.root / "configs" / filename

    def dataset_config_path_with_fallback(self, filename: str, fallback_filename: str) -> Path:
        preferred = self.dataset_config_path(filename)
        if preferred.exists():
            return preferred
        return self.dataset_config_path(fallback_filename)


def _resolve_project_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "configs" / "llm.yaml").exists() and (candidate / "src" / "medenvscale").exists():
            return candidate
    return config_path.parent.parent


def _dataset_scope(values: dict[str, Any], dataset: str | None = None) -> str | None:
    if dataset:
        return slugify(dataset, max_length=48)
    dataset_cfg = values.get("dataset", {})
    default_dataset = dataset_cfg.get("default_dataset")
    if default_dataset:
        return slugify(str(default_dataset), max_length=48)
    config_dataset = values.get("default_dataset")
    if config_dataset:
        return slugify(str(config_dataset), max_length=48)
    return None


def _rewrite_dataset_file_layout(values: dict[str, Any], llm_values: dict[str, Any], dataset_name: str | None) -> None:
    if not dataset_name:
        return

    dataset_cfg = values.setdefault("dataset", {})
    dataset_cfg["active_dataset"] = dataset_name
    output = values.setdefault("output", {})
    raw_dir = Path("data") / dataset_name / "raw"
    interim_dir = Path("data") / dataset_name / "interim"
    processed_dir = Path("data") / dataset_name / "processed"
    split_dir = Path("data") / dataset_name / "splits"
    result_dir = Path("result") / dataset_name
    experiment_dir = Path("experiments") / dataset_name

    output["raw_dir"] = str(raw_dir)
    output["interim_dir"] = str(interim_dir)
    output["processed_dir"] = str(processed_dir)
    output["split_dir"] = str(split_dir)
    output["result_dir"] = str(result_dir)
    output["experiment_dir"] = str(experiment_dir)

    raw_path_keys = {
        "local_raw_path": raw_dir,
        "metadata_path": raw_dir,
    }
    for key, target_dir in raw_path_keys.items():
        current = dataset_cfg.get(key)
        if current:
            dataset_cfg[key] = str(target_dir / Path(str(current)).name)
    raw_paths = dataset_cfg.get("local_raw_paths")
    if isinstance(raw_paths, dict):
        dataset_cfg["local_raw_paths"] = {
            split_name: str(raw_dir / Path(str(path)).name)
            for split_name, path in raw_paths.items()
        }

    source_dir = raw_dir / "source"
    if dataset_cfg.get("source_zip_path"):
        dataset_cfg["source_zip_path"] = str(source_dir / Path(str(dataset_cfg["source_zip_path"])).name)
    if dataset_cfg.get("extract_dir"):
        dataset_cfg["extract_dir"] = str(source_dir / "extracted")

    dataset_root = dataset_cfg.get("dataset_root")
    task_files = dataset_cfg.get("task_files")
    if dataset_root and isinstance(task_files, dict):
        scoped_root = Path(str(dataset_root)) / dataset_name
        dataset_cfg["task_files"] = {
            split_name: str(scoped_root / str(relative_path))
            if not Path(str(relative_path)).is_absolute()
            else str(relative_path)
            for split_name, relative_path in task_files.items()
        }

    llm_values.setdefault("cache", {})
    llm_values["cache"]["dir"] = str(Path(".cache") / "llm" / dataset_name)
    llm_values.setdefault("trace", {})
    llm_values["trace"]["path"] = str(processed_dir / Path(str(llm_values["trace"].get("path", "generation_trace.jsonl"))).name)


def _rewrite_training_file_layout(values: dict[str, Any], dataset_name: str | None) -> None:
    if not dataset_name:
        return

    split_dir = Path("data") / dataset_name / "splits"
    for key in ("dataset_path", "eval_path"):
        current = values.get(key)
        if current:
            values[key] = str(split_dir / Path(str(current)).name)

    output_dir = values.get("output_dir")
    if output_dir:
        values["output_dir"] = str(Path("experiments") / dataset_name / Path(str(output_dir)).name)

    values["active_dataset"] = dataset_name


def load_app_config(config_path: str | Path, dataset: str | None = None) -> AppConfig:
    config_path = Path(config_path).resolve()
    root = _resolve_project_root(config_path)
    values = deepcopy(load_yaml(config_path))
    llm_values = deepcopy(load_yaml(root / "configs" / "llm.yaml"))
    dataset_name = _dataset_scope(values, dataset=dataset)
    _rewrite_dataset_file_layout(values, llm_values, dataset_name)
    cfg = AppConfig(root=root, values=values, llm_values=llm_values, config_path=config_path, dataset_name=dataset_name)
    for path in cfg.output_dirs.values():
        ensure_dir(path)
    ensure_dir(root / cfg.llm_values["cache"]["dir"])
    return cfg


def load_training_config(config_path: str | Path, dataset: str | None = None) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path).resolve()
    root = _resolve_project_root(config_path)
    values = deepcopy(load_yaml(config_path))
    dataset_name = _dataset_scope({"default_dataset": values.get("default_dataset")}, dataset=dataset)
    _rewrite_training_file_layout(values, dataset_name)
    return root, values


def resolve_dataset_config_path(root: Path, filename: str, dataset: str | None = None) -> Path:
    if dataset:
        scoped = root / "configs" / dataset / filename
        if scoped.exists():
            return scoped
    return root / "configs" / filename


def resolve_dataset_config_path_with_fallback(
    root: Path,
    filename: str,
    fallback_filename: str,
    dataset: str | None = None,
) -> Path:
    preferred = resolve_dataset_config_path(root, filename, dataset=dataset)
    if preferred.exists():
        return preferred
    return resolve_dataset_config_path(root, fallback_filename, dataset=dataset)
