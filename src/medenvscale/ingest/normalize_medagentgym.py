from __future__ import annotations

from typing import Any

from medenvscale.schemas import MedAgentGymTask
from medenvscale.utils import slugify

from .load_medagentgym import collect_resource_files
from .placeholder_analyzer import PLACEHOLDER_TOKEN, summarize_context


def _first_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_raw_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return None


def _list_of_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def normalize_row(row: dict[str, Any], source_split: str, index: int) -> MedAgentGymTask:
    raw_id = _first_text(row, ["task_id", "id", "instance_id", "question_id", "idx"]) or f"{source_split}_{index:06d}"
    task_id = raw_id if raw_id.startswith("medagentgym_") else f"medagentgym_{source_split}_{slugify(raw_id, max_length=64)}"

    problem = _first_text(row, ["problem", "instruction", "question", "prompt", "description"]) or "Complete the biomedical coding task."
    solution = _first_text(row, ["solution", "ground_truth", "answer", "expected_answer"]) or ""
    context = _first_raw_text(row, ["context", "starter_code", "template_code", "code"]) or PLACEHOLDER_TOKEN
    signature = _first_text(row, ["signature", "function_signature"])
    code = _first_raw_text(row, ["code", "full_code", "reference_code"])
    wrong_code = _first_raw_text(row, ["wrong_code"])
    task_family = _first_text(row, ["task_family", "category", "task_type", "domain", "dataset"])
    category = _first_text(row, ["category", "subcategory", "domain", "topic"])
    category_path = _list_of_texts(row.get("category_path") or row.get("categories") or row.get("tags"))
    ground_truth_output_signature = row.get("ground_truth_output_signature") if isinstance(row.get("ground_truth_output_signature"), dict) else {}
    code_execution_result = row.get("code_execution_result") if isinstance(row.get("code_execution_result"), dict) else {}
    seed_execution_case = row.get("seed_execution_case") if isinstance(row.get("seed_execution_case"), dict) else {}
    seed_case_audit = row.get("seed_case_audit") if isinstance(row.get("seed_case_audit"), dict) else {}
    execution_status = _first_text(row, ["execution_status"])
    repair_attempts = row.get("repair_attempts")
    repair_succeeded = bool(row.get("repair_succeeded", False))

    raw_metadata = dict(row)
    for key in [
        "task_id",
        "id",
        "instance_id",
        "question_id",
        "idx",
        "problem",
        "instruction",
        "question",
        "prompt",
        "description",
        "solution",
        "ground_truth",
        "answer",
        "expected_answer",
        "context",
        "starter_code",
        "template_code",
        "signature",
        "function_signature",
        "code",
        "full_code",
        "reference_code",
        "wrong_code",
        "task_family",
        "category",
        "task_type",
        "domain",
        "dataset",
        "subcategory",
        "topic",
        "category_path",
        "categories",
        "tags",
        "resource_files",
        "resources",
        "resource_paths",
        "files",
        "artifacts",
        "source_split",
        "ground_truth_output_signature",
        "code_execution_result",
        "seed_execution_case",
        "seed_case_audit",
        "execution_status",
        "repair_attempts",
        "repair_succeeded",
        "repair_history",
        "ground_truth",
    ]:
        raw_metadata.pop(key, None)

    return MedAgentGymTask(
        task_id=task_id,
        source_split=str(row.get("source_split") or source_split),
        idx=_first_text(row, ["idx", "task_index"]),
        problem=problem,
        solution=solution,
        context=context,
        signature=signature,
        code=code,
        wrong_code=wrong_code,
        task_family=task_family,
        category=category,
        category_path=category_path,
        resource_files=collect_resource_files(row),
        has_placeholder=PLACEHOLDER_TOKEN in context,
        context_summary=summarize_context(context),
        ground_truth_output_signature=ground_truth_output_signature,
        code_execution_result=code_execution_result,
        seed_execution_case=seed_execution_case,
        seed_case_audit=seed_case_audit,
        execution_status=execution_status,
        repair_attempts=int(repair_attempts or 0),
        repair_succeeded=repair_succeeded,
        raw_metadata=raw_metadata,
    )


def normalize_rows(rows: list[dict[str, Any]], source_split: str = "train") -> list[MedAgentGymTask]:
    return [normalize_row(row, str(row.get("source_split") or source_split), index) for index, row in enumerate(rows, start=1)]
