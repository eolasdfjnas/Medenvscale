from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec


def build_requirement_rubrics(env: ExecutableEnvSpec) -> list[dict[str, Any]]:
    metadata = [row for row in (env.output_requirement_metadata or []) if isinstance(row, dict)]
    requirements = _dedupe_strings([str(item).strip() for item in (env.output_requirements or []) if str(item).strip()])
    if not requirements and not metadata:
        return []
    cases = [case for case in (env.validated_oracle_cases or env.scaled_oracle_cases or []) if isinstance(case, dict)]
    rubrics = []
    if metadata:
        requirement_rows = metadata
    else:
        requirement_rows = [
            {
                "requirement_id": f"req_{index:03d}",
                "text": requirement,
                "source": "output_requirements",
                "category": _requirement_category(requirement),
                "operator_id": None,
                "axis": "global",
            }
            for index, requirement in enumerate(requirements, start=1)
        ]
    for index, row in enumerate(requirement_rows, start=1):
        requirement_id = str(row.get("requirement_id") or f"req_{index:03d}")
        requirement = str(row.get("text") or "").strip()
        if not requirement:
            continue
        covered_case_ids = _covered_case_ids(requirement, cases, requirement_id=requirement_id)
        category = str(row.get("category") or _requirement_category(requirement))
        rubrics.append(
            {
                "rubric_id": f"{env.env_id}_{requirement_id}",
                "env_id": env.env_id,
                "requirement_id": requirement_id,
                "operator_id": row.get("operator_id"),
                "axis": row.get("axis"),
                "source": str(row.get("source") or "output_requirements"),
                "requirement": requirement,
                "criterion": requirement,
                "category": category,
                "weight": _category_weight(category),
                "evidence_type": "oracle_case" if covered_case_ids else "uncovered",
                "covered_by_cases": covered_case_ids,
                "score_type": "binary_all_cases_pass",
                "max_score": 1.0,
            }
        )
    return rubrics


def score_requirement_rubrics(
    rubrics: list[dict[str, Any]] | None,
    case_reports: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    case_passed = {str(row.get("case_id") or ""): bool(row.get("passed")) for row in (case_reports or []) if isinstance(row, dict)}
    scores = []
    weighted_sum = 0.0
    weight_sum = 0.0
    by_category: dict[str, dict[str, float]] = defaultdict(lambda: {"weighted_sum": 0.0, "weight_sum": 0.0, "scored": 0.0, "satisfied": 0.0})
    for rubric in rubrics or []:
        if not isinstance(rubric, dict):
            continue
        covered = [str(item) for item in (rubric.get("covered_by_cases") or []) if str(item)]
        weight = float(rubric.get("weight") or 1.0)
        category = str(rubric.get("category") or "other")
        score: float | None
        passed_covered = 0
        if covered:
            passed_covered = sum(1 for case_id in covered if case_passed.get(case_id, False))
            score = passed_covered / len(covered)
        else:
            score = None
        row = {
            "rubric_id": str(rubric.get("rubric_id") or ""),
            "requirement_id": str(rubric.get("requirement_id") or ""),
            "requirement": str(rubric.get("requirement") or rubric.get("criterion") or ""),
            "category": category,
            "weight": weight,
            "covered_by_cases": covered,
            "passed_covered_cases": passed_covered if score is not None else None,
            "total_covered_cases": len(covered) if score is not None else None,
            "score": round(score, 6) if score is not None else None,
            "satisfied": bool(score == 1.0) if score is not None else None,
            "evidence_type": str(rubric.get("evidence_type") or ("oracle_case" if covered else "uncovered")),
        }
        scores.append(row)
        if score is None:
            continue
        weighted_sum += score * weight
        weight_sum += weight
        by_category[category]["weighted_sum"] += score * weight
        by_category[category]["weight_sum"] += weight
        by_category[category]["scored"] += 1.0
        by_category[category]["satisfied"] += 1.0 if score == 1.0 else 0.0
    rubric_score = weighted_sum / weight_sum if weight_sum else 0.0
    category_scores = {
        category: {
            "rubric_score": round(values["weighted_sum"] / values["weight_sum"], 6) if values["weight_sum"] else 0.0,
            "scored_rubrics": int(values["scored"]),
            "satisfied_rubrics": int(values["satisfied"]),
        }
        for category, values in sorted(by_category.items())
    }
    return {
        "rubric_score": round(rubric_score, 6),
        "rubric_scores": scores,
        "rubric_by_category": category_scores,
        "scored_rubrics": sum(1 for row in scores if row["score"] is not None),
        "total_rubrics": len(scores),
    }


def _covered_case_ids(requirement: str, cases: list[dict[str, Any]], *, requirement_id: str | None = None) -> list[str]:
    matched = []
    for case in cases:
        if requirement_id and requirement_id in {str(item).strip() for item in (case.get("covered_requirement_ids") or [])}:
            case_id = str(case.get("case_id") or "").strip()
            if case_id:
                matched.append(case_id)
            continue
        covered_requirements = [
            str(item).strip()
            for item in ((case.get("covered_requirements") or case.get("covers_requirements")) or [])
            if str(item).strip()
        ]
        if any(_requirements_match(requirement, item) for item in covered_requirements):
            case_id = str(case.get("case_id") or "").strip()
            if case_id:
                matched.append(case_id)
    return _dedupe_strings(matched)


def _requirements_match(left: str, right: str) -> bool:
    left_norm = _normalize_requirement(left)
    right_norm = _normalize_requirement(right)
    if not left_norm or not right_norm:
        return False
    return left_norm in right_norm or right_norm in left_norm


def _requirement_category(requirement: str) -> str:
    text = requirement.lower()
    if "original seed" in text or "seed behavior" in text or "seed call" in text:
        return "seed_behavior"
    if "stdout" in text or "stderr" in text or "print" in text or "format" in text:
        return "output_format"
    if "file artifact" in text or "file path" in text or "create or update file" in text:
        return "output_format"
    if "return type" in text or "signature" in text or "compile" in text:
        return "runtime_contract"
    return "scaled_requirement"


def _category_weight(category: str) -> float:
    return {
        "seed_behavior": 2.0,
        "scaled_requirement": 2.0,
        "robustness": 2.0,
        "output_format": 1.5,
        "runtime_contract": 1.0,
        "tool_process": 0.3,
    }.get(category, 1.0)


def _normalize_requirement(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9_ .:/+-]+", "", normalized)
    return normalized.strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
