from __future__ import annotations

from medenvscale.schemas import RubricCriterion


def deduplicate_rubrics(rubrics: list[RubricCriterion]) -> list[RubricCriterion]:
    seen: set[str] = set()
    deduped: list[RubricCriterion] = []
    for rubric in rubrics:
        key = rubric.criterion.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(rubric)
    return deduped
