from __future__ import annotations

from medenvscale.schemas import RubricCriterion


def normalize_rubric_weights(rubrics: list[RubricCriterion]) -> list[RubricCriterion]:
    normalized: list[RubricCriterion] = []
    for rubric in rubrics:
        weight = rubric.weight
        if rubric.category == "critical_safety" and weight < 4:
            weight = 4
        normalized.append(rubric.model_copy(update={"weight": weight}))
    return normalized
