from __future__ import annotations

from medenvscale.schemas import RubricCriterion


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "is",
    "what",
    "when",
    "use",
    "clear",
    "patient",
    "facing",
}


def score_answer_against_rubrics(answer: str, rubrics: list[RubricCriterion]) -> tuple[float, dict[str, str]]:
    answer_lower = answer.lower()
    total = sum(r.weight for r in rubrics) or 1
    earned = 0
    comparison: dict[str, str] = {}
    for rubric in rubrics:
        tokens = [token.strip(".,:;!?").lower() for token in rubric.criterion.split()]
        tokens = [token for token in tokens if len(token) > 3 and token not in STOPWORDS]
        matched_count = sum(1 for token in tokens if token in answer_lower)
        threshold = max(1, min(3, len(tokens) // 2))
        matched = matched_count >= threshold
        if matched:
            earned += rubric.weight
            comparison[rubric.rubric_id] = "pass"
        else:
            comparison[rubric.rubric_id] = "fail"
    return earned / total, comparison
