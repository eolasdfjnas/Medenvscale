from __future__ import annotations

from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import ClinicalEnvironment, QuestionPoint, RubricCriterion


ALLOWED_SCORE_TYPES = {"binary", "scalar"}
DEFAULT_WEIGHT_BY_CATEGORY = {
    "critical_safety": 5,
    "clinical_reasoning": 3,
    "medical_knowledge": 3,
    "evidence_use": 3,
    "communication": 3,
    "uncertainty_handling": 2,
}


def generate_rubrics(env_id: str, points: list[QuestionPoint]) -> list[RubricCriterion]:
    rubrics: list[RubricCriterion] = []
    for idx, point in enumerate(points, start=1):
        weight = 5 if point.type == "critical_safety" else 3
        rubrics.append(
            RubricCriterion(
                env_id=env_id,
                rubric_id=f"{env_id}_R{idx}",
                source_point_id=point.point_id,
                criterion=point.expected_model_behavior,
                score_type="binary",
                weight=weight,
                category=point.type,
                fail_cap=0.4 if point.type == "critical_safety" else None,
            )
        )
    return rubrics


def _mock_rubric_builder(context: dict[str, Any]) -> dict[str, Any]:
    environment: ClinicalEnvironment = context["environment"]
    points: list[QuestionPoint] = context["question_points"]
    rubrics = generate_rubrics(environment.env_id, points)
    return {
        "rubrics": [
            {
                "source_point_id": rubric.source_point_id,
                "criterion": rubric.criterion,
                "score_type": rubric.score_type,
                "weight": rubric.weight,
                "category": rubric.category,
                "fail_cap": rubric.fail_cap,
            }
            for rubric in rubrics
        ]
    }


def _coerce_rubric_payload(payload: dict[str, Any], environment: ClinicalEnvironment, points: list[QuestionPoint]) -> list[RubricCriterion]:
    raw_rubrics = payload.get("rubrics", payload if isinstance(payload, list) else [])
    if not isinstance(raw_rubrics, list):
        raw_rubrics = []

    point_ids = {point.point_id for point in points}
    point_categories = {point.point_id: point.type for point in points}
    fallback = generate_rubrics(environment.env_id, points)
    coerced: list[RubricCriterion] = []
    for idx, raw_rubric in enumerate(raw_rubrics, start=1):
        if not isinstance(raw_rubric, dict):
            continue
        source_point_id = str(raw_rubric.get("source_point_id") or "").strip()
        if source_point_id not in point_ids:
            source_point_id = points[min(idx - 1, len(points) - 1)].point_id if points else f"{environment.env_id}_QP1"
        category = str(raw_rubric.get("category") or point_categories.get(source_point_id, "clinical_reasoning")).strip()
        if category not in DEFAULT_WEIGHT_BY_CATEGORY:
            category = point_categories.get(source_point_id, "clinical_reasoning")
        criterion = str(raw_rubric.get("criterion") or "").strip()
        if not criterion:
            criterion = fallback[min(idx - 1, len(fallback) - 1)].criterion if fallback else "Provide a clinically grounded answer."
        score_type = str(raw_rubric.get("score_type") or "binary").strip()
        if score_type not in ALLOWED_SCORE_TYPES:
            score_type = "binary"
        try:
            weight = int(raw_rubric.get("weight", DEFAULT_WEIGHT_BY_CATEGORY[category]))
        except (TypeError, ValueError):
            weight = DEFAULT_WEIGHT_BY_CATEGORY[category]
        if weight < 1:
            weight = DEFAULT_WEIGHT_BY_CATEGORY[category]
        fail_cap = raw_rubric.get("fail_cap")
        if category == "critical_safety":
            try:
                fail_cap = float(fail_cap if fail_cap is not None else 0.4)
            except (TypeError, ValueError):
                fail_cap = 0.4
        else:
            fail_cap = None
        coerced.append(
            RubricCriterion(
                env_id=environment.env_id,
                rubric_id=f"{environment.env_id}_R{idx}",
                source_point_id=source_point_id,
                criterion=criterion,
                score_type=score_type,
                weight=weight,
                category=category,
                fail_cap=fail_cap,
            )
        )

    if not coerced:
        return fallback
    return coerced


def generate_rubrics_with_llm(
    environment: ClinicalEnvironment,
    points: list[QuestionPoint],
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
) -> list[RubricCriterion]:
    prompt = prompt_runner.render(
        "rubric_generate.jinja",
        env_id=environment.env_id,
        primary_domain=environment.primary_domain,
        secondary_domains=environment.secondary_domains,
        primary_task_type=environment.primary_task_type,
        secondary_task_types=environment.secondary_task_types,
        clinical_topic=environment.clinical_topic,
        level=environment.level,
        user_prompt=environment.user_prompt,
        patient_state=environment.patient_state,
        evidence_state=environment.evidence_state,
        gold_state=environment.gold_state,
        difficulty=environment.difficulty.model_dump(),
        operators_applied=environment.operators_applied,
        question_points=[point.model_dump() for point in points],
    )
    response = llm_client.complete_json(
        task_name="rubric_generate",
        prompt=prompt,
        context={
            "environment": environment,
            "question_points": points,
        },
        mock_builder=_mock_rubric_builder,
    )
    return _coerce_rubric_payload(response.payload, environment, points)
