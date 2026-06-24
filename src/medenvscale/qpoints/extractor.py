from __future__ import annotations

from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import ClinicalEnvironment, QuestionPoint


QUESTION_POINT_TYPES = {
    "critical_safety",
    "clinical_reasoning",
    "medical_knowledge",
    "evidence_use",
    "communication",
    "uncertainty_handling",
}

SECONDARY_TASK_BEHAVIORS = {
    "triage_urgent_management": ("critical_safety", "Address what features would make this scenario require urgent in-person evaluation."),
    "medication_safety": ("critical_safety", "Mention medication-specific cautions, interactions, or reasons not to self-adjust treatment."),
    "treatment_planning": ("communication", "State the most appropriate treatment or management plan in a clear, practical way."),
    "diagnostic_workup": ("evidence_use", "Explain the next test or diagnostic step that would most appropriately clarify the case."),
    "evidence_interpretation": ("evidence_use", "Interpret the key test or study findings that influence the recommendation."),
    "diagnosis_reasoning": ("clinical_reasoning", "Explain the most likely diagnosis or mechanism that fits the presentation."),
    "prevention_screening_counseling": ("communication", "Add concise prevention, screening, or counseling guidance relevant to the case."),
    "medical_knowledge_mechanism": ("medical_knowledge", "Include the core factual medical concept needed to understand the case."),
    "ethics_quality_safety": ("critical_safety", "Address the key patient-safety, communication, or professionalism issue in the scenario."),
    "biostatistics_calculation": ("evidence_use", "Explain the key calculation or study-design interpretation that supports the answer."),
}


def extract_question_points(environment: ClinicalEnvironment) -> list[QuestionPoint]:
    must_include = environment.gold_state.get("must_include", []) or [environment.clinical_topic]
    points: list[QuestionPoint] = []

    base = [
        ("critical_safety", "Identify the main safety concern and avoid false reassurance."),
        ("clinical_reasoning", "Connect the symptom pattern to the most likely explanation."),
        ("evidence_use", "Explain what supporting details matter and what is still missing."),
        ("communication", "Use clear patient-facing language and practical next steps."),
    ]

    if environment.primary_task_type == "medication_safety":
        base.append(("critical_safety", "State why medication decisions should not be made casually in this context."))
    else:
        base.append(("uncertainty_handling", "Acknowledge uncertainty and explain when medical review is needed."))

    for secondary_task_type in environment.secondary_task_types[:2]:
        behavior = SECONDARY_TASK_BEHAVIORS.get(secondary_task_type)
        if behavior is not None:
            base.append(behavior)

    for idx, (point_type, behavior) in enumerate(base, start=1):
        evidence = [must_include[min(idx - 1, len(must_include) - 1)]] if must_include else [environment.clinical_topic]
        points.append(
            QuestionPoint(
                env_id=environment.env_id,
                point_id=f"{environment.env_id}_QP{idx}",
                name=f"Question point {idx}",
                type=point_type,
                evidence=evidence,
                expected_model_behavior=behavior,
                must_have=True,
            )
        )
    return points


def _mock_qpoint_builder(context: dict[str, Any]) -> dict[str, Any]:
    environment: ClinicalEnvironment = context["environment"]
    points = extract_question_points(environment)
    return {
        "question_points": [
            {
                "name": point.name,
                "type": point.type,
                "evidence": point.evidence,
                "expected_model_behavior": point.expected_model_behavior,
                "must_have": point.must_have,
            }
            for point in points
        ]
    }


def _coerce_question_points_payload(payload: dict[str, Any], environment: ClinicalEnvironment) -> list[QuestionPoint]:
    raw_points = payload.get("question_points", payload if isinstance(payload, list) else [])
    if not isinstance(raw_points, list):
        raw_points = []

    default_points = extract_question_points(environment)
    coerced: list[QuestionPoint] = []
    for idx, raw_point in enumerate(raw_points, start=1):
        if not isinstance(raw_point, dict):
            continue
        point_type = str(raw_point.get("type") or "").strip()
        if point_type not in QUESTION_POINT_TYPES:
            point_type = default_points[min(idx - 1, len(default_points) - 1)].type if default_points else "clinical_reasoning"
        evidence = raw_point.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        if not evidence:
            fallback = default_points[min(idx - 1, len(default_points) - 1)].evidence if default_points else [environment.clinical_topic]
            evidence = fallback
        behavior = str(raw_point.get("expected_model_behavior") or "").strip()
        if not behavior:
            behavior = default_points[min(idx - 1, len(default_points) - 1)].expected_model_behavior if default_points else "Provide a clinically grounded answer."
        name = str(raw_point.get("name") or f"Question point {idx}").strip()
        coerced.append(
            QuestionPoint(
                env_id=environment.env_id,
                point_id=f"{environment.env_id}_QP{idx}",
                name=name,
                type=point_type,
                evidence=evidence,
                expected_model_behavior=behavior,
                must_have=bool(raw_point.get("must_have", True)),
            )
        )

    if not coerced:
        return default_points
    return coerced


def extract_question_points_with_llm(
    environment: ClinicalEnvironment,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
) -> list[QuestionPoint]:
    prompt = prompt_runner.render(
        "qpoint_extract.jinja",
        env_id=environment.env_id,
        primary_domain=environment.primary_domain,
        secondary_domains=environment.secondary_domains,
        primary_task_type=environment.primary_task_type,
        secondary_task_types=environment.secondary_task_types,
        clinical_topic=environment.clinical_topic,
        level=environment.level,
        user_prompt=environment.user_prompt,
        patient_state=environment.patient_state,
        clinical_context=environment.clinical_context,
        evidence_state=environment.evidence_state,
        gold_state=environment.gold_state,
        difficulty=environment.difficulty.model_dump(),
        operators_applied=environment.operators_applied,
    )
    response = llm_client.complete_json(
        task_name="qpoint_extract",
        prompt=prompt,
        context={
            "environment": environment,
        },
        mock_builder=_mock_qpoint_builder,
    )
    return _coerce_question_points_payload(response.payload, environment)
