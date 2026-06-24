from __future__ import annotations

from medenvscale.schemas import QuestionPoint


def calibrate_question_points(points: list[QuestionPoint], minimum: int = 4) -> list[QuestionPoint]:
    if len(points) >= minimum:
        return points
    calibrated = list(points)
    while len(calibrated) < minimum:
        idx = len(calibrated) + 1
        calibrated.append(
            QuestionPoint(
                env_id=points[0].env_id if points else "unknown",
                point_id=f"{points[0].env_id if points else 'unknown'}_QP{idx}",
                name=f"Question point {idx}",
                type="communication",
                evidence=["Provide a clear next step."],
                expected_model_behavior="Provide a concrete next step and explain why it matters.",
                must_have=True,
            )
        )
    return calibrated
