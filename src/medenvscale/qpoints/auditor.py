from __future__ import annotations

from medenvscale.schemas import QuestionPoint


def audit_question_points(points: list[QuestionPoint]) -> list[str]:
    issues: list[str] = []
    if len(points) < 4:
        issues.append("too_few_question_points")
    for point in points:
        if not point.evidence:
            issues.append(f"{point.point_id}:missing_evidence")
        if "be helpful" in point.expected_model_behavior.lower():
            issues.append(f"{point.point_id}:generic_behavior")
    return issues
