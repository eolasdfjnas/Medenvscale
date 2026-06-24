from __future__ import annotations

from medenvscale.schemas import ExecutableEnvSpec, PRMSample, QuestionPoint


def export_prm_samples(environment: ExecutableEnvSpec, question_points: list[QuestionPoint]) -> list[PRMSample]:
    samples: list[PRMSample] = []
    for step_id, point in enumerate(question_points, start=1):
        samples.append(
            PRMSample(
                env_id=environment.env_id,
                trajectory_id=f"traj_{environment.env_id}",
                step_id=step_id,
                state=point.description,
                action={"kind": "reason_about_requirement", "point_id": point.point_id},
                label="correct",
                rubric_hits=[point.point_id],
                score=1.0,
                related_axes=point.related_axes,
                related_operator_ids=[row["operator_id"] for row in environment.operator_instances],
            )
        )
    return samples
