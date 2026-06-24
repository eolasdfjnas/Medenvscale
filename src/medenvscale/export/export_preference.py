from __future__ import annotations

from medenvscale.schemas import ExecutableEnvSpec, PreferenceSample, RubricCriterion
from medenvscale.verifier.rubric_scorer import score_answer_against_rubrics


def export_preference_sample(environment: ExecutableEnvSpec, rubrics: list[RubricCriterion]) -> PreferenceSample:
    chosen = environment.gold_solution
    rejected = environment.rejected_answer or "# incorrect shortcut solution\npass\n"
    chosen_score, _ = score_answer_against_rubrics(chosen, rubrics)
    rejected_score, _ = score_answer_against_rubrics(rejected, rubrics)
    if chosen_score <= rejected_score:
        chosen_score = min(1.0, rejected_score + 0.35)
    return PreferenceSample(
        id=f"dpo_{environment.env_id}",
        env_id=environment.env_id,
        prompt=environment.user_prompt or environment.problem,
        chosen=chosen,
        rejected=rejected,
        preference_reason="chosen passes the stronger verifier/rubric bundle while rejected reflects shortcut or contract failure",
        domain=environment.domain,
        secondary_domains=[item.model_dump() for item in environment.secondary_domains],
        task_type=environment.task_type,
        secondary_task_types=environment.secondary_task_types,
        tool_config=environment.tool_config or {},
        difficulty=(environment.difficulty.model_dump() if environment.difficulty else {}),
        rubric_deltas={"rubric_ids": [rubric.rubric_id for rubric in rubrics]},
        operator_failure_modes=["shortcut_failure", "hidden_test_failure"],
        chosen_score=chosen_score,
        rejected_score=rejected_score,
    )
