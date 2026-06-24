from __future__ import annotations

from medenvscale.schemas import ChatMessage, ExecutableEnvSpec, RubricCriterion, SFTSample


def export_sft_sample(environment: ExecutableEnvSpec, rubrics: list[RubricCriterion], system_prompt: str | None = None) -> SFTSample:
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    if environment.user_prompt:
        messages.append(ChatMessage(role="user", content=environment.user_prompt))
    messages.append(ChatMessage(role="assistant", content=environment.gold_solution))
    return SFTSample(
        id=f"sft_{environment.env_id}",
        env_id=environment.env_id,
        messages=messages,
        domain=environment.domain,
        secondary_domains=[item.model_dump() for item in environment.secondary_domains],
        task_type=environment.task_type,
        secondary_task_types=environment.secondary_task_types,
        solution_form=environment.solution_form,
        tool_config=environment.tool_config or {},
        difficulty=(environment.difficulty.model_dump() if environment.difficulty else {}),
        operator_mode=environment.gold_state.get("operator_mode", "gold_compatible"),
        rubrics=[rubric.rubric_id for rubric in rubrics],
        verifier_id=(environment.verifier_spec or {}).get("verifier_id", f"verifier_{environment.env_id}"),
    )
