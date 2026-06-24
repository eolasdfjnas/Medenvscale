from __future__ import annotations

from medenvscale.schemas import ExecutableEnvSpec, RLVREnv


def export_rlvr_stub(environment: ExecutableEnvSpec) -> RLVREnv:
    tool_config = environment.tool_config or {}
    action_space = [tool["tool_name"] for tool in tool_config.get("allowed_tools", [])] + ["submit_answer"]
    return RLVREnv(
        env_id=environment.env_id,
        initial_observation=environment.user_prompt or environment.problem,
        action_space=action_space,
        tool_config=tool_config,
        verifier=environment.verifier_spec or {},
        reward_fn={
            "verifier_score": True,
            "rubric_score": True,
            "tool_budget_penalty": True,
        },
        max_steps=int((tool_config.get("tool_budget") or {}).get("max_total_tool_calls", 0)) + 1,
        difficulty=(environment.difficulty.model_dump() if environment.difficulty else {}),
        secondary_domains=[item.model_dump() for item in environment.secondary_domains],
        secondary_task_types=environment.secondary_task_types,
    )
