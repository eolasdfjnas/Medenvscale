from __future__ import annotations

from copy import deepcopy

from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.schemas.scaling import DynamicOperatorInstance


def apply_operator_instances(base_environment: ExecutableEnvSpec, operator_instances: list[DynamicOperatorInstance]) -> ExecutableEnvSpec:
    env = base_environment.model_copy(deep=True)
    task_state = deepcopy(env.task_state)
    data_state = deepcopy(env.data_state)
    tool_state = deepcopy(env.tool_state)
    visible_state = deepcopy(env.visible_state)
    gold_state = deepcopy(env.gold_state)
    verifier_state = deepcopy(env.verifier_state)
    test_state = deepcopy(env.test_state)
    turn_state = deepcopy(env.turn_state)

    for op in operator_instances:
        _merge_patch(task_state, op.state_updates.task_state_patch)
        _merge_patch(data_state, op.state_updates.data_state_patch)
        _merge_patch(tool_state, op.state_updates.tool_state_patch)
        _merge_patch(visible_state, op.state_updates.visible_state_patch)
        _merge_patch(gold_state, op.state_updates.gold_state_patch)
        _merge_patch(verifier_state, op.state_updates.verifier_state_patch)
        _merge_patch(test_state, op.state_updates.test_state_patch)
        _merge_patch(turn_state, op.state_updates.turn_state_patch)

    return env.model_copy(
        update={
            "task_state": task_state,
            "data_state": data_state,
            "tool_state": tool_state,
            "visible_state": visible_state,
            "gold_state": gold_state,
            "verifier_state": verifier_state,
            "test_state": test_state,
            "turn_state": turn_state,
            "operator_instances": [op.model_dump() for op in operator_instances],
        }
    )


def _merge_patch(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key].update(value)
        else:
            target[key] = value
