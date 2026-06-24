from __future__ import annotations

from medenvscale.schemas.scaling import AXES, DynamicOperatorInstance, StateUpdates

STATE_PATCH_FIELDS = [
    "task_state_patch",
    "data_state_patch",
    "tool_state_patch",
    "visible_state_patch",
    "gold_state_patch",
    "verifier_state_patch",
    "test_state_patch",
    "turn_state_patch",
]


def has_non_empty_state_update(op: DynamicOperatorInstance) -> bool:
    state_updates = op.state_updates.model_dump()
    return any(bool(state_updates.get(field)) for field in STATE_PATCH_FIELDS)


def has_critical_semantic_update(op: DynamicOperatorInstance) -> bool:
    return any(
        [
            bool(op.state_updates.visible_state_patch),
            bool(op.state_updates.gold_state_patch),
            bool(op.state_updates.verifier_state_patch),
            bool(op.state_updates.test_state_patch),
            bool(op.state_updates.data_state_patch),
        ]
    )


def repair_missing_state_updates(op: DynamicOperatorInstance) -> DynamicOperatorInstance:
    updates = op.state_updates.model_dump()
    for field in STATE_PATCH_FIELDS:
        if not isinstance(updates.get(field), dict):
            updates[field] = {}

    if op.axis == "D":
        updates["task_state_patch"].setdefault(
            "extra_constraints",
            ["Handle scaled input/data variants from the oracle cases."],
        )
        updates["data_state_patch"].setdefault("resource_variants", [f"{op.operator_id}_resource_variant"])
        updates["visible_state_patch"].setdefault(
            "input_description",
            ["Scaled inputs may differ in format, shape, or parsing assumptions."],
        )
        updates["test_state_patch"].setdefault("tests_with_new_resource_variants", [op.transformation_goal])
        _mark_gold_changed(updates, "D-axis modifies executable input/data semantics.")
    elif op.axis == "C":
        updates["task_state_patch"].setdefault("extra_constraints", [op.transformation_goal])
        updates["visible_state_patch"].setdefault("output_constraints", [op.transformation_goal])
        updates["verifier_state_patch"].setdefault("constraint_checks", [op.transformation_goal])
        updates["test_state_patch"].setdefault("constraint_hidden_tests", [op.transformation_goal])
        _mark_gold_changed(updates, "C-axis modifies executable constraints or return contract.")
    elif op.axis == "A":
        updates["task_state_patch"].setdefault("shortcut_traps", [op.transformation_goal])
        updates["task_state_patch"].setdefault("must_not_follow_shortcut", True)
        updates["visible_state_patch"].setdefault("robustness_trap", op.transformation_goal)
        updates["visible_state_patch"].setdefault(
            "must_not_assume",
            ["Do not hardcode seed-only filenames, parameters, or happy-path assumptions."],
        )
        updates["verifier_state_patch"].setdefault("anti_shortcut_checks", [op.transformation_goal])
        updates["test_state_patch"].setdefault("shortcut_hidden_tests", [op.transformation_goal])
        _mark_gold_changed(updates, "A-axis modifies robustness behavior under adversarial cases.")
    elif op.axis == "V":
        updates["visible_state_patch"].setdefault(
            "execution_requirements",
            ["Solutions must satisfy stronger validated oracle case coverage."],
        )
        updates["verifier_state_patch"].setdefault(
            "stronger_checks",
            [op.transformation_goal],
        )
        updates["test_state_patch"].setdefault("semantic_hidden_tests", [op.transformation_goal])
        updates["gold_state_patch"].setdefault("gold_changed", False)
        updates["gold_state_patch"].setdefault("answer_invariant", True)
        updates["gold_state_patch"].setdefault("seed_gold_compatible_with_scaled_task", True)
        updates["gold_state_patch"].setdefault(
            "gold_change_reason",
            "V-axis strengthens executable verification without changing task semantics by itself.",
        )
    else:
        updates["task_state_patch"].setdefault("operator_goal", op.transformation_goal)
    return op.model_copy(update={"state_updates": StateUpdates.model_validate(updates)})


def repair_operator_instances(operator_instances: list[DynamicOperatorInstance]) -> list[DynamicOperatorInstance]:
    return [repair_missing_state_updates(op) for op in operator_instances]


def validate_dynamic_operator_instances(
    operator_instances: list[DynamicOperatorInstance],
    scaling_plan: dict,
) -> list[str]:
    errors: list[str] = []
    by_axis: dict[str, int] = {axis: 0 for axis in AXES}
    selected = set(scaling_plan["selected_axes"])
    expected_intensity = scaling_plan["axis_intensity"]

    for op in operator_instances:
        visible_patch = op.state_updates.visible_state_patch or {}
        data_patch = op.state_updates.data_state_patch or {}
        verifier_patch = op.state_updates.verifier_state_patch or {}
        if op.axis not in selected:
            errors.append(f"{op.operator_id}: axis {op.axis} not selected")
        if op.operator_intensity not in {1, 2, 3}:
            errors.append(f"{op.operator_id}: invalid operator_intensity")
        patches = op.state_updates.model_dump()
        if not has_non_empty_state_update(op):
            errors.append(f"{op.operator_id}: missing structured state update")
        if not has_critical_semantic_update(op):
            errors.append(f"{op.operator_id}: missing semantic state update")
        if "user_prompt" in str(patches):
            errors.append(f"{op.operator_id}: operator must not write user_prompt")
        if op.constraints.must_not_change_solution_form and op.state_updates.task_state_patch.get("solution_form"):
            errors.append(f"{op.operator_id}: operator must preserve solution_form")
        if op.axis == "D" and not data_patch:
            errors.append(f"{op.operator_id}: D axis must add a data/input patch")
        if op.axis == "C" and not (
            op.state_updates.task_state_patch.get("extra_constraints")
            or visible_patch.get("output_constraints")
            or verifier_patch.get("constraint_checks")
        ):
            errors.append(f"{op.operator_id}: C axis must add executable constraints")
        if op.axis == "A" and not (
            visible_patch.get("robustness_trap")
            or visible_patch.get("must_not_assume")
            or verifier_patch.get("anti_shortcut_checks")
        ):
            errors.append(f"{op.operator_id}: A axis must add a robustness challenge")
        if op.axis == "V" and not (
            op.verifier_delta.new_hidden_tests
            or op.verifier_delta.new_checks
            or op.semantic_test_specs
            or verifier_patch.get("stronger_checks")
        ):
            errors.append(f"{op.operator_id}: V axis must add verification signals")
        by_axis[op.axis] += op.operator_intensity

    for axis in selected:
        if by_axis[axis] == 0:
            errors.append(f"selected axis {axis} has no operator")
        if by_axis[axis] != expected_intensity.get(axis, 0):
            errors.append(f"axis {axis} intensity mismatch")
    for axis in AXES:
        if expected_intensity.get(axis, 0) == 0 and by_axis[axis] != 0:
            errors.append(f"axis {axis} has operator despite zero intensity")
    return errors


def _mark_gold_changed(updates: dict, reason: str) -> None:
    updates["gold_state_patch"].setdefault("gold_changed", True)
    updates["gold_state_patch"].setdefault("answer_invariant", False)
    updates["gold_state_patch"].setdefault("seed_gold_compatible_with_scaled_task", False)
    updates["gold_state_patch"].setdefault("gold_change_reason", reason)
