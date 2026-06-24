from __future__ import annotations


def plan_operator_specs(axis_priority: list[str], axis_to_operator: dict[str, list[str]], difficulty_budget: dict[str, int]) -> list[dict[str, int | str]]:
    specs: list[dict[str, int | str]] = []
    for axis in axis_priority:
        strength = int(difficulty_budget.get(axis, 0))
        if strength <= 0:
            continue
        operator_names = axis_to_operator.get(axis, [])
        if not operator_names:
            continue
        specs.append({"axis": axis, "name": operator_names[0], "strength": strength})
    return specs


def plan_operators(axis_priority: list[str], axis_to_operator: dict[str, list[str]], difficulty_budget: dict[str, int]) -> list[str]:
    return [str(spec["name"]) for spec in plan_operator_specs(axis_priority, axis_to_operator, difficulty_budget)]
