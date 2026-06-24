from __future__ import annotations

import random

from medenvscale.schemas.scaling import AXES, AxisWeightPlannerResult, ScalingPlan
from medenvscale.utils import stable_hash


def build_scaling_plan(
    env_id: str,
    global_level: str,
    task_type: str,
    secondary_task_types: list[str],
    domain: str,
    solution_form: str,
    axis_priority_cfg: dict,
    budgets_cfg: dict,
    fusion_cfg: dict,
    axis_weights: AxisWeightPlannerResult,
    axis_weight_source: str = "fallback",
) -> ScalingPlan:
    budget = budgets_cfg["m_level_budgets"][global_level]
    axis_priority = list(axis_priority_cfg["task_axis_priority"][task_type]["axis_priority"])
    secondary_strength = float(fusion_cfg["axis_weight_fusion"]["secondary_fusion_strength"])
    final_weights = fuse_axis_weights(axis_weights, secondary_strength)
    seed = int(stable_hash({"env_id": env_id, "level": global_level})[:8], 16)
    selected_axes = sample_selected_axes(global_level, budget, axis_priority, final_weights, seed)
    axis_intensity, total_intensity = allocate_axis_intensity(selected_axes, final_weights, budget, seed + 17)
    plan = ScalingPlan(
        env_id=env_id,
        global_level=global_level,
        task_type=task_type,
        secondary_task_types=secondary_task_types[:3],
        domain=domain,
        solution_form=solution_form,
        axis_weight_source=axis_weight_source,
        primary_axis_weight_hint=axis_weights.primary_axis_weight_hint,
        secondary_axis_weight_hints=axis_weights.secondary_axis_weight_hints,
        axis_weight_reason=axis_weights.axis_weight_reason,
        axis_priority=axis_priority,
        final_axis_weights=final_weights,
        secondary_fusion_strength=secondary_strength,
        selected_axes=selected_axes,
        axis_intensity=axis_intensity,
        total_intensity=total_intensity,
        sampling_seed=seed,
        allow_multiturn=global_level in {"M3", "M4"},
        allow_adversarial=bool(budget.get("allow_adversarial", False)),
        require_safety_gate=bool(budget.get("require_safety_gate", False)),
    )
    validate_scaling_plan(plan, budgets_cfg)
    return plan


def fuse_axis_weights(axis_weights: AxisWeightPlannerResult, secondary_strength: float) -> dict[str, float]:
    fused = {axis: float(axis_weights.primary_axis_weight_hint[axis]) for axis in AXES}
    secondary = axis_weights.secondary_axis_weight_hints
    relevance_total = sum(max(0.0, hint.relevance) for hint in secondary)
    if relevance_total <= 0:
        return fused
    for axis in AXES:
        average = sum(hint.relevance * hint.axis_weight_hint[axis] for hint in secondary) / relevance_total
        fused[axis] += secondary_strength * average
    return fused


def sample_selected_axes(
    global_level: str,
    budget: dict,
    axis_priority: list[str],
    final_axis_weights: dict[str, float],
    seed: int,
) -> list[str]:
    if global_level == "M1":
        return []

    rng = random.Random(seed)
    allowed_axes = [axis for axis in budget.get("allowed_axes", AXES) if axis in AXES]
    selected: list[str] = []

    for axis in axis_priority[: int(budget.get("include_primary_top_k", 0))]:
        if axis in allowed_axes and axis not in selected:
            selected.append(axis)

    lower, upper = budget["num_axes_range"]
    target = rng.randint(lower, upper)
    candidates = [axis for axis in allowed_axes if axis not in selected]
    if not budget.get("allow_adversarial", False):
        candidates = [axis for axis in candidates if axis != "A"]
    while len(selected) < target and candidates:
        axis = weighted_pick(candidates, final_axis_weights, rng)
        selected.append(axis)
        candidates.remove(axis)
    return selected


def allocate_axis_intensity(
    selected_axes: list[str],
    final_axis_weights: dict[str, float],
    budget: dict,
    seed: int,
) -> tuple[dict[str, int], int]:
    intensity = {axis: 0 for axis in AXES}
    low, high = budget["total_intensity_range"]
    if not selected_axes or high == 0:
        return intensity, 0
    rng = random.Random(seed)
    total = rng.randint(low, high)
    for axis in selected_axes:
        intensity[axis] = 1
    remaining = total - len(selected_axes)
    cap = int(budget["per_axis_intensity_range"][1])
    while remaining > 0:
        eligible = [axis for axis in selected_axes if intensity[axis] < per_axis_cap(axis, cap, budget)]
        if not eligible:
            break
        axis = weighted_pick(eligible, final_axis_weights, rng)
        intensity[axis] += 1
        remaining -= 1
    return intensity, sum(intensity.values())


def weighted_pick(axes: list[str], weights: dict[str, float], rng: random.Random) -> str:
    total = sum(max(weights.get(axis, 0.0), 0.01) for axis in axes)
    draw = rng.random() * total
    cursor = 0.0
    for axis in axes:
        cursor += max(weights.get(axis, 0.0), 0.01)
        if draw <= cursor:
            return axis
    return axes[-1]


def per_axis_cap(axis: str, default_cap: int, budget: dict) -> int:
    if axis == "A" and not budget.get("allow_adversarial", False):
        return 0
    return default_cap


def validate_scaling_plan(plan: ScalingPlan, budgets_cfg: dict) -> None:
    budget = budgets_cfg["m_level_budgets"][plan.global_level]
    allowed_axes = set(axis for axis in budget.get("allowed_axes", AXES) if axis in AXES)
    lower_axes, upper_axes = budget["num_axes_range"]
    if not (lower_axes <= len(plan.selected_axes) <= upper_axes):
        raise ValueError(f"{plan.env_id}: selected_axes out of range for {plan.global_level}")
    if any(axis not in allowed_axes for axis in plan.selected_axes):
        raise ValueError(f"{plan.env_id}: selected_axes must stay within budget.allowed_axes")
    lower_total, upper_total = budget["total_intensity_range"]
    if not (lower_total <= plan.total_intensity <= upper_total):
        raise ValueError(f"{plan.env_id}: total_intensity out of range for {plan.global_level}")
    for axis in AXES:
        value = plan.axis_intensity.get(axis, 0)
        if axis not in plan.selected_axes and value != 0:
            raise ValueError(f"{plan.env_id}: unselected axis {axis} has non-zero intensity")
        if axis in plan.selected_axes and value < 1:
            raise ValueError(f"{plan.env_id}: selected axis {axis} has zero intensity")
        if value < 0 or value > 3:
            raise ValueError(f"{plan.env_id}: axis {axis} intensity must be in [0, 3]")
    if sum(plan.axis_intensity.values()) != plan.total_intensity:
        raise ValueError(f"{plan.env_id}: axis intensity sum mismatch")
    if plan.global_level == "M1" and plan.selected_axes:
        raise ValueError(f"{plan.env_id}: M1 must not select axes")
    required_top_k = int(budget.get("include_primary_top_k", 0))
    required_axes = [axis for axis in plan.axis_priority[:required_top_k] if axis in allowed_axes]
    if not set(required_axes).issubset(set(plan.selected_axes)):
        raise ValueError(f"{plan.env_id}: required top-priority axes missing for {plan.global_level}")
    if plan.global_level == "M2" and len(plan.selected_axes) != 1:
        raise ValueError(f"{plan.env_id}: M2 must select exactly one axis")
    if plan.global_level == "M3" and len(plan.selected_axes) != 2:
        raise ValueError(f"{plan.env_id}: M3 must select exactly two axes")
    if plan.global_level == "M4" and not (3 <= len(plan.selected_axes) <= 4):
        raise ValueError(f"{plan.env_id}: M4 must select three to four axes")
    if not plan.allow_adversarial and plan.axis_intensity.get("A", 0) > 0:
        raise ValueError(f"{plan.env_id}: A axis not allowed for {plan.global_level}")
