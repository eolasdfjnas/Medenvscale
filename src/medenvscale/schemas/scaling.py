from __future__ import annotations

from typing import Literal

from .common import StrictBaseModel

AXES = ["D", "C", "A", "V"]
AXIS_WEIGHT_MIN = 1
AXIS_WEIGHT_MAX = 7


class SecondaryAxisWeightHint(StrictBaseModel):
    task_type: str
    relevance: float
    axis_weight_hint: dict[str, int]
    reason: str | None = None


class AxisWeightPlannerResult(StrictBaseModel):
    primary_axis_weight_hint: dict[str, int]
    secondary_axis_weight_hints: list[SecondaryAxisWeightHint] = []
    axis_weight_reason: str | None = None


class ToolSpec(StrictBaseModel):
    tool_name: str
    description: str
    input_schema: dict
    output_schema: dict | None = None
    when_to_use: str
    limitations: list[str] = []
    examples: list[dict] = []


class ToolBudget(StrictBaseModel):
    max_total_tool_calls: int
    max_calls_per_tool: dict[str, int]
    max_consecutive_calls_per_tool: dict[str, int] = {}
    max_debug_calls: int = 0
    max_validation_calls: int = 0


class OutputRequirement(StrictBaseModel):
    output_format: Literal["text", "json", "code", "file", "table"]
    required_fields: list[str] = []
    forbidden_fields: list[str] = []
    json_schema: dict | None = None
    strict: bool = True


class ToolConfig(StrictBaseModel):
    env_id: str
    global_level: Literal["M1", "M2", "M3", "M4"]
    planning_source: Literal["llm", "fallback", "repaired"]
    allowed_tools: list[ToolSpec]
    tool_budget: ToolBudget
    output_requirement: OutputRequirement
    tool_choice_reason: str
    budget_reason: str
    related_axes: list[str] = []
    validation_trace: list[str] = []


class ScalingPlan(StrictBaseModel):
    env_id: str
    global_level: Literal["M1", "M2", "M3", "M4"]
    task_type: str
    secondary_task_types: list[str] = []
    domain: str
    solution_form: str
    axis_weight_source: Literal["llm", "fallback", "repaired"]
    primary_axis_weight_hint: dict[str, int]
    secondary_axis_weight_hints: list[SecondaryAxisWeightHint] = []
    axis_weight_reason: str | None = None
    axis_priority: list[str]
    final_axis_weights: dict[str, float]
    axis_weight_fusion_mode: str = "primary_plus_relevance_weighted_secondary"
    secondary_fusion_strength: float = 0.30
    selected_axes: list[str]
    axis_intensity: dict[str, int]
    total_intensity: int
    sampling_seed: int
    allow_multiturn: bool
    allow_adversarial: bool
    require_safety_gate: bool


class StateUpdates(StrictBaseModel):
    task_state_patch: dict = {}
    data_state_patch: dict = {}
    tool_state_patch: dict = {}
    visible_state_patch: dict = {}
    gold_state_patch: dict = {}
    verifier_state_patch: dict = {}
    test_state_patch: dict = {}
    turn_state_patch: dict = {}


class VerifierDelta(StrictBaseModel):
    new_checks: list[dict] = []
    new_hidden_tests: list[dict] = []
    exception_tests: list[dict] = []
    numeric_tolerance_tests: list[dict] = []
    array_close_tests: list[dict] = []
    dataframe_equal_tests: list[dict] = []
    file_output_tests: list[dict] = []
    object_state_tests: list[dict] = []
    static_checks: list[dict] = []
    expected_failure_modes: list[str] = []


class OperatorConstraints(StrictBaseModel):
    must_preserve_core_task: bool = True
    must_not_change_ground_truth: bool = True
    must_not_leak_answer: bool = True
    must_keep_verifier_executable: bool = True
    must_not_change_solution_form: bool = True


class VerificationContract(StrictBaseModel):
    generation_mode: Literal["gold_compatible", "gold_regenerating"] = "gold_compatible"
    gold_solution_must_pass: bool = True
    must_add_executable_test: bool = True
    must_update_verifier_if_behavior_changes: bool = True
    must_not_change_solution_form: bool = True
    must_not_leak_answer: bool = True


class DynamicOperatorInstance(StrictBaseModel):
    operator_id: str
    axis: Literal["D", "C", "A", "V"]
    operator_type: str
    operator_intensity: int
    transformation_goal: str
    rationale: str
    semantic_change: bool = False
    state_updates: StateUpdates
    verifier_delta: VerifierDelta
    semantic_test_specs: list[dict] = []
    output_requirements: list[str] = []
    output_constraint_spec: dict = {}
    gold_update_policy: dict = {}
    expected_failure_modes: list[str] = []
    rubric_delta: list[dict] = []
    expected_effect: dict = {}
    verification_contract: VerificationContract
    constraints: OperatorConstraints
