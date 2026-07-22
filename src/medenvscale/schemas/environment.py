from __future__ import annotations

from typing import Any

from medenvscale.classify.taxonomy import normalize_domain_name, normalize_task_type_name

from .common import StrictBaseModel
from .difficulty import DifficultyProfile
from .routing import DomainHint


class ExecutableEnvSpec(StrictBaseModel):
    env_id: str
    original_task_id: str
    split: str
    problem: str
    context: str
    signature: str | None = None
    solution_form: str
    primary_domain: str
    domain: str | None = None
    secondary_domains: list[DomainHint] = []
    primary_task_type: str
    task_type: str | None = None
    secondary_task_types: list[str] = []
    verifier_type_hint: str | None = None
    code: str | None = None
    gold_solution: str
    seed_gold_solution: str | None = None
    scaled_gold_solution: str | None = None
    scaled_executable_gold_code: str | None = None
    scaled_oracle_cases: list[dict[str, Any]] = []
    validated_oracle_cases: list[dict[str, Any]] = []
    scaled_oracle_case_failures: list[dict[str, Any]] = []
    scaled_oracle_coverage_summary: dict[str, Any] = {}
    scaled_case_plan: dict[str, Any] = {}
    oracle_case_validation_report: list[dict[str, Any]] = []
    oracle_case_rule_repair_report: list[dict[str, Any]] = []
    oracle_case_repair_trace: list[dict[str, Any]] = []
    scaled_gold_case_execution_report: list[dict[str, Any]] = []
    hidden_tests_mode: str | None = None
    seed_ground_truth_output_signature: dict[str, Any] = {}
    scaled_ground_truth_output_signature: dict[str, Any] = {}
    seed_execution_case: dict[str, Any] = {}
    seed_case_audit: dict[str, Any] = {}
    output_requirements: list[str] = []
    output_requirement_metadata: list[dict[str, Any]] = []
    output_constraint_spec: dict[str, Any] = {}
    repair_trace: list[dict[str, Any]] = []
    resource_files: list[str] = []
    task_state: dict[str, Any] = {}
    data_state: dict[str, Any] = {}
    tool_state: dict[str, Any] = {}
    visible_state: dict[str, Any] = {"include": [], "hide": []}
    gold_state: dict[str, Any] = {}
    verifier_state: dict[str, Any] = {}
    test_state: dict[str, Any] = {}
    turn_state: dict[str, Any] = {}
    resource_manifest: list[dict[str, Any]] = []
    action_space: list[str] = []
    execution_config: dict[str, Any] = {}
    safety_gate_required: bool = False
    robust_verifier_required: bool = False
    suitable_for_dpo_or_stress_test: bool = False
    immutable_fields: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    user_prompt: str | None = None
    system_prompt: str | None = None
    prompt_format: str = "single_turn"
    base_difficulty: dict[str, Any] | None = None
    difficulty: DifficultyProfile | None = None
    tool_config: dict[str, Any] | None = None
    scaling: dict[str, Any] | None = None
    scaling_plan: dict[str, Any] | None = None
    operator_mode: str | None = None
    operator_instances: list[dict[str, Any]] = []
    verifier_delta: dict[str, Any] | None = None
    verifier_spec: dict[str, Any] | None = None
    hidden_tests: list[dict[str, Any]] = []
    semantic_test_specs: list[dict[str, Any]] = []
    gold_change_metadata: dict[str, Any] = {}
    operator_realization_report: list[dict[str, Any]] = []
    gate_results: dict[str, Any] = {}
    stage05_quality_report: dict[str, Any] | None = None
    stage05_passed: bool | None = None
    question_point_ids: list[str] = []
    rubric_ids: list[str] = []
    rubrics: list[dict[str, Any]] = []
    ideal_answer: str | None = None
    rejected_answer: str | None = None
    quality_flags: list[str] = []
    blocking_quality_flags: list[str] = []
    export_status: str = "raw"

    def __init__(self, **kwargs: Any) -> None:
        legacy_oracle_examples = kwargs.pop("oracle_examples", None)
        legacy_executable_examples = kwargs.pop("executable_examples", None)
        legacy_oracle_failures = kwargs.pop("oracle_example_failures", None)
        legacy_oracle_coverage = kwargs.pop("oracle_coverage_summary", None)
        kwargs["primary_domain"] = normalize_domain_name(kwargs.get("primary_domain") or kwargs.get("domain"))
        kwargs["domain"] = normalize_domain_name(kwargs.get("domain") or kwargs["primary_domain"])
        secondary_domains = kwargs.get("secondary_domains", []) or []
        if isinstance(secondary_domains, dict):
            secondary_domains = [secondary_domains]
        kwargs["secondary_domains"] = [item if isinstance(item, DomainHint) else DomainHint.model_validate(item) for item in secondary_domains]
        kwargs["primary_task_type"] = normalize_task_type_name(kwargs.get("primary_task_type") or kwargs.get("task_type"))
        kwargs["task_type"] = normalize_task_type_name(kwargs.get("task_type") or kwargs["primary_task_type"])
        kwargs.setdefault("visible_state", {"include": [], "hide": []})
        kwargs.setdefault("scaling", kwargs.get("scaling_plan"))
        kwargs.setdefault("operator_mode", kwargs.get("gold_state", {}).get("operator_mode"))
        kwargs.setdefault("seed_gold_solution", kwargs.get("gold_solution"))
        kwargs.setdefault("scaled_gold_solution", kwargs.get("gold_solution"))
        kwargs.setdefault("scaled_executable_gold_code", kwargs.get("scaled_gold_solution"))
        kwargs.setdefault(
            "scaled_oracle_cases",
            kwargs.get("scaled_oracle_cases")
            or legacy_oracle_examples
            or legacy_executable_examples
            or [],
        )
        kwargs.setdefault(
            "validated_oracle_cases",
            kwargs.get("validated_oracle_cases")
            or kwargs.get("scaled_oracle_cases")
            or legacy_oracle_examples
            or legacy_executable_examples
            or [],
        )
        kwargs.setdefault(
            "scaled_oracle_case_failures",
            kwargs.get("scaled_oracle_case_failures") or legacy_oracle_failures or [],
        )
        kwargs.setdefault(
            "scaled_oracle_coverage_summary",
            kwargs.get("scaled_oracle_coverage_summary") or legacy_oracle_coverage or {},
        )
        super().__init__(**kwargs)


ClinicalEnvironment = ExecutableEnvSpec
