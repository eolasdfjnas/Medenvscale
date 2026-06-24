from __future__ import annotations

import json
from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.schemas.scaling import (
    AXES,
    DynamicOperatorInstance,
    OperatorConstraints,
    StateUpdates,
    VerificationContract,
    VerifierDelta,
)


def synthesize_dynamic_operator_instances(
    env_id: str,
    task_id: str,
    task_type: str,
    domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    scaling_plan: dict,
    tool_config: dict,
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
    seed_task: dict[str, Any] | None = None,
    base_environment: ExecutableEnvSpec | None = None,
    domain_concepts: list[str] | None = None,
    intensity_rubric: dict[str, Any] | None = None,
) -> list[DynamicOperatorInstance]:
    if llm_client is not None and prompt_runner is not None:
        try:
            prompt = prompt_runner.render(
                "dynamic_verifiable_operator_planner.jinja",
                seed_task=json.dumps(seed_task or {}, ensure_ascii=False, indent=2),
                base_environment=json.dumps((base_environment.model_dump() if base_environment else {}), ensure_ascii=False, indent=2),
                domain=domain,
                task_type=task_type,
                solution_form=solution_form,
                domain_concepts=json.dumps(domain_concepts or [], ensure_ascii=False),
                scaling_plan=json.dumps(scaling_plan, ensure_ascii=False, indent=2),
                tool_config=json.dumps(tool_config, ensure_ascii=False, indent=2),
                intensity_rubric=json.dumps(intensity_rubric or {}, ensure_ascii=False, indent=2),
            )
            response = llm_client.complete_json(
                task_name="dynamic_verifiable_operator_planner_4axis",
                prompt=prompt,
                context={
                    "env_id": env_id,
                    "task_id": task_id,
                    "task_type": task_type,
                    "domain": domain,
                    "secondary_domains": _secondary_domain_names(secondary_domains),
                    "solution_form": solution_form,
                    "scaling_plan": scaling_plan,
                    "tool_config": tool_config,
                },
                mock_builder=_mock_operator_builder,
            )
            return repair_operator_payload(
                payload=response.payload,
                env_id=env_id,
                task_id=task_id,
                task_type=task_type,
                domain=domain,
                secondary_domains=secondary_domains,
                solution_form=solution_form,
                scaling_plan=scaling_plan,
                tool_config=tool_config,
            )
        except Exception:
            pass

    return fallback_operator_instances(
        env_id=env_id,
        task_id=task_id,
        task_type=task_type,
        domain=domain,
        secondary_domains=secondary_domains,
        solution_form=solution_form,
        scaling_plan=scaling_plan,
        tool_config=tool_config,
    )


def repair_operator_payload(
    payload: dict[str, Any],
    env_id: str,
    task_id: str,
    task_type: str,
    domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    scaling_plan: dict,
    tool_config: dict,
) -> list[DynamicOperatorInstance]:
    selected_axes = set(scaling_plan["selected_axes"])
    expected = dict(scaling_plan["axis_intensity"])
    raw_instances = payload.get("operator_instances", payload)
    if isinstance(raw_instances, dict):
        raw_instances = [raw_instances]
    if not isinstance(raw_instances, list):
        return fallback_operator_instances(
            env_id=env_id,
            task_id=task_id,
            task_type=task_type,
            domain=domain,
            secondary_domains=secondary_domains,
            solution_form=solution_form,
            scaling_plan=scaling_plan,
            tool_config=tool_config,
        )

    by_axis: dict[str, list[DynamicOperatorInstance]] = {axis: [] for axis in AXES}
    for index, item in enumerate(raw_instances, start=1):
        instance = _coerce_operator_instance(
            env_id=env_id,
            task_id=task_id,
            task_type=task_type,
            domain=domain,
            secondary_domains=secondary_domains,
            solution_form=solution_form,
            raw=item,
            sequence=index,
        )
        if instance is None or instance.axis not in selected_axes:
            continue
        by_axis[instance.axis].append(instance)

    repaired: list[DynamicOperatorInstance] = []
    for axis in AXES:
        target = int(expected.get(axis, 0))
        if target <= 0:
            continue
        axis_instances = by_axis[axis]
        total = sum(item.operator_intensity for item in axis_instances)
        if total > target:
            axis_instances = _trim_axis_instances(axis_instances, target)
            total = sum(item.operator_intensity for item in axis_instances)
        if total < target:
            axis_instances.extend(
                fallback_axis_operators(
                    env_id=env_id,
                    task_id=task_id,
                    task_type=task_type,
                    domain=domain,
                    secondary_domains=secondary_domains,
                    solution_form=solution_form,
                    axis=axis,
                    intensity=target - total,
                    tool_config=tool_config,
                    start_index=len(axis_instances) + 1,
                )
            )
        repaired.extend(axis_instances)
    return repaired


def fallback_operator_instances(
    env_id: str,
    task_id: str,
    task_type: str,
    domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    scaling_plan: dict,
    tool_config: dict,
) -> list[DynamicOperatorInstance]:
    operators: list[DynamicOperatorInstance] = []
    for axis in AXES:
        intensity = int(scaling_plan["axis_intensity"].get(axis, 0))
        if intensity <= 0:
            continue
        operators.extend(
            fallback_axis_operators(
                env_id=env_id,
                task_id=task_id,
                task_type=task_type,
                domain=domain,
                secondary_domains=secondary_domains,
                solution_form=solution_form,
                axis=axis,
                intensity=intensity,
                tool_config=tool_config,
                start_index=1,
            )
        )
    return operators


def fallback_axis_operators(
    env_id: str,
    task_id: str,
    task_type: str,
    domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    axis: str,
    intensity: int,
    tool_config: dict,
    start_index: int,
) -> list[DynamicOperatorInstance]:
    chunks = _split_intensity(intensity)
    secondary_domain_names = _secondary_domain_names(secondary_domains)
    operators: list[DynamicOperatorInstance] = []
    for offset, chunk in enumerate(chunks, start=start_index):
        instance = DynamicOperatorInstance(
            operator_id=f"{env_id}_{axis.lower()}_{offset:02d}",
            axis=axis,
            operator_type=f"{task_type}_{axis.lower()}_{domain}_{offset:02d}",
            operator_intensity=chunk,
            transformation_goal=transformation_goal(axis, task_type, solution_form, domain, secondary_domain_names),
            rationale=build_rationale(axis, domain, secondary_domain_names),
            semantic_change=semantic_change_for_axis(axis),
            state_updates=state_updates_for_axis(axis, chunk, tool_config, domain, secondary_domain_names),
            verifier_delta=verifier_delta_for_axis(axis, task_id, chunk, domain, secondary_domain_names),
            semantic_test_specs=semantic_test_specs_for_axis(axis, task_id, chunk),
            output_requirements=output_requirements_for_axis(axis, chunk),
            output_constraint_spec={"checks": []},
            gold_update_policy=gold_update_policy_for_axis(axis),
            expected_failure_modes=_expected_failure_modes_for_axis(axis),
            rubric_delta=rubric_delta_for_axis(axis, domain, secondary_domain_names),
            expected_effect={
                "axis": axis,
                "intensity": chunk,
                "primary_domain": domain,
                "secondary_domains": secondary_domain_names,
                "planning_source": "fallback",
            },
            verification_contract=VerificationContract(),
            constraints=OperatorConstraints(),
        )
        _attach_semantic_generation_metadata(instance, task_id=task_id)
        _annotate_operator_verifier_delta(instance)
        operators.append(instance)
    return operators


def _coerce_operator_instance(
    env_id: str,
    task_id: str,
    task_type: str,
    domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    raw: Any,
    sequence: int,
) -> DynamicOperatorInstance | None:
    if not isinstance(raw, dict):
        return None
    axis = str(raw.get("axis") or "").strip()
    if axis not in AXES:
        return None
    try:
        intensity = int(raw.get("operator_intensity", 1))
    except (TypeError, ValueError):
        intensity = 1
    intensity = min(3, max(1, intensity))
    secondary_domain_names = _secondary_domain_names(secondary_domains)
    payload = dict(raw)
    payload.setdefault("operator_id", f"{env_id}_{axis.lower()}_{sequence:02d}")
    payload.setdefault("operator_type", f"{task_type}_{axis.lower()}_{domain}_{sequence:02d}")
    payload.setdefault("transformation_goal", transformation_goal(axis, task_type, solution_form, domain, secondary_domain_names))
    payload.setdefault("rationale", build_rationale(axis, domain, secondary_domain_names))
    payload.setdefault("semantic_change", semantic_change_for_axis(axis))
    payload.setdefault("state_updates", state_updates_for_axis(axis, intensity, {}, domain, secondary_domain_names).model_dump())
    payload.setdefault("verifier_delta", verifier_delta_for_axis(axis, task_id, intensity, domain, secondary_domain_names).model_dump())
    payload.setdefault("semantic_test_specs", semantic_test_specs_for_axis(axis, task_id, intensity))
    payload.setdefault("output_requirements", output_requirements_for_axis(axis, intensity))
    payload.setdefault("output_constraint_spec", {"checks": []})
    payload.setdefault("gold_update_policy", gold_update_policy_for_axis(axis))
    payload.setdefault("expected_failure_modes", _expected_failure_modes_for_axis(axis))
    payload.setdefault("rubric_delta", rubric_delta_for_axis(axis, domain, secondary_domain_names))
    payload.setdefault("expected_effect", {"axis": axis, "intensity": intensity, "primary_domain": domain, "secondary_domains": secondary_domain_names})
    payload.setdefault("verification_contract", VerificationContract().model_dump())
    payload.setdefault("constraints", OperatorConstraints().model_dump())
    payload["operator_intensity"] = intensity
    try:
        instance = DynamicOperatorInstance.model_validate(payload)
        _attach_semantic_generation_metadata(instance, task_id=task_id)
        _annotate_operator_verifier_delta(instance)
        return instance
    except Exception:
        return None


def _trim_axis_instances(instances: list[DynamicOperatorInstance], target: int) -> list[DynamicOperatorInstance]:
    trimmed: list[DynamicOperatorInstance] = []
    total = 0
    for instance in instances:
        if total >= target:
            break
        remaining = target - total
        if instance.operator_intensity <= remaining:
            trimmed.append(instance)
            total += instance.operator_intensity
            continue
        trimmed.append(instance.model_copy(update={"operator_intensity": remaining}))
        total += remaining
    return trimmed


def _split_intensity(total: int) -> list[int]:
    chunks: list[int] = []
    remaining = total
    while remaining > 0:
        chunk = min(2 if remaining > 1 else 1, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def transformation_goal(axis: str, task_type: str, solution_form: str, primary_domain: str, secondary_domain_names: list[str]) -> str:
    goals = {
        "D": "Increase input/data complexity through format variants, malformed records, missing values, or content-based parsing.",
        "C": "Increase computation and contract complexity through parameter rules, edge cases, ordering, or output constraints.",
        "A": "Increase robustness pressure through shortcut traps, conflicting inputs, randomized names, or brittle-assumption checks.",
        "V": "Increase oracle and verification strength through more executable, semantically targeted case coverage.",
    }
    secondary_text = ", ".join(secondary_domain_names) if secondary_domain_names else "none"
    return (
        f"{goals[axis]} Task type={task_type}; solution_form={solution_form}; "
        f"primary_domain={primary_domain}; secondary_domains={secondary_text}."
    )


def state_updates_for_axis(
    axis: str,
    intensity: int,
    tool_config: dict,
    primary_domain: str,
    secondary_domain_names: list[str],
) -> StateUpdates:
    task_patch = {
        "axis_constraints": [axis],
        "axis_intensity": intensity,
        "domain_context": {"primary_domain": primary_domain, "secondary_domains": secondary_domain_names},
    }
    visible_patch: dict[str, Any] = {}
    data_patch: dict[str, Any] = {}
    gold_patch: dict[str, Any] = {}
    verifier_patch: dict[str, Any] = {}
    test_patch: dict[str, Any] = {}

    if axis == "D":
        task_patch["extra_constraints"] = ["Handle richer input variants derived from the validated seed execution path."]
        data_patch["resource_variants"] = [f"d_variant_intensity_{intensity}"]
        data_patch["additional_inputs"] = [f"d_additional_input_{intensity}"]
        data_patch["input_format_variants"] = ["extensionless", "comment_lines", "empty_sections"][: intensity + 1]
        visible_patch["input_description"] = ["Input format and data-shape assumptions are now stricter and must be handled explicitly."]
        visible_patch["resource_complexity_notes"] = ["Scaled cases may alter file layout, formatting, or record structure."]
        verifier_patch["resource_variation_checks"] = [f"d_resource_check_{intensity}"]
        test_patch["tests_with_new_resource_variants"] = [f"d_case_variant_{intensity}"]
        gold_patch.update(
            {
                "gold_changed": True,
                "answer_invariant": False,
                "seed_gold_compatible_with_scaled_task": False,
                "gold_change_reason": "D-axis changes require the executable gold to handle new input/data variants.",
            }
        )
    elif axis == "C":
        task_patch["extra_constraints"] = ["Respect new parameter, boundary, ordering, or return-contract constraints from scaled cases."]
        visible_patch["output_constraints"] = ["Scaled cases introduce stronger computation and output-contract requirements."]
        visible_patch["format_constraints"] = ["Preserve the required return structure and edge-case behavior."]
        verifier_patch["constraint_checks"] = [f"c_constraint_check_{intensity}"]
        test_patch["constraint_hidden_tests"] = [f"c_constraint_case_{intensity}"]
        gold_patch.update(
            {
                "gold_changed": True,
                "answer_invariant": False,
                "seed_gold_compatible_with_scaled_task": False,
                "gold_change_reason": "C-axis changes require updated executable logic for new constraints and edge cases.",
            }
        )
    elif axis == "A":
        task_patch["shortcut_traps"] = ["Avoid brittle shortcuts on duplicate, conflicting, randomized, or misleading inputs."]
        task_patch["must_not_follow_shortcut"] = True
        visible_patch["robustness_trap"] = "Scaled cases include adversarial conditions that punish hardcoding and brittle assumptions."
        visible_patch["robustness_challenges"] = ["Handle duplicate/conflicting inputs and randomized names correctly."]
        visible_patch["must_not_assume"] = ["Do not hardcode common filenames, flags, or only-happy-path inputs."]
        verifier_patch["anti_shortcut_checks"] = [f"a_shortcut_check_{intensity}"]
        verifier_patch["shortcut_rejection_checks"] = [f"a_rejection_check_{intensity}"]
        test_patch["shortcut_hidden_tests"] = [f"a_shortcut_case_{intensity}"]
        gold_patch.update(
            {
                "gold_changed": True,
                "answer_invariant": False,
                "seed_gold_compatible_with_scaled_task": False,
                "gold_change_reason": "A-axis changes require stronger robustness logic in the executable gold.",
            }
        )
    else:
        visible_patch["execution_requirements"] = ["Scaled task must satisfy stronger executable oracle coverage, not just the original seed case."]
        verifier_patch["stronger_checks"] = [f"v_case_coverage_{intensity}"]
        verifier_patch["verifier_hardening"] = ["validated_oracle_cases"]
        test_patch["semantic_hidden_tests"] = [f"v_semantic_case_{intensity}"]
        gold_patch.update(
            {
                "gold_changed": False,
                "answer_invariant": True,
                "seed_gold_compatible_with_scaled_task": True,
                "gold_change_reason": "V-axis strengthens executable verification without changing core task semantics by itself.",
            }
        )

    return StateUpdates(
        task_state_patch=task_patch,
        data_state_patch=data_patch,
        tool_state_patch={},
        visible_state_patch=visible_patch,
        gold_state_patch=gold_patch,
        verifier_state_patch=verifier_patch,
        test_state_patch=test_patch,
        turn_state_patch={},
    )


def verifier_delta_for_axis(axis: str, task_id: str, intensity: int, primary_domain: str, secondary_domain_names: list[str]) -> VerifierDelta:
    new_checks = [
        {
            "check_id": f"{task_id}_{axis.lower()}_{intensity}_check",
            "kind": "semantic_case_alignment",
            "description": transformation_goal(axis, "task", "code", primary_domain, secondary_domain_names),
            "source_axis": axis,
        }
    ]
    expected_failure_modes = _expected_failure_modes_for_axis(axis)
    return VerifierDelta(
        new_checks=new_checks,
        new_hidden_tests=[],
        exception_tests=[],
        numeric_tolerance_tests=[],
        array_close_tests=[],
        dataframe_equal_tests=[],
        file_output_tests=[],
        object_state_tests=[],
        static_checks=[],
        expected_failure_modes=expected_failure_modes,
    )


def semantic_test_specs_for_axis(axis: str, task_id: str, intensity: int) -> list[dict[str, Any]]:
    if axis == "D":
        return [
            {
                "spec_id": f"{task_id}_{axis.lower()}_{intensity}_spec",
                "semantic_intent": "The solution must handle a scaled input/data variant that the seed path did not require.",
                "target_constraint": "Support richer file/data shapes such as missing sections, altered formatting, or content-based parsing.",
                "expected_failure_mode": "solution_only_handles_original_seed_input_shape",
                "test_template_type": "data_variant_case",
                "input_variant": {"kind": "data_variant", "value": "scaled_seed_case_variant"},
                "expected_behavior": {"kind": "oracle_output_match", "mode": "data_variant"},
                "test_case_description": "Semantic case for D-axis data/input complexity.",
            }
        ]
    if axis == "C":
        return [
            {
                "spec_id": f"{task_id}_{axis.lower()}_{intensity}_spec",
                "semantic_intent": "The solution must satisfy new boundary, parameter, or return-contract constraints.",
                "target_constraint": "Scaled cases add explicit edge-case or output-contract requirements.",
                "expected_failure_mode": "solution_matches_seed_case_but_breaks_new_constraint",
                "test_template_type": "constraint_boundary_case",
                "input_variant": {"kind": "constraint_variant", "value": "scaled_seed_case_variant"},
                "expected_behavior": {"kind": "oracle_output_match", "mode": "constraint_boundary"},
                "test_case_description": "Semantic case for C-axis computation/constraint complexity.",
            }
        ]
    if axis == "A":
        return [
            {
                "spec_id": f"{task_id}_{axis.lower()}_{intensity}_spec",
                "semantic_intent": "The solution must resist a shortcut or brittle assumption that now fails on adversarial input.",
                "target_constraint": "Scaled cases include conflicting, duplicated, randomized, or misleading inputs.",
                "expected_failure_mode": "solution_hardcodes_seed_assumption_or_brittle_shortcut",
                "test_template_type": "anti_shortcut_case",
                "input_variant": {"kind": "adversarial_variant", "value": "scaled_seed_case_variant"},
                "expected_behavior": {"kind": "oracle_output_match", "mode": "anti_shortcut"},
                "test_case_description": "Semantic case for A-axis robustness complexity.",
            }
        ]
    return [
        {
            "spec_id": f"{task_id}_{axis.lower()}_{intensity}_spec",
            "semantic_intent": "The solution must pass stronger executable oracle coverage across main, edge, and adversarial variants.",
            "target_constraint": "Validated oracle cases must jointly cover the scaled task semantics.",
            "expected_failure_mode": "solution_passes_seed_case_but_not_stronger_scaled_case_suite",
            "test_template_type": "semantic_coverage_case",
            "input_variant": {"kind": "coverage_variant", "value": "scaled_seed_case_suite"},
            "expected_behavior": {"kind": "oracle_output_match", "mode": "oracle_coverage"},
            "test_case_description": "Coverage-oriented case for V-axis verification complexity.",
        }
    ]


def output_requirements_for_axis(axis: str, intensity: int) -> list[str]:
    requirements = {
        "D": ["Scaled executable cases must expose the new input/data complexity introduced by this operator."],
        "C": ["Scaled executable cases must expose the new computation and contract constraints introduced by this operator."],
        "A": ["Scaled executable cases must expose the new anti-shortcut robustness requirement introduced by this operator."],
        "V": ["Scaled executable cases must provide stronger executable verification coverage for this operator."],
    }
    return list(requirements[axis])


def gold_update_policy_for_axis(axis: str) -> dict[str, Any]:
    if axis == "V":
        return {"mode": "preserve_if_semantics_unchanged", "requires_case_alignment": True}
    return {"mode": "rewrite_executable_gold", "requires_case_alignment": True}


def rubric_delta_for_axis(axis: str, primary_domain: str, secondary_domain_names: list[str]) -> list[dict[str, Any]]:
    criteria = {
        "D": [{"criterion": "handles scaled input/data variants correctly", "category": "data"}],
        "C": [{"criterion": "satisfies scaled boundary and contract constraints", "category": "constraint"}],
        "A": [{"criterion": "resists shortcuts and brittle assumptions", "category": "robustness"}],
        "V": [{"criterion": "passes stronger executable oracle coverage", "category": "verification"}],
    }
    return criteria[axis]


def semantic_change_for_axis(axis: str) -> bool:
    return axis in {"D", "C", "A"}


def build_rationale(axis: str, primary_domain: str, secondary_domain_names: list[str]) -> str:
    secondary_text = ", ".join(secondary_domain_names) if secondary_domain_names else "none"
    return f"Axis {axis} was selected for domain={primary_domain} with secondary_domains={secondary_text}."


def _expected_failure_modes_for_axis(axis: str) -> list[str]:
    return {
        "D": ["original_solution_does_not_support_scaled_input_variant"],
        "C": ["original_solution_violates_new_constraint_or_return_contract"],
        "A": ["original_solution_uses_brittle_shortcut_or_hardcoded_assumption"],
        "V": ["solution_passes_seed_case_but_not_stronger_scaled_case_suite"],
    }[axis]


def _attach_semantic_generation_metadata(instance: DynamicOperatorInstance, task_id: str) -> None:
    specs = []
    for index, spec in enumerate(instance.semantic_test_specs or [], start=1):
        copied = dict(spec)
        copied.setdefault("targets_operator_id", instance.operator_id)
        copied.setdefault("axis", instance.axis)
        copied.setdefault("counts_as_hidden_test", instance.axis != "V")
        copied.setdefault("eligible_for_clean_export", True)
        copied.setdefault("test_tier", "semantic")
        copied.setdefault("materialization_status", "pending")
        copied.setdefault("source_operator_id", instance.operator_id)
        copied.setdefault("name", copied.get("spec_id") or f"{task_id}_{instance.axis.lower()}_{index}")
        specs.append(copied)
    instance.semantic_test_specs = specs


def _annotate_operator_verifier_delta(instance: DynamicOperatorInstance) -> None:
    annotated_checks = []
    for check in instance.verifier_delta.new_checks:
        copied = dict(check)
        copied.setdefault("source_operator_id", instance.operator_id)
        copied.setdefault("axis", instance.axis)
        annotated_checks.append(copied)
    instance.verifier_delta.new_checks = annotated_checks
    instance.output_constraint_spec = {"checks": []}


def _secondary_domain_names(secondary_domains: list[dict] | list) -> list[str]:
    names: list[str] = []
    for item in secondary_domains:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = str(item.get("domain") or "")
        else:
            name = str(getattr(item, "domain", "") or "")
        if name:
            names.append(name)
    return names


def _mock_operator_builder(context: dict[str, Any]) -> dict[str, Any]:
    instances = fallback_operator_instances(
        env_id=context["env_id"],
        task_id=context["task_id"],
        task_type=context["task_type"],
        domain=context["domain"],
        secondary_domains=context.get("secondary_domains", []),
        solution_form=context["solution_form"],
        scaling_plan=context["scaling_plan"],
        tool_config=context.get("tool_config", {}),
    )
    return {"operator_instances": [instance.model_dump() for instance in instances]}
