from __future__ import annotations

import json
from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.schemas.scaling import OutputRequirement, ToolBudget, ToolConfig, ToolSpec


def build_tool_config(
    env_id: str,
    global_level: str,
    task_type: str,
    primary_domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    resource_manifest: list[str],
    required_capabilities: list[str] | None,
    scaling_plan: dict,
    budgets_cfg: dict,
    tool_pool_cfg: dict | None = None,
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
    seed_task: dict[str, Any] | None = None,
    base_environment: ExecutableEnvSpec | None = None,
) -> ToolConfig:
    if llm_client is not None and prompt_runner is not None:
        try:
            prompt = prompt_runner.render(
                "tool_config_planner.jinja",
                seed_task=json.dumps(seed_task or {}, ensure_ascii=False, indent=2),
                base_environment=json.dumps((base_environment.model_dump() if base_environment else {}), ensure_ascii=False, indent=2),
                domain=primary_domain,
                secondary_domains=json.dumps(_secondary_domain_names(secondary_domains), ensure_ascii=False),
                task_type=task_type,
                solution_form=solution_form,
                resource_manifest=json.dumps(resource_manifest, ensure_ascii=False),
                scaling_plan=json.dumps(scaling_plan, ensure_ascii=False, indent=2),
                tool_budget_bounds=json.dumps(budgets_cfg["tool_budget_bounds"][global_level], ensure_ascii=False, indent=2),
                candidate_agent_tools=json.dumps([tool.model_dump() for tool in available_tool_catalog(tool_pool_cfg)], ensure_ascii=False, indent=2),
                evaluator_only_tools=json.dumps([tool.model_dump() for tool in evaluator_tool_catalog(tool_pool_cfg)], ensure_ascii=False, indent=2),
            )
            response = llm_client.complete_json(
                task_name="tool_config_planner",
                prompt=prompt,
                context={
                    "env_id": env_id,
                    "global_level": global_level,
                    "task_type": task_type,
                    "primary_domain": primary_domain,
                    "secondary_domains": _secondary_domain_names(secondary_domains),
                    "solution_form": solution_form,
                    "resource_manifest": resource_manifest,
                    "required_capabilities": required_capabilities or [],
                    "scaling_plan": scaling_plan,
                    "tool_pool_cfg": tool_pool_cfg,
                },
                mock_builder=_mock_tool_config_builder,
            )
            repaired = repair_tool_config_payload(
                payload=response.payload,
                env_id=env_id,
                global_level=global_level,
                task_type=task_type,
                primary_domain=primary_domain,
                secondary_domains=secondary_domains,
                solution_form=solution_form,
                resource_manifest=resource_manifest,
                required_capabilities=required_capabilities or [],
                scaling_plan=scaling_plan,
                budgets_cfg=budgets_cfg,
                tool_pool_cfg=tool_pool_cfg,
            )
            validate_tool_config(
                repaired,
                budgets_cfg,
                tool_pool_cfg,
                resource_manifest,
                scaling_plan,
                primary_domain,
                _secondary_domain_names(secondary_domains),
            )
            return repaired
        except Exception:
            pass

    fallback = fallback_tool_config(
        env_id=env_id,
        global_level=global_level,
        task_type=task_type,
        primary_domain=primary_domain,
        secondary_domains=secondary_domains,
        solution_form=solution_form,
        resource_manifest=resource_manifest,
        required_capabilities=required_capabilities or [],
        scaling_plan=scaling_plan,
        budgets_cfg=budgets_cfg,
        tool_pool_cfg=tool_pool_cfg,
        planning_source="fallback",
        validation_trace=["fallback_tool_config"],
    )
    validate_tool_config(
        fallback,
        budgets_cfg,
        tool_pool_cfg,
        resource_manifest,
        scaling_plan,
        primary_domain,
        _secondary_domain_names(secondary_domains),
    )
    return fallback


def repair_tool_config_payload(
    payload: dict[str, Any],
    env_id: str,
    global_level: str,
    task_type: str,
    primary_domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    resource_manifest: list[str],
    required_capabilities: list[str],
    scaling_plan: dict,
    budgets_cfg: dict,
    tool_pool_cfg: dict | None,
) -> ToolConfig:
    trace: list[str] = []
    fallback = fallback_tool_config(
        env_id=env_id,
        global_level=global_level,
        task_type=task_type,
        primary_domain=primary_domain,
        secondary_domains=secondary_domains,
        solution_form=solution_form,
        resource_manifest=resource_manifest,
        required_capabilities=required_capabilities,
        scaling_plan=scaling_plan,
        budgets_cfg=budgets_cfg,
        tool_pool_cfg=tool_pool_cfg,
        planning_source="fallback",
        validation_trace=["repair_tool_config_fallback_seed"],
    )

    raw_tools = payload.get("allowed_tools", []) or []
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    registry = agent_tool_registry(tool_pool_cfg)
    allowed_tools = [_coerce_tool_spec(item, trace, registry) for item in raw_tools]
    allowed_tools = deduplicate_tools([tool for tool in allowed_tools if tool is not None])

    forced_tools = required_tools_for_context(
        task_type=task_type,
        primary_domain=primary_domain,
        secondary_domains=secondary_domains,
        solution_form=solution_form,
        resource_manifest=resource_manifest,
        required_capabilities=required_capabilities,
        scaling_plan=scaling_plan,
        tool_pool_cfg=tool_pool_cfg,
    )
    allowed_by_name = {tool.tool_name: tool for tool in allowed_tools}
    for tool in forced_tools:
        if tool.tool_name not in allowed_by_name:
            allowed_tools.append(tool)
            allowed_by_name[tool.tool_name] = tool
            trace.append(f"added_required_tool:{tool.tool_name}")

    bounds = budgets_cfg["tool_budget_bounds"][global_level]
    lower, upper = bounds["allowed_tool_count_range"]
    allowed_tools = clamp_tools_with_priority(
        allowed_tools,
        upper=upper,
        preserve_names=hard_required_tool_names(primary_domain, scaling_plan, resource_manifest),
    )
    if len(allowed_tools) < lower:
        for tool in fallback.allowed_tools:
            if tool.tool_name not in {item.tool_name for item in allowed_tools}:
                allowed_tools.append(tool)
                trace.append(f"filled_tool_from_fallback:{tool.tool_name}")
            if len(allowed_tools) >= lower:
                break
    allowed_tools = deduplicate_tools(allowed_tools)

    tool_budget = _repair_tool_budget(payload.get("tool_budget") or {}, allowed_tools, bounds, trace)
    output_requirement = _repair_output_requirement(
        payload.get("output_requirement") or {},
        task_type=task_type,
        solution_form=solution_form,
        primary_domain=primary_domain,
        secondary_domains=secondary_domains,
        scaling_plan=scaling_plan,
        trace=trace,
    )
    if scaling_plan["axis_intensity"].get("C", 0) >= 2 and "constraint_checked_output" not in output_requirement.required_fields:
        output_requirement.required_fields.append("constraint_checked_output")
        trace.append("added_constraint_checked_output_field")

    planning_source = "repaired" if trace else "llm"
    return ToolConfig(
        env_id=env_id,
        global_level=global_level,
        planning_source=planning_source,
        allowed_tools=allowed_tools,
        tool_budget=tool_budget,
        output_requirement=output_requirement,
        tool_choice_reason=str(payload.get("tool_choice_reason") or build_tool_choice_reason(primary_domain, _secondary_domain_names(secondary_domains), task_type)),
        budget_reason=str(payload.get("budget_reason") or "Budget was repaired to remain within the configured M-level bounds."),
        related_axes=list(scaling_plan["selected_axes"]),
        validation_trace=trace,
    )


def fallback_tool_config(
    env_id: str,
    global_level: str,
    task_type: str,
    primary_domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    resource_manifest: list[str],
    required_capabilities: list[str],
    scaling_plan: dict,
    budgets_cfg: dict,
    tool_pool_cfg: dict | None,
    planning_source: str,
    validation_trace: list[str],
) -> ToolConfig:
    tools = required_tools_for_context(
        task_type=task_type,
        primary_domain=primary_domain,
        secondary_domains=secondary_domains,
        solution_form=solution_form,
        resource_manifest=resource_manifest,
        required_capabilities=required_capabilities,
        scaling_plan=scaling_plan,
        tool_pool_cfg=tool_pool_cfg,
    )
    bounds = budgets_cfg["tool_budget_bounds"][global_level]
    min_tools, max_tools = bounds["allowed_tool_count_range"]
    tools = clamp_tools_with_priority(
        deduplicate_tools(tools),
        upper=max_tools,
        preserve_names=hard_required_tool_names(primary_domain, scaling_plan, resource_manifest),
    )
    if len(tools) < min_tools:
        for tool in fallback_seed_tools(tool_pool_cfg):
            if tool.tool_name not in {item.tool_name for item in tools}:
                tools.append(tool)
            if len(tools) >= min_tools:
                break
    tools = deduplicate_tools(tools)

    allowed_names = [tool.tool_name for tool in tools]
    max_total_low, max_total_high = bounds["max_total_tool_calls_range"]
    max_total = min(max_total_high, max(max_total_low, len(tools) + (2 if global_level in {"M3", "M4"} else 1)))
    max_per_tool = {name: min(bounds["max_calls_per_tool_upper"], 2 if name == "debug_traceback" else 3) for name in allowed_names}
    budget = ToolBudget(
        max_total_tool_calls=max_total,
        max_calls_per_tool=max_per_tool,
        max_consecutive_calls_per_tool={"run_custom_test": 2} if "run_custom_test" in allowed_names else {},
        max_debug_calls=min(bounds.get("max_debug_calls_upper", 0), max_per_tool.get("debug_traceback", 0)),
        max_validation_calls=_max_validation_budget(max_per_tool),
    )

    output_requirement = OutputRequirement(output_format="code", strict=True)
    if primary_domain == "omics_measurement_analysis" or "omics_measurement_analysis" in _secondary_domain_names(secondary_domains):
        output_requirement.required_fields = ["domain_grounded_handling"]
    if scaling_plan["axis_intensity"].get("C", 0) >= 2:
        output_requirement.required_fields.append("constraint_checked_output")

    return ToolConfig(
        env_id=env_id,
        global_level=global_level,
        planning_source=planning_source,
        allowed_tools=tools,
        tool_budget=budget,
        output_requirement=output_requirement,
        tool_choice_reason=build_tool_choice_reason(primary_domain, _secondary_domain_names(secondary_domains), task_type),
        budget_reason="Budget is clamped within the configured M-level bounds.",
        related_axes=scaling_plan["selected_axes"],
        validation_trace=validation_trace,
    )


def required_tools_for_context(
    task_type: str,
    primary_domain: str,
    secondary_domains: list[dict] | list,
    solution_form: str,
    resource_manifest: list[str],
    required_capabilities: list[str],
    scaling_plan: dict,
    tool_pool_cfg: dict | None,
) -> list[ToolSpec]:
    registry = agent_tool_registry(tool_pool_cfg)
    tool_names: list[str] = []
    selected_axes = scaling_plan["selected_axes"]
    axis_intensity = scaling_plan["axis_intensity"]
    signature_like = solution_form in {"function_definition", "function_body", "statement_block_completion", "patch_or_bugfix"}

    tool_names.extend(["get_task_context", "validate_candidate_code", "submit_final_code"])
    if resource_manifest or any("file" in capability or "import" in capability for capability in required_capabilities):
        tool_names.extend(["create_test_file", "run_custom_test"])
    if task_type in {"code_validation_and_utility", "io_format_and_cli", "structured_data_processing"}:
        tool_names.append("get_task_context")
    if task_type == "numerical_computation":
        tool_names.append("run_custom_test")
    if axis_intensity.get("V", 0) >= 1 or axis_intensity.get("C", 0) >= 1 or signature_like:
        tool_names.extend(["validate_candidate_code", "run_custom_test"])
    if axis_intensity.get("C", 0) >= 2 or axis_intensity.get("A", 0) >= 1:
        tool_names.append("run_custom_test")
    if axis_intensity.get("D", 0) >= 2:
        tool_names.extend(["get_task_context", "run_custom_test"])
    return deduplicate_tools(_materialize_tool_names(tool_names, registry))


def validate_tool_config(
    config: ToolConfig,
    budgets_cfg: dict,
    tool_pool_cfg: dict | None,
    resource_manifest: list[str],
    scaling_plan: dict,
    primary_domain: str,
    secondary_domain_names: list[str],
) -> None:
    bounds = budgets_cfg["tool_budget_bounds"][config.global_level]
    lower, upper = bounds["allowed_tool_count_range"]
    count = len(config.allowed_tools)
    if not (lower <= count <= upper):
        raise ValueError(f"{config.env_id}: allowed_tools count violates {config.global_level} bounds")
    low_total, high_total = bounds["max_total_tool_calls_range"]
    if not (low_total <= config.tool_budget.max_total_tool_calls <= high_total):
        raise ValueError(f"{config.env_id}: max_total_tool_calls violates {config.global_level} bounds")
    allowed_names = {tool.tool_name for tool in config.allowed_tools}
    agent_names = set(agent_tool_registry(tool_pool_cfg))
    evaluator_names = set(evaluator_tool_registry(tool_pool_cfg))
    if not allowed_names.issubset(agent_names):
        raise ValueError(f"{config.env_id}: allowed_tools must come from the dataset agent tool pool")
    if allowed_names.intersection(evaluator_names):
        raise ValueError(f"{config.env_id}: evaluator-only tools leaked into allowed_tools")
    for tool in config.allowed_tools:
        if not tool.description or not tool.input_schema or not tool.when_to_use:
            raise ValueError(f"{config.env_id}: malformed tool spec for {tool.tool_name}")
    for tool_name, count in config.tool_budget.max_calls_per_tool.items():
        if tool_name not in allowed_names:
            raise ValueError(f"{config.env_id}: max_calls_per_tool includes non-allowed tool {tool_name}")
        if count > bounds["max_calls_per_tool_upper"]:
            raise ValueError(f"{config.env_id}: tool {tool_name} exceeds per-tool budget")
    if not bounds["allow_debug_tool"] and "debug_traceback" in allowed_names:
        raise ValueError(f"{config.env_id}: debug_traceback not allowed for {config.global_level}")
    if resource_manifest and scaling_plan["axis_intensity"].get("D", 0) >= 2:
        if "run_custom_test" not in allowed_names:
            raise ValueError(f"{config.env_id}: resource-heavy task requires a public runtime test tool")
    if scaling_plan["axis_intensity"].get("V", 0) >= 2 and "run_custom_test" not in allowed_names:
        raise ValueError(f"{config.env_id}: V-intensive task requires run_custom_test")
    if scaling_plan["axis_intensity"].get("C", 0) >= 2:
        has_constraint_checker = "run_custom_test" in allowed_names or "constraint_checked_output" in config.output_requirement.required_fields
        if not has_constraint_checker:
            raise ValueError(f"{config.env_id}: C-intensive task requires constraint-aware output or validation")
    if config.output_requirement.output_format != "code":
        raise ValueError(f"{config.env_id}: output format must remain code for code-completion tasks")


def _repair_tool_budget(raw_budget: dict[str, Any], allowed_tools: list[ToolSpec], bounds: dict[str, Any], trace: list[str]) -> ToolBudget:
    allowed_names = [tool.tool_name for tool in allowed_tools]
    low_total, high_total = bounds["max_total_tool_calls_range"]
    try:
        max_total = int(raw_budget.get("max_total_tool_calls", low_total))
    except (TypeError, ValueError):
        max_total = low_total
        trace.append("repaired_max_total_tool_calls")
    max_total = min(high_total, max(low_total, max_total))

    raw_max_calls = raw_budget.get("max_calls_per_tool", {}) or {}
    max_calls_per_tool: dict[str, int] = {}
    for name in allowed_names:
        raw_value = raw_max_calls.get(name, 2 if name == "debug_traceback" else 3)
        try:
            count = int(raw_value)
        except (TypeError, ValueError):
            count = 1
            trace.append(f"repaired_max_calls_per_tool:{name}")
        count = max(1, min(bounds["max_calls_per_tool_upper"], count))
        max_calls_per_tool[name] = count

    raw_consecutive = raw_budget.get("max_consecutive_calls_per_tool", {}) or {}
    max_consecutive = {}
    for name, value in raw_consecutive.items():
        if name not in allowed_names:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        max_consecutive[name] = max(1, min(bounds["max_calls_per_tool_upper"], parsed))

    try:
        max_debug_calls = int(raw_budget.get("max_debug_calls", 0))
    except (TypeError, ValueError):
        max_debug_calls = 0
        trace.append("repaired_max_debug_calls")
    try:
        max_validation_calls = int(raw_budget.get("max_validation_calls", _max_validation_budget(max_calls_per_tool)))
    except (TypeError, ValueError):
        max_validation_calls = _max_validation_budget(max_calls_per_tool)
        trace.append("repaired_max_validation_calls")

    return ToolBudget(
        max_total_tool_calls=max_total,
        max_calls_per_tool=max_calls_per_tool,
        max_consecutive_calls_per_tool=max_consecutive,
        max_debug_calls=min(bounds.get("max_debug_calls_upper", 0), max_debug_calls, max_calls_per_tool.get("debug_traceback", 0)),
        max_validation_calls=min(max_validation_calls, _max_validation_budget(max_calls_per_tool)),
    )


def _repair_output_requirement(
    raw_requirement: dict[str, Any],
    task_type: str,
    solution_form: str,
    primary_domain: str,
    secondary_domains: list[dict] | list,
    scaling_plan: dict,
    trace: list[str],
) -> OutputRequirement:
    output_format = str(raw_requirement.get("output_format") or "code").strip().lower()
    if output_format not in {"text", "json", "code", "file", "table"}:
        output_format = "code"
        trace.append("repaired_output_format")
    required_fields = [str(item).strip() for item in raw_requirement.get("required_fields", []) or [] if str(item).strip()]
    forbidden_fields = [str(item).strip() for item in raw_requirement.get("forbidden_fields", []) or [] if str(item).strip()]
    strict = bool(raw_requirement.get("strict", True))
    if primary_domain == "omics_measurement_analysis" or "omics_measurement_analysis" in _secondary_domain_names(secondary_domains):
        if "domain_grounded_handling" not in required_fields:
            required_fields.append("domain_grounded_handling")
    return OutputRequirement(
        output_format="code" if task_type or solution_form else output_format,
        required_fields=required_fields,
        forbidden_fields=forbidden_fields,
        json_schema=raw_requirement.get("json_schema"),
        strict=strict,
    )


def _coerce_tool_spec(raw_tool: Any, trace: list[str], registry: dict[str, ToolSpec]) -> ToolSpec | None:
    if isinstance(raw_tool, str):
        canonical = registry.get(raw_tool)
        if canonical is None:
            trace.append(f"dropped_unknown_tool:{raw_tool}")
            return None
        trace.append(f"expanded_tool_name:{raw_tool}")
        return canonical
    if not isinstance(raw_tool, dict):
        trace.append("dropped_non_object_tool_spec")
        return None
    tool_name = str(raw_tool.get("tool_name") or "").strip()
    canonical = registry.get(tool_name)
    if canonical is None:
        trace.append(f"dropped_unknown_tool:{tool_name}")
        return None
    merged = canonical.model_dump()
    for key in ["description", "input_schema", "output_schema", "when_to_use", "limitations", "examples"]:
        if key in raw_tool and raw_tool[key]:
            merged[key] = raw_tool[key]
    try:
        return ToolSpec.model_validate({"tool_name": tool_name, **merged})
    except Exception:
        trace.append(f"tool_spec_validation_failed:{tool_name}")
        return canonical


def _secondary_domain_names(secondary_domains: list[dict] | list) -> list[str]:
    return [item["domain"] if isinstance(item, dict) else getattr(item, "domain", "") for item in secondary_domains if (item["domain"] if isinstance(item, dict) else getattr(item, "domain", ""))]


def hard_required_tool_names(primary_domain: str, scaling_plan: dict, resource_manifest: list[str]) -> set[str]:
    required: set[str] = set()
    required.add("get_task_context")
    required.add("validate_candidate_code")
    required.add("submit_final_code")
    if resource_manifest:
        required.add("create_test_file")
        required.add("run_custom_test")
    if scaling_plan["axis_intensity"].get("V", 0) >= 2:
        required.add("run_custom_test")
    if scaling_plan["axis_intensity"].get("C", 0) >= 2:
        required.add("run_custom_test")
    return required


def clamp_tools_with_priority(tools: list[ToolSpec], upper: int, preserve_names: set[str]) -> list[ToolSpec]:
    if len(tools) <= upper:
        return tools
    preserved = [tool for tool in tools if tool.tool_name in preserve_names]
    others = [tool for tool in tools if tool.tool_name not in preserve_names]
    clamped = preserved[:upper]
    if len(clamped) < upper:
        clamped.extend(others[: upper - len(clamped)])
    return deduplicate_tools(clamped)


def global_debug_allowed(global_level: str | None, selected_axes: list[str], axis_intensity: dict[str, int]) -> bool:
    return global_level in {"M3", "M4"} or ("C" in selected_axes and axis_intensity.get("C", 0) >= 2)


def deduplicate_tools(tools: list[ToolSpec]) -> list[ToolSpec]:
    deduped: list[ToolSpec] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.tool_name in seen:
            continue
        deduped.append(tool)
        seen.add(tool.tool_name)
    return deduped


def available_tool_catalog(tool_pool_cfg: dict | None = None) -> list[ToolSpec]:
    return list(agent_tool_registry(tool_pool_cfg).values())


def evaluator_tool_catalog(tool_pool_cfg: dict | None = None) -> list[ToolSpec]:
    return list(evaluator_tool_registry(tool_pool_cfg).values())


def tool_spec_registry(tool_pool_cfg: dict | None = None) -> dict[str, ToolSpec]:
    registry = {}
    registry.update(agent_tool_registry(tool_pool_cfg))
    registry.update(evaluator_tool_registry(tool_pool_cfg))
    return registry


def agent_tool_registry(tool_pool_cfg: dict | None = None) -> dict[str, ToolSpec]:
    return _tool_registry_from_section(tool_pool_cfg, "agent_tools")


def evaluator_tool_registry(tool_pool_cfg: dict | None = None) -> dict[str, ToolSpec]:
    return _tool_registry_from_section(tool_pool_cfg, "evaluator_tools")


def _tool_registry_from_section(tool_pool_cfg: dict | None, section: str) -> dict[str, ToolSpec]:
    rows = (tool_pool_cfg or {}).get(section, [])
    registry: dict[str, ToolSpec] = {}
    for row in rows:
        tool = ToolSpec.model_validate(row)
        registry[tool.tool_name] = tool
    return registry


def _materialize_tool_names(tool_names: list[str], registry: dict[str, ToolSpec]) -> list[ToolSpec]:
    return [registry[name] for name in tool_names if name in registry]


def fallback_seed_tools(tool_pool_cfg: dict | None) -> list[ToolSpec]:
    return _materialize_tool_names(
        ["get_task_context", "validate_candidate_code", "run_custom_test", "submit_final_code"],
        agent_tool_registry(tool_pool_cfg),
    )


def _max_validation_budget(max_calls_per_tool: dict[str, int]) -> int:
    validation_names = {"validate_candidate_code", "run_custom_test", "submit_final_code"}
    return max([max_calls_per_tool.get(name, 0) for name in validation_names] or [0])


def build_tool_choice_reason(primary_domain: str, secondary_domain_names: list[str], task_type: str) -> str:
    secondary_text = ", ".join(secondary_domain_names) if secondary_domain_names else "none"
    return (
        f"Tool choices follow task_type={task_type}, primary_domain={primary_domain}, "
        f"and secondary_domains={secondary_text}, while remaining within M-level bounds."
    )


def _mock_tool_config_builder(context: dict[str, Any]) -> dict[str, Any]:
    scaling_plan = context["scaling_plan"]
    tools = required_tools_for_context(
        task_type=context["task_type"],
        primary_domain=context["primary_domain"],
        secondary_domains=[{"domain": name} for name in context.get("secondary_domains", [])],
        solution_form=context["solution_form"],
        resource_manifest=context.get("resource_manifest", []),
        required_capabilities=context.get("required_capabilities", []),
        scaling_plan=scaling_plan,
        tool_pool_cfg=context.get("tool_pool_cfg"),
    )
    bounds = {
        "M1": {"allowed_tool_count_range": [0, 2], "max_total_tool_calls_range": [0, 2], "max_calls_per_tool_upper": 2, "max_debug_calls_upper": 0},
        "M2": {"allowed_tool_count_range": [1, 3], "max_total_tool_calls_range": [2, 5], "max_calls_per_tool_upper": 3, "max_debug_calls_upper": 0},
        "M3": {"allowed_tool_count_range": [2, 5], "max_total_tool_calls_range": [4, 9], "max_calls_per_tool_upper": 4, "max_debug_calls_upper": 2},
        "M4": {"allowed_tool_count_range": [3, 7], "max_total_tool_calls_range": [7, 14], "max_calls_per_tool_upper": 6, "max_debug_calls_upper": 4},
    }[context["global_level"]]
    min_tools, max_tools = bounds["allowed_tool_count_range"]
    tools = deduplicate_tools(tools)[:max_tools]
    if len(tools) < min_tools:
        tools = deduplicate_tools((tools + fallback_seed_tools(context.get("tool_pool_cfg")))[:min_tools])
    allowed_names = [tool.tool_name for tool in tools]
    low_total, high_total = bounds["max_total_tool_calls_range"]
    max_total = min(high_total, max(low_total, len(tools) + (2 if context["global_level"] in {"M3", "M4"} else 1)))
    max_calls = {name: min(bounds["max_calls_per_tool_upper"], 2 if name == "debug_traceback" else 3) for name in allowed_names}
    return {
        "allowed_tools": [tool.model_dump() for tool in tools],
        "tool_budget": {
            "max_total_tool_calls": max_total,
            "max_calls_per_tool": max_calls,
            "max_consecutive_calls_per_tool": {"run_custom_test": 2} if "run_custom_test" in allowed_names else {},
            "max_debug_calls": min(bounds["max_debug_calls_upper"], max_calls.get("debug_traceback", 0)),
            "max_validation_calls": _max_validation_budget(max_calls),
        },
        "output_requirement": {"output_format": "code", "required_fields": [], "forbidden_fields": [], "json_schema": None, "strict": True},
        "tool_choice_reason": build_tool_choice_reason(context["primary_domain"], context.get("secondary_domains", []), context["task_type"]),
        "budget_reason": f"{context['global_level']} allows the selected inspection and validation tools without exceeding bounds.",
    }
