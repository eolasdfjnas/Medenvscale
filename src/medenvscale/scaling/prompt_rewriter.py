from __future__ import annotations

import re
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.schemas.scaling import DynamicOperatorInstance, ScalingPlan, ToolConfig


def rewrite_prompt(
    environment: ExecutableEnvSpec,
    scaling_plan: ScalingPlan,
    operator_instances: list[DynamicOperatorInstance],
    tool_config: ToolConfig | None = None,
) -> ExecutableEnvSpec:
    tool_section = ""
    if tool_config is not None:
        tools_block = []
        for index, tool in enumerate(tool_config.allowed_tools, start=1):
            tools_block.append(
                f"{index}. {tool.tool_name}({', '.join(tool.input_schema.keys())})\n"
                f"   {tool.description}\n"
                f"   Use when: {tool.when_to_use}"
            )
        budget_lines = [f"- max total tool calls: {tool_config.tool_budget.max_total_tool_calls}"]
        for tool_name, count in tool_config.tool_budget.max_calls_per_tool.items():
            budget_lines.append(f"- {tool_name}: {count}")
        tool_section = (
            f"Only the following tools may be used:\n{chr(10).join(tools_block) or 'None'}\n\n"
            f"Tool budget:\n{chr(10).join(budget_lines)}\n\n"
        )
    semantic_requirements = _collect_semantic_requirements(environment, operator_instances)
    semantic_block = ""
    if semantic_requirements:
        semantic_block = "Additional requirements:\n" + "\n".join(f"- {item}" for item in semantic_requirements) + "\n\n"
    user_prompt = (
        f"Task ID: {environment.env_id}\n"
        f"Mode: evaluation\n\n"
        f"{tool_section}"
        f"{semantic_block}"
        f"Output requirement:\n- return only the code that should replace {environment.visible_state.get('placeholder_token', '<<insert solution here>>')}\n\n"
        f"Problem:\n{environment.problem}\n\n"
        f"Context:\n{environment.context}"
    )
    prompt_format = "agent_loop" if scaling_plan.allow_multiturn else "single_turn"
    return environment.model_copy(
        update={
            "system_prompt": "You are a careful biomedical/scientific coding agent.",
            "user_prompt": user_prompt,
            "prompt_format": prompt_format,
        }
    )


def _collect_semantic_requirements(
    environment: ExecutableEnvSpec,
    operator_instances: list[DynamicOperatorInstance] | list[dict[str, Any]] | None = None,
) -> list[str]:
    visible_state = environment.visible_state or {}
    task_state = environment.task_state or {}
    items: list[str] = []
    for key in [
        "constraint_hints",
        "implicit_requirements",
        "implicit_clues",
        "execution_requirements",
        "stepwise_requirements",
        "resource_complexity_notes",
        "robustness_challenges",
        "must_not_assume",
        "output_constraints",
        "format_constraints",
    ]:
        value = visible_state.get(key) or []
        if isinstance(value, list):
            items.extend(str(item) for item in value if str(item).strip())
        elif value:
            items.append(str(value))
    for key in ["extra_constraints", "execution_steps", "required_steps", "implicit_requirements", "safety_critical_constraints"]:
        value = task_state.get(key) or []
        if isinstance(value, list):
            items.extend(str(item) for item in value if str(item).strip())
    trap = visible_state.get("robustness_trap")
    if trap:
        items.append(str(trap))
    operator_requirements = _collect_operator_case_requirements(environment, operator_instances or [])
    if operator_requirements:
        items = [item for item in items if not _is_generic_requirement(item)]
        items.extend(operator_requirements)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _dedupe_key(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _collect_operator_case_requirements(
    environment: ExecutableEnvSpec,
    operator_instances: list[DynamicOperatorInstance] | list[dict[str, Any]],
) -> list[str]:
    cases = environment.validated_oracle_cases or environment.scaled_oracle_cases or []
    requirements: list[str] = []
    for operator in operator_instances:
        op = _operator_dict(operator)
        op_id = str(op.get("operator_id") or "").strip()
        axis = str(op.get("axis") or "").strip()
        linked_cases = _linked_cases(op_id, axis, cases)
        for case in linked_cases:
            if not isinstance(case, dict):
                continue
            case_requirements: list[str] = []
            for key in ["target_constraint", "semantic_intent", "description"]:
                requirement = _normalize_requirement(case.get(key))
                if requirement and not _is_generic_requirement(requirement) and _has_specific_content(requirement):
                    case_requirements.append(requirement)
                if case_requirements and key in {"target_constraint", "semantic_intent"}:
                    continue
            requirements.extend(case_requirements[:2])
            for item in case.get("covered_requirements") or case.get("covers_requirements") or []:
                requirement = _normalize_requirement(item)
                if requirement:
                    requirements.append(requirement)
        for item in [
            op.get("transformation_goal"),
            *(op.get("output_requirements") or []),
        ]:
            requirement = _normalize_requirement(item)
            if requirement:
                requirements.append(requirement)
    return [item for item in requirements if not _is_generic_requirement(item) and _has_specific_content(item)]


def _operator_dict(operator: DynamicOperatorInstance | dict[str, Any]) -> dict[str, Any]:
    if isinstance(operator, dict):
        return operator
    return operator.model_dump()


def _linked_cases(op_id: str, axis: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linked: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        targets = str(case.get("targets_operator_id") or "").strip()
        case_axis = str(case.get("axis") or "").strip()
        if targets and any(token.strip() == op_id for token in targets.split(",")):
            linked.append(case)
        elif not targets and case_axis == axis:
            linked.append(case)
    return linked


def _normalize_requirement(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^verify that\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^verify\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^increase data complexity by requiring\s+", "Require ", text, flags=re.IGNORECASE)
    text = re.sub(r"^increase data complexity by adding\s+", "Add ", text, flags=re.IGNORECASE)
    text = re.sub(r"^increase data complexity by introducing\s+", "Introduce ", text, flags=re.IGNORECASE)
    text = re.sub(r"^scaled executable cases? must (verify|check|include|expose)\s+", "Handle ", text, flags=re.IGNORECASE)
    text = re.sub(r"^the solution must\s+", "The solution must ", text, flags=re.IGNORECASE)
    if len(text) > 320:
        text = text[:317].rstrip() + "..."
    return text


def _is_generic_requirement(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    generic_patterns = [
        "handle scaled input/data variants from the oracle cases",
        "scaled executable cases must expose",
        "scaled cases add explicit edge-case",
        "scaled cases introduce stronger computation",
        "solution must satisfy new boundary",
        "respect new parameter, boundary, ordering, or return-contract constraints from scaled cases",
        "preserve the required return structure and edge-case behavior",
        "handle scaled input",
        "increase computation and contract complexity through parameter rules",
        "new requirement introduced by this operator",
        "new computation and contract constraints introduced by this operator",
    ]
    return any(pattern in text for pattern in generic_patterns)


def _has_specific_content(value: Any) -> bool:
    tokens = _specific_tokens(value)
    return len(tokens) >= 2


def _specific_tokens(value: Any) -> set[str]:
    generic_tokens = {
        "additional",
        "behavior",
        "boundary",
        "case",
        "cases",
        "computation",
        "constraint",
        "constraints",
        "contract",
        "correctly",
        "data",
        "edge",
        "edge-case",
        "executable",
        "expected",
        "explicit",
        "expose",
        "file",
        "function",
        "gold",
        "handle",
        "input",
        "introduce",
        "introduced",
        "must",
        "new",
        "operator",
        "oracle",
        "ordering",
        "output",
        "output-contract",
        "parameter",
        "parameters",
        "preserve",
        "requirement",
        "requirements",
        "required",
        "respect",
        "return",
        "return-contract",
        "rules",
        "scaled",
        "solution",
        "stronger",
        "structure",
        "task",
        "test",
        "tests",
        "testing",
        "value",
        "values",
        "variant",
        "variants",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(value or "").lower())
        if token not in generic_tokens
    }


def _dedupe_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
