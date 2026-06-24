from __future__ import annotations

from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas.scaling import AXES, AxisWeightPlannerResult, SecondaryAxisWeightHint


def fallback_rank_weights(axis_priority: list[str]) -> dict[str, int]:
    rank_to_weight = {1: 7, 2: 5, 3: 3, 4: 2}
    weights = {axis: 1 for axis in AXES}
    for index, axis in enumerate(axis_priority, start=1):
        weights[axis] = rank_to_weight.get(index, 1)
    return weights


def plan_axis_weights(
    task_type: str,
    secondary_task_types: list[str],
    task_axis_priority_cfg: dict,
    problem: str,
    context_summary: str,
    signature: str | None,
    verifier_type_hint: str | None,
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
    domain: str | None = None,
    solution_form: str | None = None,
) -> tuple[AxisWeightPlannerResult, str, list[str]]:
    trace: list[str] = []
    context = {
        "task_type": task_type,
        "secondary_task_types": list(secondary_task_types[:3]),
        "task_axis_priority": list(task_axis_priority_cfg["task_axis_priority"][task_type]["axis_priority"]),
        "secondary_task_axis_priorities": {
            secondary: list(task_axis_priority_cfg["task_axis_priority"].get(secondary, {}).get("axis_priority", []))
            for secondary in secondary_task_types[:3]
        },
        "problem": problem,
        "context_summary": context_summary,
        "signature": signature or "",
        "verifier_type_hint": verifier_type_hint,
        "domain": domain or "",
        "solution_form": solution_form or "",
    }

    if llm_client is not None and prompt_runner is not None:
        try:
            prompt = prompt_runner.render(
                "axis_weight_planner_4axis.jinja",
                problem=problem,
                context_summary=context_summary,
                signature=signature or "",
                verifier_type_hint=verifier_type_hint or "",
                domain=domain or "",
                task_type=task_type,
                solution_form=solution_form or "",
                task_axis_priority=context["task_axis_priority"],
                secondary_task_types=context["secondary_task_types"],
                secondary_task_axis_priorities=context["secondary_task_axis_priorities"],
            )
            response = llm_client.complete_json(
                task_name="axis_weight_planner_4axis",
                prompt=prompt,
                context=context,
                mock_builder=_mock_axis_weight_builder,
            )
            repaired, repaired_trace = repair_axis_weight_payload(
                response.payload,
                task_type=task_type,
                secondary_task_types=secondary_task_types,
                task_axis_priority_cfg=task_axis_priority_cfg,
            )
            trace.extend(repaired_trace)
            source = "repaired" if repaired_trace else "llm"
            return repaired, source, trace
        except Exception as exc:
            trace.append(f"llm_axis_weight_planner_failed: {exc}")

    trace.append("fallback_axis_weight_planner")
    return fallback_axis_weight_result(
        task_type=task_type,
        secondary_task_types=secondary_task_types,
        task_axis_priority_cfg=task_axis_priority_cfg,
        problem=problem,
        context_summary=context_summary,
        signature=signature,
        verifier_type_hint=verifier_type_hint,
    ), "fallback", trace


def fallback_axis_weight_result(
    task_type: str,
    secondary_task_types: list[str],
    task_axis_priority_cfg: dict,
    problem: str,
    context_summary: str,
    signature: str | None,
    verifier_type_hint: str | None,
) -> AxisWeightPlannerResult:
    primary_cfg = task_axis_priority_cfg["task_axis_priority"][task_type]
    primary = fallback_rank_weights(primary_cfg["axis_priority"])
    text = f"{problem}\n{context_summary}\n{signature or ''}".lower()

    if any(token in text for token in ["csv", "tsv", "json", "yaml", "fasta", "fastq", "path", "file", "directory", "column"]):
        primary["D"] = min(7, primary["D"] + 1)
    if any(token in text for token in ["raise", "exception", "constraint", "sort", "order", "boundary", "empty", "none", "duplicate", "parameter"]):
        primary["C"] = min(7, primary["C"] + 1)
    if any(token in text for token in ["hardcode", "shortcut", "random", "duplicate", "conflict", "misleading", "robust"]):
        primary["A"] = min(7, primary["A"] + 1)
    if verifier_type_hint in {"file_output", "object_state_check", "unit_test"} or any(
        token in text for token in ["stdout", "stderr", "artifact", "oracle", "verify", "assert", "log"]
    ):
        primary["V"] = min(7, primary["V"] + 1)

    secondary_hints: list[SecondaryAxisWeightHint] = []
    for secondary in secondary_task_types[:3]:
        secondary_cfg = task_axis_priority_cfg["task_axis_priority"].get(secondary)
        if secondary_cfg is None:
            continue
        weights = fallback_rank_weights(secondary_cfg["axis_priority"])
        relevance = _secondary_relevance(secondary, text)
        secondary_hints.append(
            SecondaryAxisWeightHint(
                task_type=secondary,
                relevance=relevance,
                axis_weight_hint=weights,
                reason=f"Fallback judged {secondary} relevant to the sample context.",
            )
        )

    return AxisWeightPlannerResult(
        primary_axis_weight_hint=primary,
        secondary_axis_weight_hints=secondary_hints,
        axis_weight_reason="Fallback weights are driven by task type and sample-level heuristics.",
    )


def repair_axis_weight_payload(
    payload: dict[str, Any],
    task_type: str,
    secondary_task_types: list[str],
    task_axis_priority_cfg: dict,
) -> tuple[AxisWeightPlannerResult, list[str]]:
    trace: list[str] = []
    fallback = fallback_axis_weight_result(
        task_type=task_type,
        secondary_task_types=secondary_task_types,
        task_axis_priority_cfg=task_axis_priority_cfg,
        problem="",
        context_summary="",
        signature=None,
        verifier_type_hint=None,
    )
    axis_priority = list(task_axis_priority_cfg["task_axis_priority"][task_type]["axis_priority"])
    top1 = axis_priority[:1]
    top2 = axis_priority[:2]

    raw_primary = payload.get("primary_axis_weight_hint")
    primary = _coerce_axis_weight_map(raw_primary, fallback.primary_axis_weight_hint, trace, prefix="primary")
    for axis in top1:
        if primary[axis] < 5:
            primary[axis] = 5
            trace.append(f"raised_primary_top1_axis_{axis}_to_5")
    for axis in top2:
        if primary[axis] < 4:
            primary[axis] = 4
            trace.append(f"raised_primary_top2_axis_{axis}_to_4")

    secondary_payload = payload.get("secondary_axis_weight_hints", []) or []
    if isinstance(secondary_payload, dict):
        secondary_payload = [secondary_payload]
    repaired_secondary: list[SecondaryAxisWeightHint] = []
    seen: set[str] = set()
    fallback_by_type = {item.task_type: item for item in fallback.secondary_axis_weight_hints}
    for item in secondary_payload:
        if not isinstance(item, dict):
            trace.append("dropped_non_object_secondary_axis_weight_hint")
            continue
        secondary_type = str(item.get("task_type") or "").strip()
        if secondary_type not in secondary_task_types[:3]:
            trace.append(f"dropped_unknown_secondary_task_type:{secondary_type}")
            continue
        if secondary_type in seen:
            trace.append(f"dropped_duplicate_secondary_task_type:{secondary_type}")
            continue
        seen.add(secondary_type)
        try:
            relevance = float(item.get("relevance", 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
            trace.append(f"repaired_secondary_relevance:{secondary_type}")
        relevance = min(1.0, max(0.0, relevance))
        fallback_hint = fallback_by_type.get(secondary_type)
        default_map = fallback_hint.axis_weight_hint if fallback_hint else fallback.primary_axis_weight_hint
        axis_map = _coerce_axis_weight_map(item.get("axis_weight_hint"), default_map, trace, prefix=f"secondary:{secondary_type}")
        repaired_secondary.append(
            SecondaryAxisWeightHint(
                task_type=secondary_type,
                relevance=relevance,
                axis_weight_hint=axis_map,
                reason=str(item.get("reason") or f"Secondary task type {secondary_type} contributes sample-level axis hints."),
            )
        )
    if len(repaired_secondary) > 3:
        repaired_secondary = repaired_secondary[:3]
        trace.append("trimmed_secondary_axis_weight_hints_to_3")
    for secondary in secondary_task_types[:3]:
        if secondary in seen:
            continue
        fallback_hint = fallback_by_type.get(secondary)
        if fallback_hint is None:
            continue
        repaired_secondary.append(fallback_hint)
        trace.append(f"filled_missing_secondary_axis_weight_hint:{secondary}")

    result = AxisWeightPlannerResult(
        primary_axis_weight_hint=primary,
        secondary_axis_weight_hints=repaired_secondary[:3],
        axis_weight_reason=str(payload.get("axis_weight_reason") or "LLM planned primary and secondary axis weights for this sample."),
    )
    return result, trace


def _coerce_axis_weight_map(
    raw_map: Any,
    fallback_map: dict[str, int],
    trace: list[str],
    prefix: str,
) -> dict[str, int]:
    repaired = dict(fallback_map)
    if not isinstance(raw_map, dict):
        trace.append(f"{prefix}_axis_weight_hint_used_fallback")
        return repaired
    for axis in AXES:
        raw_value = raw_map.get(axis)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            trace.append(f"{prefix}_axis_{axis}_used_fallback")
            continue
        if value < 1 or value > 7:
            value = min(7, max(1, value))
            trace.append(f"{prefix}_axis_{axis}_clamped")
        repaired[axis] = value
    return repaired


def _secondary_relevance(task_type: str, text: str) -> float:
    if task_type == "io_format_and_cli":
        return 0.85 if any(token in text for token in ["path", "file", "cli", "arg", "flag", "yaml", "json"]) else 0.5
    if task_type == "structured_data_processing":
        return 0.8 if any(token in text for token in ["table", "dataframe", "column", "row", "record"]) else 0.5
    if task_type == "numerical_computation":
        return 0.8 if any(token in text for token in ["float", "mean", "std", "energy", "matrix", "array"]) else 0.5
    if task_type == "code_validation_and_utility":
        return 0.8 if any(token in text for token in ["validate", "assert", "exception", "utility"]) else 0.5
    return 0.5


def _mock_axis_weight_builder(context: dict[str, Any]) -> dict[str, Any]:
    task_axis_priority_cfg = {
        "task_axis_priority": {
            context["task_type"]: {"axis_priority": context["task_axis_priority"]},
            **{
                task_type: {"axis_priority": axis_priority}
                for task_type, axis_priority in context.get("secondary_task_axis_priorities", {}).items()
                if axis_priority
            },
        }
    }
    result = fallback_axis_weight_result(
        task_type=context["task_type"],
        secondary_task_types=context.get("secondary_task_types", []),
        task_axis_priority_cfg=task_axis_priority_cfg,
        problem=context.get("problem", ""),
        context_summary=context.get("context_summary", ""),
        signature=context.get("signature"),
        verifier_type_hint=context.get("verifier_type_hint"),
    )
    return result.model_dump()
