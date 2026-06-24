from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from medenvscale.agent.tool_runtime import ToolRuntime
from medenvscale.agent.tool_schemas import stage06_tool_names, stage06_tool_schemas
from medenvscale.config import AppConfig
from medenvscale.llm import LLMClient
from medenvscale.llm.json_repair import parse_json_payload
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.utils import load_yaml, slugify, write_jsonl
from tqdm.auto import tqdm


def run_stage06_tool_agent(
    *,
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    llm_client: LLMClient,
    output_dir,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected = environments[:limit] if limit is not None else environments
    stage_cfg = cfg.values.get("stage06", {}) or {}
    agent_cfg = stage_cfg.get("tool_agent", stage_cfg) or {}
    tool_pool_cfg = load_yaml(cfg.dataset_config_path("tool_pool.yaml"))
    budget_cfg = load_yaml(
        cfg.dataset_config_path_with_fallback("m_level_budgets_4axis.yaml", "m_level_budgets_7axis.yaml")
    )
    runs = []
    traces = []
    eval_rows = []
    abort_on_llm_error = bool(agent_cfg.get("abort_on_llm_error", False))
    progress = tqdm(total=len(selected), desc="Stage06 Tool Agent", unit="env", leave=True)
    try:
        for env in selected:
            try:
                row = run_tool_agent_for_env(
                    env=env,
                    cfg=cfg,
                    llm_client=llm_client,
                    agent_cfg=agent_cfg,
                    tool_pool_cfg=tool_pool_cfg,
                    budget_cfg=budget_cfg,
                )
            except Exception as exc:
                if abort_on_llm_error:
                    raise
                tqdm.write(f"Stage06 LLM error on {env.env_id}: {exc.__class__.__name__}: {str(exc)[:300]}")
                row = _failed_agent_row(env, exc)
            runs.append(row["run"])
            traces.append(row["trace"])
            eval_rows.append(row["eval"])
            progress.update(1)
    finally:
        progress.close()
    output_dir = output_dir / agent_output_slug(llm_client)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "agent_runs.jsonl", runs)
    write_jsonl(output_dir / "agent_traces.jsonl", traces)
    write_jsonl(output_dir / "agent_eval_report.jsonl", eval_rows)
    summary = build_stage06_summary(runs=runs, traces=traces, eval_rows=eval_rows, llm_client=llm_client)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"runs": runs, "traces": traces, "eval_report": eval_rows, "summary": summary}


def run_tool_agent_for_env(
    *,
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    agent_cfg: dict[str, Any],
    tool_pool_cfg: dict[str, Any] | None = None,
    budget_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    level = _env_level(env)
    stage06_budget_cfg = (budget_cfg or {}).get("stage06_tool_agent", {})
    env_budget = _budget_for_level(level=level, agent_cfg=agent_cfg, stage06_budget_cfg=stage06_budget_cfg)
    allowed_tools = stage06_tool_names(tool_pool_cfg)
    runtime = ToolRuntime(
        env,
        cfg,
        budget=env_budget,
        allowed_tools=allowed_tools,
        submit_excluded_from_total=bool(stage06_budget_cfg.get("submit_final_code_excluded_from_total", True)),
    )
    max_turns = int((stage06_budget_cfg.get("max_turns_by_level") or {}).get(level, agent_cfg.get("max_turns", 8)))
    tools = stage06_tool_schemas(tool_pool_cfg)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _public_user_prompt(env)},
    ]
    final_code = ""
    notes: list[str] = []

    for _ in range(max_turns):
        response = llm_client.complete_with_tools(
            task_name="stage06_tool_agent",
            messages=messages,
            tools=tools,
            context={"env_id": env.env_id, "difficulty": env.difficulty.model_dump() if env.difficulty else {}},
            mock_builder=lambda context: _mock_agent_message(env),
        )
        if response.tool_calls:
            messages.append(response.raw_message)
            for call in response.tool_calls:
                result = runtime.execute(call.name, call.arguments)
                if runtime.terminated:
                    final_code = runtime.final_code or ""
                    break
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            if runtime.terminated:
                break
            continue
        parsed = _parse_final_response(response.content)
        final_code = str(parsed.get("final_code") or "")
        notes = [str(item) for item in parsed.get("notes", []) if str(item).strip()]
        if final_code:
            result = runtime.execute("submit_final_code", {"code": final_code})
            if runtime.terminated:
                break
            messages.append({"role": "assistant", "content": response.content or ""})
            messages.append({"role": "user", "content": _final_code_repair_prompt(result)})
            continue
        messages.append({"role": "assistant", "content": response.content or ""})
        messages.append(
            {
                "role": "user",
                "content": (
                    "NO_FINAL_CODE_SUBMITTED: You must submit complete executable Python code. "
                    "Call submit_final_code or return JSON with a non-empty final_code string. "
                    "Do not use Markdown fences or prose."
                ),
            }
        )
        continue

    if not runtime.terminated:
        runtime.mark_no_final_code("MAX_TURNS_WITHOUT_VALID_FINAL_CODE")
    eval_report = runtime.final_eval or {}
    case_reports = eval_report.get("case_reports", [])
    run = {
        "env_id": env.env_id,
        "original_task_id": env.original_task_id,
        "difficulty": env.difficulty.model_dump() if env.difficulty else {},
        "final_code": runtime.final_code or final_code,
        "notes": notes,
        "tool_budget_used": {
            "total": runtime.total_calls,
            "per_tool": runtime.call_counts,
        },
        "passed": bool(eval_report.get("compile_passed") and eval_report.get("execution_passed")),
    }
    trace = {
        "env_id": env.env_id,
        "tool_trace": runtime.trace,
        "message_count": len(messages),
    }
    eval_row = {
        "env_id": env.env_id,
        "compile_passed": bool(eval_report.get("compile_passed")),
        "execution_passed": bool(eval_report.get("execution_passed")),
        "evaluation_case_source": str(eval_report.get("evaluation_case_source") or ""),
        "passed_cases": sum(1 for report in case_reports if report.get("passed")),
        "total_cases": len(case_reports),
        "failure_reasons": list(eval_report.get("failure_reasons", [])),
        "case_reports": case_reports,
    }
    return {"run": run, "trace": trace, "eval": eval_row}


def _failed_agent_row(env: ExecutableEnvSpec, exc: Exception) -> dict[str, dict[str, Any]]:
    error_type = exc.__class__.__name__
    error_message = str(exc)
    failure_reason = f"LLM_API_ERROR:{error_type}"
    run = {
        "env_id": env.env_id,
        "original_task_id": env.original_task_id,
        "difficulty": env.difficulty.model_dump() if env.difficulty else {},
        "final_code": "",
        "notes": [error_message[:2000]],
        "tool_budget_used": {"total": 0, "per_tool": {}},
        "passed": False,
    }
    trace = {
        "env_id": env.env_id,
        "tool_trace": [
            {
                "tool_name": "llm_client.complete_with_tools",
                "arguments": {},
                "result": {
                    "ok": False,
                    "error_type": error_type,
                    "error_message": error_message[:4000],
                },
            }
        ],
        "message_count": 0,
    }
    eval_row = {
        "env_id": env.env_id,
        "compile_passed": False,
        "execution_passed": False,
        "evaluation_case_source": "llm_error",
        "passed_cases": 0,
        "total_cases": 0,
        "failure_reasons": [failure_reason],
        "case_reports": [],
    }
    return {"run": run, "trace": trace, "eval": eval_row}


def _system_prompt() -> str:
    return (
        "You are solving a coding task. Return a complete executable Python program. "
        "You may use tools to inspect public context and test your own code. "
        "The oracle cases are hidden and unavailable. Do not ask for hidden tests. "
        "Do not return a patch or snippet. "
        "Your submitted code must be self-contained except for the Python standard library and imports "
        "that are visibly provided and available in the task context. Do not import placeholder or "
        "project-local modules such as my_module, hidden helper modules, or optional helper packages unless "
        "a tool check confirms they are available. If the scaffold mentions a missing helper module, "
        "implement the needed helper behavior inline in final_code. Prefer standard-library or already "
        "visible dependency alternatives, for example requests files/data instead of requests_toolbelt for "
        "multipart form uploads. Final answer must be JSON with final_code and notes."
    )


def _public_user_prompt(env: ExecutableEnvSpec) -> str:
    return (
        "Task:\n"
        f"{env.user_prompt or env.problem}\n\n"
        "Signature:\n"
        f"{env.signature or ''}\n\n"
        "Context:\n"
        f"{env.context}\n\n"
        "Resource files:\n"
        f"{json.dumps(env.resource_manifest or [{'path': path} for path in env.resource_files], ensure_ascii=False)}\n\n"
        "Return a complete executable Python program."
    )


def _mock_agent_message(env: ExecutableEnvSpec) -> dict[str, Any]:
    code = env.scaled_executable_gold_code or env.gold_solution or ""
    return {
        "role": "assistant",
        "content": json.dumps(
            {
                "final_code": code,
                "notes": ["mock mode submits the available reference code for pipeline validation"],
            },
            ensure_ascii=False,
        ),
    }


def _parse_final_response(content: str) -> dict[str, Any]:
    try:
        parsed = parse_json_payload(content)
    except Exception:
        return {"final_code": content, "notes": ["non_json_final_response"]}
    if not isinstance(parsed, dict):
        return {"final_code": str(content or ""), "notes": ["unexpected_final_response_shape"]}
    return parsed


def _final_code_repair_prompt(result: dict[str, Any]) -> str:
    return (
        "Your submitted final_code failed preflight and was not evaluated. "
        "Repair it and submit again as complete executable Python code only.\n"
        f"Errors: {json.dumps(result.get('errors', []), ensure_ascii=False)}\n"
        f"Repair hints: {json.dumps(result.get('repair_hints', []), ensure_ascii=False)}"
    )


def _env_level(env: ExecutableEnvSpec) -> str:
    if env.difficulty and env.difficulty.global_level:
        return str(env.difficulty.global_level)
    return str((env.scaling or {}).get("global_level") or "M2")


def _budget_for_level(level: str, agent_cfg: dict[str, Any], stage06_budget_cfg: dict[str, Any]) -> dict[str, Any]:
    per_level = stage06_budget_cfg.get("per_level") or {}
    if level in per_level:
        return dict(per_level[level])
    if agent_cfg.get("tool_budget"):
        return dict(agent_cfg["tool_budget"])
    return {
        "max_total_tool_calls": 6,
        "max_calls_per_tool": {
            "get_task_context": 1,
            "read_resource_file": 3,
            "validate_candidate_code": 2,
            "run_custom_test": 3,
            "submit_final_code": 1,
        },
    }


def agent_model_slug(llm_client: LLMClient) -> str:
    model = str(((llm_client.config or {}).get("api") or {}).get("model") or llm_client.mode or "agent")
    return slugify(model, max_length=80)


def agent_output_slug(llm_client: LLMClient) -> str:
    model_slug = agent_model_slug(llm_client)
    if llm_client.mode == "api":
        return model_slug
    return slugify(f"{model_slug}_{llm_client.mode}", max_length=80)


def build_stage06_summary(
    *,
    runs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    llm_client: LLMClient,
) -> dict[str, Any]:
    by_env_run = {row["env_id"]: row for row in runs}
    case_total = sum(int(row.get("total_cases") or 0) for row in eval_rows)
    case_passed = sum(int(row.get("passed_cases") or 0) for row in eval_rows)
    nonzero = [row for row in eval_rows if int(row.get("total_cases") or 0) > 0]
    levels: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        run = by_env_run.get(row["env_id"], {})
        level = str((run.get("difficulty") or {}).get("global_level") or "unknown")
        grouped[level].append(row)
    for level, rows in sorted(grouped.items()):
        total_cases = sum(int(row.get("total_cases") or 0) for row in rows)
        passed_cases = sum(int(row.get("passed_cases") or 0) for row in rows)
        levels[level] = {
            "samples": len(rows),
            "execution_passed": sum(bool(row.get("execution_passed")) for row in rows),
            "zero_case_samples": sum(int(row.get("total_cases") or 0) == 0 for row in rows),
            "sample_pass_rate": _rate(sum(bool(row.get("execution_passed")) for row in rows), len(rows)),
            "case_pass_rate": _rate(passed_cases, total_cases),
            "passed_cases": passed_cases,
            "total_cases": total_cases,
        }
    tool_counts: Counter[str] = Counter()
    trace_lengths = []
    for trace in traces:
        steps = trace.get("tool_trace") or []
        trace_lengths.append(len(steps))
        for step in steps:
            tool_counts[str(step.get("tool_name") or "")] += 1
    case_sources = Counter(str(row.get("evaluation_case_source") or "unknown") for row in eval_rows)
    return {
        "model": str(((llm_client.config or {}).get("api") or {}).get("model") or ""),
        "mode": llm_client.mode,
        "num_samples": len(eval_rows),
        "compile_passed": sum(bool(row.get("compile_passed")) for row in eval_rows),
        "execution_passed": sum(bool(row.get("execution_passed")) for row in eval_rows),
        "sample_pass_rate": _rate(sum(bool(row.get("execution_passed")) for row in eval_rows), len(eval_rows)),
        "nonzero_case_samples": len(nonzero),
        "nonzero_case_execution_passed": sum(bool(row.get("execution_passed")) for row in nonzero),
        "nonzero_case_sample_pass_rate": _rate(sum(bool(row.get("execution_passed")) for row in nonzero), len(nonzero)),
        "passed_cases": case_passed,
        "total_cases": case_total,
        "case_pass_rate": _rate(case_passed, case_total),
        "evaluation_case_sources": dict(case_sources),
        "levels": levels,
        "tool_counts": dict(tool_counts),
        "trace_steps": {
            "min": min(trace_lengths) if trace_lengths else 0,
            "max": max(trace_lengths) if trace_lengths else 0,
            "mean": round(sum(trace_lengths) / len(trace_lengths), 4) if trace_lengths else 0.0,
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
