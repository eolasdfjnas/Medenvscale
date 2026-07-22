from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from medenvscale.agent.tool_runtime import ToolRuntime
from medenvscale.agent.tool_schemas import stage06_tool_names, stage06_tool_schemas
from medenvscale.config import AppConfig
from medenvscale.llm import LLMClient
from medenvscale.llm.json_repair import parse_json_payload
from medenvscale.rubrics import score_requirement_rubrics
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.utils import append_jsonl, load_yaml, read_jsonl, slugify, write_jsonl
from tqdm.auto import tqdm


def run_stage06_tool_agent(
    *,
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    llm_client: LLMClient,
    output_dir,
    limit: int | None = None,
    retry_failed: bool = False,
    user_feedback: bool = False,
    resume: bool = False,
    parallel_workers: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    output_dir = output_dir / agent_output_slug(llm_client)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = read_jsonl(output_dir / "agent_runs.jsonl") if (retry_failed or resume) else []
    existing_traces = read_jsonl(output_dir / "agent_traces.jsonl") if (retry_failed or resume) else []
    existing_eval_rows = read_jsonl(output_dir / "agent_eval_report.jsonl") if (retry_failed or resume) else []
    env_by_id = {env.env_id: env for env in environments}
    selected = _select_stage06_environments(
        environments=environments,
        output_dir=output_dir,
        limit=limit,
        retry_failed=retry_failed,
        existing_runs=existing_runs,
        existing_traces=existing_traces,
        existing_eval_rows=existing_eval_rows,
        env_by_id=env_by_id,
    )
    selected_env_ids = [env.env_id for env in selected]
    checkpoint_path = output_dir / "agent_checkpoint.jsonl"
    checkpoint_rows = read_jsonl(checkpoint_path) if resume else []
    completed = _completed_agent_rows(
        runs=existing_runs,
        traces=existing_traces,
        eval_rows=existing_eval_rows,
        checkpoint_rows=checkpoint_rows,
    ) if resume else {}
    completed_selected = {env_id: completed[env_id] for env_id in selected_env_ids if env_id in completed}
    if resume and selected_env_ids and len(completed_selected) == len(selected_env_ids):
        tqdm.write(f"Stage06 resume: outputs/checkpoints complete; skipping {len(selected_env_ids)} envs in {output_dir}")
        runs = [completed_selected[env_id]["run"] for env_id in selected_env_ids]
        traces = [completed_selected[env_id]["trace"] for env_id in selected_env_ids]
        eval_rows = [completed_selected[env_id]["eval"] for env_id in selected_env_ids]
        _write_agent_outputs(output_dir=output_dir, env_by_id=env_by_id, runs=runs, traces=traces, eval_rows=eval_rows, llm_client=llm_client)
        summary = build_stage06_summary(runs=runs, traces=traces, eval_rows=eval_rows, llm_client=llm_client)
        return {"runs": runs, "traces": traces, "eval_report": eval_rows, "summary": summary}
    if completed_selected:
        tqdm.write(f"Stage06 resume: loaded {len(completed_selected)} completed envs from outputs/checkpoint")
    pending = [env for env in selected if env.env_id not in completed_selected]
    stage_cfg = cfg.values.get("stage06", {}) or {}
    agent_cfg = stage_cfg.get("tool_agent", stage_cfg) or {}
    tool_pool_cfg = load_yaml(cfg.dataset_config_path("tool_pool.yaml"))
    budget_cfg = load_yaml(
        cfg.dataset_config_path_with_fallback("m_level_budgets_4axis.yaml", "m_level_budgets_7axis.yaml")
    )
    worker_count = max(1, int(parallel_workers or 1))
    if llm_client.mode == "local" and worker_count > 1:
        tqdm.write("Stage06 workers is forced to 1 for local LLM mode to avoid concurrent model.generate calls.")
        worker_count = 1
    indexed_results: list[tuple[int, dict[str, dict[str, Any]]]] = [
        (index, completed_selected[env.env_id])
        for index, env in enumerate(selected)
        if env.env_id in completed_selected
    ]
    abort_on_llm_error = bool(agent_cfg.get("abort_on_llm_error", False))
    progress = tqdm(total=len(selected), initial=len(indexed_results), desc="Stage06 Tool Agent", unit="env", leave=True)
    try:
        if worker_count == 1:
            for index, env in ((index, env) for index, env in enumerate(selected) if env.env_id not in completed_selected):
                row = _run_agent_row_with_error_handling(
                    env=env,
                    cfg=cfg,
                    llm_client=llm_client,
                    agent_cfg=agent_cfg,
                    tool_pool_cfg=tool_pool_cfg,
                    budget_cfg=budget_cfg,
                    user_feedback=user_feedback,
                    abort_on_llm_error=abort_on_llm_error,
                )
                indexed_results.append((index, row))
                if resume:
                    _append_agent_checkpoint(checkpoint_path, row)
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _run_agent_row_with_error_handling,
                        env=env,
                        cfg=cfg,
                        llm_client=llm_client,
                        agent_cfg=agent_cfg,
                        tool_pool_cfg=tool_pool_cfg,
                        budget_cfg=budget_cfg,
                        user_feedback=user_feedback,
                        abort_on_llm_error=abort_on_llm_error,
                    ): index
                    for index, env in ((index, env) for index, env in enumerate(selected) if env.env_id not in completed_selected)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    row = future.result()
                    indexed_results.append((index, row))
                    if resume:
                        _append_agent_checkpoint(checkpoint_path, row)
                    progress.update(1)
    finally:
        progress.close()
    rows = [row for _, row in sorted(indexed_results, key=lambda item: item[0])]
    runs = [row["run"] for row in rows]
    traces = [row["trace"] for row in rows]
    eval_rows = [row["eval"] for row in rows]
    if retry_failed:
        retried_ids = {row["env_id"] for row in runs}
        runs = _merge_rows_by_env_id(existing_runs, runs, retried_ids)
        traces = _merge_rows_by_env_id(existing_traces, traces, retried_ids)
        eval_rows = _merge_rows_by_env_id(existing_eval_rows, eval_rows, retried_ids)
    summary = _write_agent_outputs(output_dir=output_dir, env_by_id=env_by_id, runs=runs, traces=traces, eval_rows=eval_rows, llm_client=llm_client)
    return {"runs": runs, "traces": traces, "eval_report": eval_rows, "summary": summary}


def _run_agent_row_with_error_handling(
    *,
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    agent_cfg: dict[str, Any],
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    user_feedback: bool,
    abort_on_llm_error: bool,
) -> dict[str, dict[str, Any]]:
    try:
        return run_tool_agent_for_env(
            env=env,
            cfg=cfg,
            llm_client=llm_client,
            agent_cfg=agent_cfg,
            tool_pool_cfg=tool_pool_cfg,
            budget_cfg=budget_cfg,
            user_feedback=user_feedback,
        )
    except Exception as exc:
        if abort_on_llm_error:
            raise
        tqdm.write(f"Stage06 LLM error on {env.env_id}: {_stage06_console_error_message(exc)}")
        return _failed_agent_row(env, exc)


def _write_agent_outputs(
    *,
    output_dir: Path,
    env_by_id: dict[str, ExecutableEnvSpec],
    runs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    llm_client: LLMClient,
) -> dict[str, Any]:
    write_jsonl(output_dir / "agent_runs.jsonl", runs)
    write_jsonl(output_dir / "agent_traces.jsonl", traces)
    write_jsonl(output_dir / "agent_eval_report.jsonl", eval_rows)
    _write_retry_failed_environments(output_dir=output_dir, env_by_id=env_by_id, runs=runs, traces=traces, eval_rows=eval_rows)
    summary = build_stage06_summary(runs=runs, traces=traces, eval_rows=eval_rows, llm_client=llm_client)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _append_agent_checkpoint(path: Path, row: dict[str, dict[str, Any]]) -> None:
    append_jsonl(
        path,
        {
            "env_id": row["run"]["env_id"],
            "run": row["run"],
            "trace": row["trace"],
            "eval": row["eval"],
        },
    )


def _completed_agent_rows(
    *,
    runs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    completed: dict[str, dict[str, dict[str, Any]]] = {}
    run_by_id = {str(row.get("env_id") or ""): row for row in runs if row.get("env_id")}
    trace_by_id = {str(row.get("env_id") or ""): row for row in traces if row.get("env_id")}
    eval_by_id = {str(row.get("env_id") or ""): row for row in eval_rows if row.get("env_id")}
    for env_id, run in run_by_id.items():
        trace = trace_by_id.get(env_id)
        eval_row = eval_by_id.get(env_id)
        if trace is not None and eval_row is not None:
            completed[env_id] = {"run": run, "trace": trace, "eval": eval_row}
    for row in checkpoint_rows:
        env_id = str(row.get("env_id") or "")
        if env_id and isinstance(row.get("run"), dict) and isinstance(row.get("trace"), dict) and isinstance(row.get("eval"), dict):
            completed[env_id] = {"run": row["run"], "trace": row["trace"], "eval": row["eval"]}
    return completed


def run_tool_agent_for_env(
    *,
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    agent_cfg: dict[str, Any],
    tool_pool_cfg: dict[str, Any] | None = None,
    budget_cfg: dict[str, Any] | None = None,
    user_feedback: bool = False,
    task_name: str = "stage06_tool_agent",
    context_extra: dict[str, Any] | None = None,
    include_messages: bool = False,
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
            task_name=task_name,
            messages=messages,
            tools=tools,
            context={
                "env_id": env.env_id,
                "difficulty": env.difficulty.model_dump() if env.difficulty else {},
                **(context_extra or {}),
            },
            mock_builder=lambda context: _mock_agent_message(env),
        )
        if response.tool_calls:
            messages.append(response.raw_message)
            for call in response.tool_calls:
                result = runtime.execute(call.name, call.arguments)
                if runtime.terminated:
                    final_code = runtime.final_code or ""
                    break
                _append_tool_result(messages, call.id, call.name, result)
                if user_feedback and call.name == "submit_final_code":
                    messages.append({"role": "user", "content": _final_code_repair_prompt(result)})
            if runtime.terminated:
                break
            continue
        parsed = _parse_final_response(response.content)
        final_code = str(parsed.get("final_code") or "")
        notes = [str(item) for item in parsed.get("notes", []) if str(item).strip()]
        if final_code:
            result = _submit_final_code_as_tool(messages, runtime, final_code, f"implicit_submit_{len(messages)}")
            if runtime.terminated:
                break
            if user_feedback:
                messages.append({"role": "user", "content": _final_code_repair_prompt(result)})
            continue
        messages.append({"role": "assistant", "content": response.content or ""})
        messages.append(
            {
                "role": "user",
                "content": (
                    "NO_FINAL_CODE_SUBMITTED: No tool call or final_code was provided. "
                    "Continue with a useful public tool call if more evidence is needed, or submit complete "
                    "executable Python code with submit_final_code / JSON final_code when ready. "
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
    if include_messages:
        trace["messages"] = messages
    rubric_result = score_requirement_rubrics(env.rubrics or [], case_reports)
    eval_row = {
        "env_id": env.env_id,
        "level": level,
        "compile_passed": bool(eval_report.get("compile_passed")),
        "execution_passed": bool(eval_report.get("execution_passed")),
        "evaluation_case_source": str(eval_report.get("evaluation_case_source") or ""),
        "passed_cases": sum(1 for report in case_reports if report.get("passed")),
        "total_cases": len(case_reports),
        "failure_reasons": list(eval_report.get("failure_reasons", [])),
        "case_reports": case_reports,
        "rubric_score": rubric_result["rubric_score"],
        "rubric_scores": rubric_result["rubric_scores"],
        "rubric_by_category": rubric_result["rubric_by_category"],
        "scored_rubrics": rubric_result["scored_rubrics"],
        "total_rubrics": rubric_result["total_rubrics"],
    }
    return {"run": run, "trace": trace, "eval": eval_row}


def _select_stage06_environments(
    *,
    environments: list[ExecutableEnvSpec],
    output_dir,
    limit: int | None,
    retry_failed: bool,
    existing_runs: list[dict[str, Any]],
    existing_traces: list[dict[str, Any]],
    existing_eval_rows: list[dict[str, Any]],
    env_by_id: dict[str, ExecutableEnvSpec],
) -> list[ExecutableEnvSpec]:
    if not retry_failed:
        return environments[:limit] if limit is not None else environments
    retry_path = output_dir / "retry_failed_envs.jsonl"
    if retry_path.exists():
        selected = [ExecutableEnvSpec.model_validate(row) for row in read_jsonl(retry_path)]
    else:
        retry_ids = _retryable_stage06_env_ids(runs=existing_runs, traces=existing_traces, eval_rows=existing_eval_rows)
        selected = [env_by_id[env_id] for env_id in retry_ids if env_id in env_by_id]
    return selected[:limit] if limit is not None else selected


def _merge_rows_by_env_id(
    existing_rows: list[dict[str, Any]],
    replacement_rows: list[dict[str, Any]],
    replacement_ids: set[str],
) -> list[dict[str, Any]]:
    replacements = {row["env_id"]: row for row in replacement_rows}
    merged = []
    seen: set[str] = set()
    for row in existing_rows:
        env_id = row.get("env_id")
        if env_id in replacement_ids:
            replacement = replacements.get(env_id)
            if replacement is not None:
                merged.append(replacement)
                seen.add(env_id)
            continue
        merged.append(row)
        if env_id:
            seen.add(str(env_id))
    for row in replacement_rows:
        env_id = row.get("env_id")
        if env_id not in seen:
            merged.append(row)
    return merged


def _write_retry_failed_environments(
    *,
    output_dir,
    env_by_id: dict[str, ExecutableEnvSpec],
    runs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    retry_ids = _retryable_stage06_env_ids(runs=runs, traces=traces, eval_rows=eval_rows)
    retry_envs = [env_by_id[env_id].model_dump() for env_id in retry_ids if env_id in env_by_id]
    retry_report = []
    run_by_id = {row.get("env_id"): row for row in runs}
    trace_by_id = {row.get("env_id"): row for row in traces}
    eval_by_id = {row.get("env_id"): row for row in eval_rows}
    for env_id in retry_ids:
        row = eval_by_id.get(env_id, {})
        retry_report.append(
            {
                "env_id": env_id,
                "failure_reasons": row.get("failure_reasons") or [],
                "error_excerpt": _stage06_failure_text(
                    run=run_by_id.get(env_id, {}),
                    trace=trace_by_id.get(env_id, {}),
                    eval_row=row,
                )[:1000],
            }
        )
    write_jsonl(output_dir / "retry_failed_envs.jsonl", retry_envs)
    write_jsonl(output_dir / "retry_failed_report.jsonl", retry_report)


def _retryable_stage06_env_ids(
    *,
    runs: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[str]:
    run_by_id = {row.get("env_id"): row for row in runs}
    trace_by_id = {row.get("env_id"): row for row in traces}
    retry_ids = []
    for row in eval_rows:
        env_id = str(row.get("env_id") or "")
        if not env_id:
            continue
        if _is_retryable_stage06_llm_failure(
            eval_row=row,
            run=run_by_id.get(env_id, {}),
            trace=trace_by_id.get(env_id, {}),
        ):
            retry_ids.append(env_id)
    return retry_ids


def _is_retryable_stage06_llm_failure(
    *,
    eval_row: dict[str, Any],
    run: dict[str, Any],
    trace: dict[str, Any],
) -> bool:
    if eval_row.get("evaluation_case_source") != "llm_error":
        return False
    text = _stage06_failure_text(run=run, trace=trace, eval_row=eval_row).lower()
    non_retryable_markers = (
        "http error 400",
        "bad request",
        "http error 401",
        "unauthorized",
        "http error 403",
        "forbidden",
        "model not found",
        "not available in your region",
        "region",
        "invalid api key",
    )
    if any(marker in text for marker in non_retryable_markers):
        return False
    retryable_markers = (
        "llm network error",
        "remotedisconnected",
        "remote end closed connection",
        "transient http error",
        "http error 408",
        "http error 409",
        "http error 429",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "rate limit",
    )
    return any(marker in text for marker in retryable_markers)


def _stage06_console_error_message(exc: Exception) -> str:
    message = str(exc)
    raw_marker = ". Raw content:"
    if raw_marker in message:
        message = message.split(raw_marker, 1)[0]
    return f"{exc.__class__.__name__}: {message[:300]}"


def _stage06_failure_text(*, run: dict[str, Any], trace: dict[str, Any], eval_row: dict[str, Any]) -> str:
    chunks = []
    chunks.extend(str(item) for item in (run.get("notes") or []))
    chunks.extend(str(item) for item in (eval_row.get("failure_reasons") or []))
    for step in trace.get("tool_trace") or []:
        result = step.get("result") or {}
        chunks.append(str(result.get("error_type") or ""))
        chunks.append(str(result.get("error_message") or ""))
    return "\n".join(item for item in chunks if item)


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
        "level": _env_level(env),
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
        "You are an autonomous tool-using coding agent solving a coding task. "
        "Your goal is to submit a complete executable Python program that satisfies the public task. "
        "The oracle cases are hidden and unavailable; do not ask for hidden tests or infer hidden case details. "
        "Use public tools to decide whether your candidate code is ready. "
        "Start by inspecting the task with get_task_context when you need context. "
        "Use validate_candidate_code to check syntax, the target signature, imports, and static safety. "
        "Use create_test_file before run_custom_test when you need a small CSV, TSV, JSON, FASTA, text, "
        "or config fixture for public self-testing. "
        "Use run_custom_test for targeted public checks when behavior, edge cases, file parsing, or output "
        "formatting is uncertain. "
        "If a public tool reports a failure, decide whether to repair the code, run another targeted public "
        "test, inspect more public context, or submit only if the feedback is irrelevant. "
        "Do not call tools mechanically, and do not repair code that already satisfies the relevant public "
        "checks unless you have a concrete reason. "
        "Submit with submit_final_code only when you believe the program is complete and ready. "
        "Do not return a patch or snippet. "
        "Your submitted code must be self-contained except for the Python standard library and imports "
        "that are visibly provided and available in the task context. Do not import placeholder or "
        "project-local modules such as my_module, hidden helper modules, or optional helper packages unless "
        "a tool check confirms they are available. If the scaffold mentions a missing helper module, "
        "implement the needed helper behavior inline in final_code. Prefer standard-library or already "
        "visible dependency alternatives, for example requests files/data instead of requests_toolbelt for "
        "multipart form uploads. If you answer without a tool call, return JSON with final_code and notes."
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


def _submit_final_code_as_tool(
    messages: list[dict[str, Any]],
    runtime: ToolRuntime,
    code: str,
    call_id: str,
) -> dict[str, Any]:
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "submit_final_code", "arguments": json.dumps({"code": code}, ensure_ascii=False)},
                }
            ],
        }
    )
    result = runtime.execute("submit_final_code", {"code": code})
    if not result.get("terminated"):
        _append_tool_result(messages, call_id, "submit_final_code", result)
    return result


def _append_tool_result(messages: list[dict[str, Any]], call_id: str, name: str, result: dict[str, Any]) -> None:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


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
        "Your submitted final_code failed public preflight and was not evaluated. "
        "Use the visible feedback to decide whether to repair the code, run a targeted public test, "
        "inspect public context, or submit a corrected final_code.\n"
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
        "max_total_tool_calls": 4,
        "max_calls_per_tool": {
            "get_task_context": 1,
            "create_test_file": 3,
            "validate_candidate_code": 2,
            "run_custom_test": 3,
            "submit_final_code": 1,
        },
    }


def agent_model_slug(llm_client: LLMClient) -> str:
    if llm_client.mode == "local":
        model_path = str(((llm_client.config or {}).get("local") or {}).get("model_path") or "local_model")
        adapter_path = str(((llm_client.config or {}).get("local") or {}).get("adapter_path") or "").strip()
        if adapter_path:
            adapter = Path(adapter_path)
            adapter_name = adapter.parent.name if adapter.name == "adapter" else adapter.name
            return slugify(f"{Path(model_path).name or 'local_model'}_{adapter_name}", max_length=80)
        return slugify(Path(model_path).name or "local_model", max_length=80)
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
    zero_case_samples = sum(int(row.get("total_cases") or 0) == 0 for row in eval_rows)
    no_final_code_samples = sum(
        "MAX_TURNS_WITHOUT_VALID_FINAL_CODE" in (row.get("failure_reasons") or []) for row in eval_rows
    )
    rubric_rows = [row for row in eval_rows if int(row.get("scored_rubrics") or 0) > 0]
    zero_case_penalty_total = case_total + zero_case_samples
    levels: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        run = by_env_run.get(row["env_id"], {})
        level = str(row.get("level") or (run.get("difficulty") or {}).get("global_level") or "unknown")
        grouped[level].append(row)
    for level, rows in sorted(grouped.items()):
        total_cases = sum(int(row.get("total_cases") or 0) for row in rows)
        passed_cases = sum(int(row.get("passed_cases") or 0) for row in rows)
        level_zero_case_samples = sum(int(row.get("total_cases") or 0) == 0 for row in rows)
        level_no_final_code_samples = sum(
            "MAX_TURNS_WITHOUT_VALID_FINAL_CODE" in (row.get("failure_reasons") or []) for row in rows
        )
        level_rubric_rows = [row for row in rows if int(row.get("scored_rubrics") or 0) > 0]
        level_penalty_total = total_cases + level_zero_case_samples
        levels[level] = {
            "samples": len(rows),
            "execution_passed": sum(bool(row.get("execution_passed")) for row in rows),
            "zero_case_samples": level_zero_case_samples,
            "unevaluated_samples": level_zero_case_samples,
            "no_final_code_samples": level_no_final_code_samples,
            "sample_pass_rate": _rate(sum(bool(row.get("execution_passed")) for row in rows), len(rows)),
            "case_pass_rate": _rate(passed_cases, level_penalty_total),
            "case_pass_rate_nonzero_only": _rate(passed_cases, total_cases),
            "case_pass_rate_with_zero_case_penalty": _rate(passed_cases, level_penalty_total),
            "passed_cases": passed_cases,
            "total_cases": total_cases,
            "zero_case_penalty_total_cases": level_penalty_total,
            "mean_rubric_score": _mean(float(row.get("rubric_score") or 0.0) for row in level_rubric_rows),
            "rubric_scored_samples": len(level_rubric_rows),
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
        "case_pass_rate": _rate(case_passed, zero_case_penalty_total),
        "case_pass_rate_nonzero_only": _rate(case_passed, case_total),
        "case_pass_rate_with_zero_case_penalty": _rate(case_passed, zero_case_penalty_total),
        "zero_case_samples": zero_case_samples,
        "mean_rubric_score": _mean(float(row.get("rubric_score") or 0.0) for row in rubric_rows),
        "rubric_scored_samples": len(rubric_rows),
        "unevaluated_samples": zero_case_samples,
        "no_final_code_samples": no_final_code_samples,
        "zero_case_penalty_total_cases": zero_case_penalty_total,
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


def _mean(values) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 6) if items else 0.0
