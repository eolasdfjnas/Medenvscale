from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from medenvscale.agent.runner import (
    _budget_for_level,
    _env_level,
    _final_code_repair_prompt,
    _public_user_prompt,
    _submit_final_code_as_tool,
    _system_prompt,
    agent_output_slug,
)
from medenvscale.agent.tool_runtime import ToolRuntime
from medenvscale.agent.tool_schemas import stage06_tool_names, stage06_tool_schemas
from medenvscale.config import AppConfig
from medenvscale.llm import LLMClient
from medenvscale.llm.json_repair import parse_json_payload
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.split_assignment import has_stage05_5_split, split_envs_by_assigned_split
from medenvscale.utils import append_jsonl, read_jsonl, seeded_shuffle, stable_hash, write_jsonl
from tqdm.auto import tqdm


def generate_tool_sft_data(
    *,
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    output_paths: dict[str, Path],
    limit: int | None = None,
    user_feedback: bool = False,
    resume: bool = False,
    parallel_workers: int | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    selected = environments[:limit] if limit is not None else environments
    tool_pool_cfg = _load_dataset_yaml(cfg, "tool_pool.yaml")
    budget_cfg = _load_budget_cfg(cfg)
    agent_cfg = ((cfg.values.get("stage06", {}) or {}).get("tool_agent") or {}) or {}
    stage07_cfg = cfg.values.get("stage07_tool_sft", {}) or {}
    teacher_slug = agent_output_slug(llm_client)
    if resume and _tool_sft_outputs_complete(output_paths, expected_env_count=len(selected)):
        manifest = json.loads(output_paths["manifest"].read_text(encoding="utf-8"))
        rows = read_jsonl(output_paths["trajectories"])
        quality_report = read_jsonl(output_paths["quality_report"])
        split_rows = {name: read_jsonl(output_paths[f"split_{name}"]) for name in ("train", "dev", "test")}
        tqdm.write(f"Stage07 resume: outputs complete; skipping ({output_paths['manifest']})")
        return {"trajectories": rows, "quality_report": quality_report, "splits": split_rows, "manifest": manifest}
    split_source = "stage05_5_assigned_split" if has_stage05_5_split(selected) else "legacy_stage07_group_split"
    env_splits = (
        split_envs_by_assigned_split(selected)
        if split_source == "stage05_5_assigned_split"
        else _split_envs_by_original_task(selected, cfg.values.get("splits", {}) or {})
    )
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    rejected: list[dict[str, Any]] = []
    progress = tqdm(total=sum(len(items) for items in env_splits.values()), desc="Stage07 Tool SFT", unit="env", leave=True)
    try:
        for split_name, split_envs in env_splits.items():
            split_rows[split_name] = _generate_rows_for_envs(
                cfg=cfg,
                environments=split_envs,
                split_name=split_name,
                llm_client=llm_client,
                prompt_runner=prompt_runner,
                tool_pool_cfg=tool_pool_cfg,
                budget_cfg=budget_cfg,
                agent_cfg=agent_cfg,
                stage07_cfg=stage07_cfg,
                teacher_slug=teacher_slug,
                user_feedback=user_feedback,
                rejected=rejected,
                progress=progress,
                resume=resume,
                parallel_workers=parallel_workers,
                checkpoint_path=checkpoint_path,
            )
    finally:
        progress.close()
    rows = [row for split_name in ("train", "dev", "test") for row in split_rows[split_name]]
    quality_report = _build_quality_report(rows, rejected)
    teacher_model = str(((llm_client.config or {}).get("api") or {}).get("model") or llm_client.mode or "teacher")
    manifest = {
        "dataset": cfg.dataset_name or cfg.values.get("dataset", {}).get("dataset_slug"),
        "source_stage": "05_scaled_envs_clean",
        "tool_protocol": "stage06",
        "tool_format_version": "openai_messages_v1",
        "teacher_model": teacher_model,
        "teacher_mode": llm_client.mode,
        "teacher_output_slug": teacher_slug,
        "trajectory_recipe": "oracle_gold_tool_trajectory_plus_autonomous_teacher_agent_trajectory",
        "oracle_gold_ratio_target": float(stage07_cfg.get("oracle_gold_ratio_target", 0.30)),
        "teacher_agent_ratio_target": 1.0 - float(stage07_cfg.get("oracle_gold_ratio_target", 0.30)),
        "user_feedback_enabled": bool(user_feedback),
        "split_policy": split_source,
        "env_splits": {name: len(items) for name, items in env_splits.items()},
        "num_samples": len(rows),
        "num_rejected": len(rejected),
        "splits": {name: len(items) for name, items in split_rows.items()},
        "trajectory_types": dict(Counter(row.get("trajectory_type") for row in rows)),
        "uses_create_test_file": sum(_sample_uses_tool(row, "create_test_file") for row in rows),
        "hidden_oracle_policy": "filter_and_quality_metadata_only; hidden oracle details are not included in messages",
    }

    write_jsonl(output_paths["trajectories"], rows)
    write_jsonl(output_paths["quality_report"], quality_report)
    for split_name, split_items in split_rows.items():
        write_jsonl(output_paths[f"split_{split_name}"], split_items)
    output_paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"trajectories": rows, "quality_report": quality_report, "splits": split_rows, "manifest": manifest}


def _tool_sft_outputs_complete(output_paths: dict[str, Path], *, expected_env_count: int) -> bool:
    required = [
        output_paths["trajectories"],
        output_paths["quality_report"],
        output_paths["manifest"],
        output_paths["split_train"],
        output_paths["split_dev"],
        output_paths["split_test"],
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        manifest = json.loads(output_paths["manifest"].read_text(encoding="utf-8"))
        rows = read_jsonl(output_paths["trajectories"])
        quality_report = read_jsonl(output_paths["quality_report"])
        split_total = sum(len(read_jsonl(output_paths[f"split_{name}"])) for name in ("train", "dev", "test"))
    except Exception:
        return False
    manifest_env_count = sum(int(value) for value in (manifest.get("env_splits") or {}).values())
    return (
        manifest_env_count == expected_env_count
        and int(manifest.get("num_samples", -1)) == len(rows) == split_total
        and int(manifest.get("num_rejected", -1)) == len(quality_report)
    )


def _generate_rows_for_envs(
    *,
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    split_name: str,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    stage07_cfg: dict[str, Any],
    teacher_slug: str,
    user_feedback: bool,
    rejected: list[dict[str, Any]],
    progress: Any | None = None,
    resume: bool = False,
    parallel_workers: int | None = None,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    worker_count = max(1, int(parallel_workers or 1))
    if llm_client.mode == "local" and worker_count > 1:
        tqdm.write("Stage07 workers is forced to 1 for local LLM mode to avoid concurrent model.generate calls.")
        worker_count = 1
    completed = _load_tool_sft_checkpoint(checkpoint_path, split_name) if resume and checkpoint_path is not None else {}
    results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]] = []
    pending: list[tuple[int, ExecutableEnvSpec]] = []
    for index, env in enumerate(environments):
        key = _tool_sft_checkpoint_key(split_name, env)
        if key in completed:
            item = completed[key]
            results.append((index, list(item.get("teacher_rows") or []), list(item.get("rejected") or [])))
        else:
            pending.append((index, env))
    if completed:
        tqdm.write(f"Stage07 resume: loaded {len(completed)} checkpoint rows for split={split_name}")
        if progress is not None:
            progress.update(len(results))
    if worker_count == 1:
        for index, env in pending:
            if progress is not None:
                progress.set_postfix_str(f"{split_name}:{env.env_id}", refresh=False)
            teacher, rejected_for_env = _generate_teacher_rows_for_env(
                env=env,
                cfg=cfg,
                llm_client=llm_client,
                tool_pool_cfg=tool_pool_cfg,
                budget_cfg=budget_cfg,
                agent_cfg=agent_cfg,
                user_feedback=user_feedback,
            )
            results.append((index, teacher, rejected_for_env))
            if resume and checkpoint_path is not None:
                _append_tool_sft_checkpoint(checkpoint_path, split_name, env, teacher, rejected_for_env)
            if progress is not None:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _generate_teacher_rows_for_env,
                    env=env,
                    cfg=cfg,
                    llm_client=llm_client,
                    tool_pool_cfg=tool_pool_cfg,
                    budget_cfg=budget_cfg,
                    agent_cfg=agent_cfg,
                    user_feedback=user_feedback,
                ): (index, env)
                for index, env in pending
            }
            for future in as_completed(futures):
                index, env = futures[future]
                teacher, rejected_for_env = future.result()
                results.append((index, teacher, rejected_for_env))
                if resume and checkpoint_path is not None:
                    _append_tool_sft_checkpoint(checkpoint_path, split_name, env, teacher, rejected_for_env)
                if progress is not None:
                    progress.update(1)
    teacher_rows: list[dict[str, Any]] = []
    rejected_start = len(rejected)
    for _, teacher, rejected_for_env in sorted(results, key=lambda item: item[0]):
        teacher_rows.extend(teacher)
        rejected.extend(rejected_for_env)
    rows = []
    if bool(stage07_cfg.get("include_oracle_gold_tool_trajectory", True)):
        oracle_count = _oracle_gold_count(
            teacher_count=len(teacher_rows),
            env_count=len(environments),
            ratio=float(stage07_cfg.get("oracle_gold_ratio_target", 0.30)),
        )
        for env in environments[:oracle_count]:
            rows.extend(_maybe_build_oracle_gold_tool_sample(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg, rejected))
    if bool(stage07_cfg.get("include_teacher_agent_trajectory", True)):
        rows.extend(teacher_rows)
    rows = _dedupe_samples(rows)
    for row in rows:
        row["split"] = split_name
        row.setdefault("metadata", {})["split_assigned_before_generation"] = True
    for row in rejected[rejected_start:]:
        row["split"] = split_name
    return rows


def _generate_teacher_rows_for_env(
    *,
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    user_feedback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    rows = _maybe_build_teacher_agent_sample(
        env,
        cfg,
        llm_client,
        tool_pool_cfg,
        budget_cfg,
        agent_cfg,
        rejected,
        user_feedback=user_feedback,
    )
    return rows, rejected


def _tool_sft_checkpoint_key(split_name: str, env: ExecutableEnvSpec) -> str:
    return f"{split_name}:{env.env_id}"


def _load_tool_sft_checkpoint(path: Path | None, split_name: str) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    completed: dict[str, dict[str, Any]] = {}
    prefix = f"{split_name}:"
    for row in read_jsonl(path):
        key = str(row.get("task_key") or "")
        if key.startswith(prefix):
            completed[key] = row
    return completed


def _append_tool_sft_checkpoint(
    path: Path,
    split_name: str,
    env: ExecutableEnvSpec,
    teacher_rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    append_jsonl(
        path,
        {
            "task_key": _tool_sft_checkpoint_key(split_name, env),
            "env_id": env.env_id,
            "split": split_name,
            "teacher_rows": teacher_rows,
            "rejected": rejected,
        },
    )


def _oracle_gold_count(*, teacher_count: int, env_count: int, ratio: float) -> int:
    if env_count <= 0 or ratio <= 0:
        return 0
    if teacher_count <= 0:
        return min(env_count, 1)
    ratio = min(max(ratio, 0.0), 0.9)
    return min(env_count, max(1, math.ceil(teacher_count * ratio / max(1e-6, 1.0 - ratio))))


def _maybe_build_oracle_gold_tool_sample(
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_code(env)
    if not reference.strip():
        rejected.append(_reject(env, "oracle_gold_tool_trajectory", "missing_reference_code"))
        return []
    runtime, messages = _new_runtime_and_messages(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg)
    _call_tool(messages, runtime, "get_task_context", {"window": 6000})
    _call_tool(messages, runtime, "validate_candidate_code", {"code": reference})
    public_test_plan = _oracle_public_test_plan(env)
    for file_item in _safe_public_test_files(public_test_plan.get("public_test_files") or []):
        _call_tool(messages, runtime, "create_test_file", file_item)
    _call_tool(
        messages,
        runtime,
        "run_custom_test",
        {"code": reference, "test_snippet": str(public_test_plan["public_test_snippet"]), "timeout_seconds": 5},
    )
    quality = _submit_and_quality(messages, runtime, reference, include_visible_failed_submit=False)
    if not quality["hidden_oracle_passed"]:
        rejected.append(_reject(env, "oracle_gold_tool_trajectory", "hidden_oracle_failed", quality))
        return []
    return [
        _sample(
            env=env,
            trajectory_type="oracle_gold_tool_trajectory",
            messages=messages,
            final_code=reference,
            quality=quality,
            metadata={
                "code_source": "stage05_scaled_executable_gold_code",
                "synthetic_tool_trace": True,
                "uses_hidden_oracle_for_filter_only": True,
                "public_test_policy": public_test_plan.get("policy"),
                "public_test_features": public_test_plan.get("features"),
            },
        )
    ]


def _oracle_public_test_plan(env: ExecutableEnvSpec) -> dict[str, Any]:
    features = _oracle_public_features(env)
    files = []
    if "file_io" in features:
        files.append(
            {
                "path": "agent_public_smoke_fixture.txt",
                "content": "id,value\nA,1\nB,2\n",
            }
        )
    return {
        "policy": "task_aware_public_smoke",
        "features": features,
        "public_test_files": files,
        "public_test_snippet": _oracle_public_smoke_test_snippet(env, features),
    }


def _oracle_public_features(env: ExecutableEnvSpec) -> list[str]:
    text = "\n".join(
        str(item or "")
        for item in [
            env.problem,
            env.user_prompt,
            env.context,
            env.signature,
            env.solution_form,
            " ".join(env.output_requirements or []),
        ]
    ).lower()
    features = []
    if env.resource_manifest or env.resource_files:
        features.append("resource_file")
    file_markers = (
        "file",
        "path",
        "read_text",
        "read_csv",
        "csv",
        "tsv",
        "fasta",
        "fastq",
        "json",
        "infile",
        "outfile",
        "write",
        "open(",
        "artifact",
    )
    if any(marker in text for marker in file_markers):
        features.append("file_io")
    if any(marker in text for marker in ("stdout", "print(", "print ", "output", "return only")):
        features.append("output_format")
    if any(marker in text for marker in ("dataframe", "pandas", "columns", "row", "table")):
        features.append("tabular")
    if any(marker in text for marker in ("float", "numeric", "tolerance", "approximately", "mean", "score", "probability")):
        features.append("numeric")
    if not features:
        features.append("plain_callable")
    return list(dict.fromkeys(features))


def _oracle_public_smoke_test_snippet(env: ExecutableEnvSpec, features: list[str]) -> str:
    target = _signature_target_name(env.signature or "")
    lines = [
        "# Public smoke test generated from visible task features only.",
        "assert True",
    ]
    if target:
        lines.append(f"assert callable(globals().get({target!r})), {target!r}")
    if "file_io" in features:
        lines.extend(
            [
                "from pathlib import Path",
                "assert Path('agent_public_smoke_fixture.txt').exists()",
                "assert Path('agent_public_smoke_fixture.txt').read_text().startswith('id,value')",
            ]
        )
    if "tabular" in features:
        lines.append("# Tabular task signal: candidate imports and definitions loaded without errors.")
    if "output_format" in features:
        lines.append("# Output-format task signal: final formatting is checked by submit_final_code oracle, not leaked here.")
    if "numeric" in features:
        lines.append("# Numeric task signal: public smoke avoids hidden expected numeric values.")
    return "\n".join(lines) + "\n"


def _signature_target_name(signature: str) -> str:
    match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", str(signature or ""))
    return match.group(1) if match else ""


def _maybe_build_teacher_agent_sample(
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    rejected: list[dict[str, Any]],
    user_feedback: bool = False,
) -> list[dict[str, Any]]:
    runtime, messages = _new_runtime_and_messages(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg)
    tools = stage06_tool_schemas(tool_pool_cfg)
    level = _env_level(env)
    stage06_budget_cfg = (budget_cfg or {}).get("stage06_tool_agent", {})
    max_turns = int((stage06_budget_cfg.get("max_turns_by_level") or {}).get(level, agent_cfg.get("max_turns", 8)))
    final_code = ""
    for turn in range(max_turns):
        response = llm_client.complete_with_tools(
            task_name="stage07_teacher_agent_trajectory",
            messages=messages,
            tools=tools,
            context={"env_id": env.env_id, "difficulty": env.difficulty.model_dump() if env.difficulty else {}, "turn": turn},
            mock_builder=lambda context: _mock_teacher_agent_message(env, int(context.get("turn", 0))),
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
                if user_feedback and call.name == "submit_final_code":
                    messages.append({"role": "user", "content": _final_code_repair_prompt(result)})
            if runtime.terminated:
                break
            continue
        parsed = _parse_teacher_final_response(response.content)
        final_code = str(parsed.get("final_code") or "")
        if final_code:
            result = _submit_final_code_as_tool(messages, runtime, final_code, f"implicit_submit_{turn}")
            if runtime.terminated:
                break
            if user_feedback:
                messages.append({"role": "user", "content": _final_code_repair_prompt(result)})
            continue
        messages.append(response.raw_message if response.raw_message else {"role": "assistant", "content": response.content or ""})
        messages.append(
            {
                "role": "user",
                "content": (
                    "No final_code or tool call was provided. Continue using the available public tools, "
                    "or submit final executable Python code when ready."
                ),
            }
        )
    if not runtime.terminated:
        runtime.mark_no_final_code("MAX_TURNS_WITHOUT_VALID_FINAL_CODE")
    quality = _quality_from_runtime(runtime)
    if not quality["hidden_oracle_passed"]:
        rejected.append(_reject(env, "teacher_agent_trajectory", _teacher_reject_reason(quality), quality))
        return []
    return [
        _sample(
            env=env,
            trajectory_type="teacher_agent_trajectory",
            messages=messages,
            final_code=runtime.final_code or final_code,
            quality=quality,
            metadata={
                "teacher_source": llm_client.mode,
                "autonomous_tool_decisions": True,
                "uses_hidden_oracle_for_filter_only": True,
            },
        )
    ]


def _mock_teacher_agent_message(env: ExecutableEnvSpec, turn: int) -> dict[str, Any]:
    reference = _reference_code(env)
    if turn == 0:
        return _tool_call_message("mock_call_get_task_context", "get_task_context", {"window": 6000})
    if turn == 1:
        return _tool_call_message("mock_call_validate_candidate_code", "validate_candidate_code", {"code": reference})
    return _tool_call_message("mock_call_submit_final_code", "submit_final_code", {"code": reference})


def _tool_call_message(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


def _parse_teacher_final_response(content: str) -> dict[str, Any]:
    try:
        parsed = parse_json_payload(content)
    except Exception:
        return {"final_code": content, "notes": ["non_json_final_response"]}
    if not isinstance(parsed, dict):
        return {"final_code": str(content or ""), "notes": ["unexpected_final_response_shape"]}
    return parsed


def _quality_from_runtime(runtime: ToolRuntime) -> dict[str, Any]:
    eval_report = runtime.final_eval or {}
    case_reports = eval_report.get("case_reports") or []
    passed_cases = sum(1 for row in case_reports if row.get("passed"))
    return {
        "compile_passed": bool(eval_report.get("compile_passed")),
        "execution_passed": bool(eval_report.get("execution_passed")),
        "hidden_oracle_passed": bool(eval_report.get("compile_passed") and eval_report.get("execution_passed")),
        "evaluation_case_source": str(eval_report.get("evaluation_case_source") or ""),
        "passed_cases": passed_cases,
        "total_cases": len(case_reports),
        "failure_reasons": list(eval_report.get("failure_reasons") or []),
        "tool_budget_used": {"total": runtime.total_calls, "per_tool": dict(runtime.call_counts)},
    }


def _teacher_reject_reason(quality: dict[str, Any]) -> str:
    if quality.get("evaluation_case_source") in {"none", ""}:
        return "no_valid_final_submission"
    if not quality.get("compile_passed"):
        return "compile_failed"
    return "hidden_oracle_failed"


def _maybe_build_clean_sample(
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_code(env)
    if not reference.strip():
        rejected.append(_reject(env, "clean_gold", "missing_reference_code"))
        return []
    teacher = _teacher_clean_solution(env, llm_client, prompt_runner, fallback_code=reference)
    final_code = str(teacher.get("final_code") or reference)
    runtime, messages = _new_runtime_and_messages(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg)
    _call_tool(messages, runtime, "get_task_context", {"window": 6000})
    _call_tool(messages, runtime, "validate_candidate_code", {"code": final_code})
    _maybe_run_public_test(messages, runtime, final_code, teacher)
    quality = _submit_and_quality(messages, runtime, final_code, include_visible_failed_submit=True)
    if not quality["hidden_oracle_passed"]:
        rejected.append(_reject(env, "clean_gold", "hidden_oracle_failed", quality))
        return []
    return [
        _sample(
            env=env,
            trajectory_type="clean_gold_teacher",
            messages=messages,
            final_code=final_code,
            quality=quality,
            metadata={
                "teacher_source": llm_client.mode,
                "notes": teacher.get("notes") or [],
                "uses_hidden_oracle_for_filter_only": True,
            },
        )
    ]


def _build_bug_injection_samples(
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    stage07_cfg: dict[str, Any],
    teacher_slug: str,
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_code(env)
    if not reference.strip():
        return []
    samples = []
    for bug_type in _bug_types_for_env(env, stage07_cfg):
        buggy = _inject_bug(reference, env, bug_type)
        if not buggy or buggy == reference:
            rejected.append(_reject(env, f"bug_injection:{bug_type}", "bug_injection_noop"))
            continue
        runtime, messages = _new_runtime_and_messages(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg)
        _call_tool(messages, runtime, "get_task_context", {"window": 6000})
        feedback = _collect_bug_feedback(messages, runtime, buggy, bug_type)
        if not _has_useful_failure(feedback):
            rejected.append(_reject(env, f"bug_injection:{bug_type}", "bug_not_exposed_by_public_tools", {"feedback": feedback}))
            continue
        repaired = _teacher_repair(env, llm_client, prompt_runner, bug_type=bug_type, buggy_code=buggy, feedback=feedback, fallback_code=reference)
        final_code = str(repaired.get("final_code") or reference)
        if final_code.strip() == buggy.strip():
            rejected.append(_reject(env, f"bug_injection:{bug_type}", "repair_same_as_buggy"))
            continue
        quality = _submit_and_quality(messages, runtime, final_code, include_visible_failed_submit=True)
        if not quality["hidden_oracle_passed"]:
            rejected.append(_reject(env, f"bug_injection:{bug_type}", "hidden_oracle_failed", quality))
            continue
        samples.append(
            _sample(
                env=env,
                trajectory_type="bug_injection_repair",
                messages=messages,
                final_code=final_code,
                quality=quality,
                bug_type=bug_type,
                metadata={
                    "teacher_source": llm_client.mode,
                    "public_feedback_tools": [item["tool_name"] for item in feedback],
                    "uses_hidden_oracle_for_filter_only": True,
                    "notes": repaired.get("notes") or [],
                },
            )
        )
    return samples


def _maybe_build_strong_flawed_sample(
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    stage07_cfg: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not bool(stage07_cfg.get("include_strong_flawed", True)):
        return []
    reference = _reference_code(env)
    if not reference.strip():
        return []
    flawed = _teacher_flawed_draft(env, llm_client, prompt_runner, fallback_code=reference)
    buggy = str(flawed.get("buggy_code") or "")
    bug_type = str(flawed.get("bug_type") or "strong_model_flawed_draft")
    if not buggy.strip() or buggy.strip() == reference.strip():
        rejected.append(_reject(env, "strong_flawed_draft", "missing_or_unmodified_buggy_code"))
        return []
    runtime, messages = _new_runtime_and_messages(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg)
    _call_tool(messages, runtime, "get_task_context", {"window": 6000})
    feedback = _collect_bug_feedback(messages, runtime, buggy, bug_type, public_test_plan=flawed)
    if not _has_useful_failure(feedback):
        rejected.append(_reject(env, "strong_flawed_draft", "bug_not_exposed_by_public_tools", {"feedback": feedback}))
        return []
    repaired = _teacher_repair(env, llm_client, prompt_runner, bug_type=bug_type, buggy_code=buggy, feedback=feedback, fallback_code=reference)
    final_code = str(repaired.get("final_code") or reference)
    quality = _submit_and_quality(messages, runtime, final_code, include_visible_failed_submit=True)
    if not quality["hidden_oracle_passed"]:
        rejected.append(_reject(env, "strong_flawed_draft", "hidden_oracle_failed", quality))
        return []
    return [
        _sample(
            env=env,
            trajectory_type="strong_model_flawed_draft_repair",
            messages=messages,
            final_code=final_code,
            quality=quality,
            bug_type=bug_type,
            metadata={
                "teacher_source": llm_client.mode,
                "public_feedback_tools": [item["tool_name"] for item in feedback],
                "uses_hidden_oracle_for_filter_only": True,
                "notes": repaired.get("notes") or [],
            },
        )
    ]


def _mine_stage06_samples(
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    stage07_cfg: dict[str, Any],
    teacher_slug: str,
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not bool(stage07_cfg.get("include_mined_stage06", True)):
        return []
    env_by_id = {env.env_id: env for env in environments}
    max_samples = int(stage07_cfg.get("max_mined_stage06_samples", max(1, len(environments) // 2)))
    mined = []
    for run_path in sorted((cfg.output_dirs["result"] / "06").glob("*/agent_runs.jsonl")):
        if run_path.parent.name != teacher_slug:
            continue
        trace_path = run_path.parent / "agent_traces.jsonl"
        eval_path = run_path.parent / "agent_eval_report.jsonl"
        traces = {row.get("env_id"): row for row in read_jsonl(trace_path)}
        evals = {row.get("env_id"): row for row in read_jsonl(eval_path)}
        for run in read_jsonl(run_path):
            if len(mined) >= max_samples:
                return mined
            env = env_by_id.get(str(run.get("env_id") or ""))
            final_code = str(run.get("final_code") or "")
            if env is None or not final_code.strip() or not bool(run.get("passed")):
                continue
            trace = traces.get(env.env_id, {})
            eval_row = evals.get(env.env_id, {})
            if not _trace_has_public_failure(trace):
                continue
            runtime, messages = _new_runtime_and_messages(env, cfg, tool_pool_cfg, budget_cfg, agent_cfg)
            _call_tool(messages, runtime, "get_task_context", {"window": 6000})
            _call_tool(messages, runtime, "validate_candidate_code", {"code": final_code})
            quality = _submit_and_quality(messages, runtime, final_code, include_visible_failed_submit=True)
            if not quality["hidden_oracle_passed"]:
                rejected.append(_reject(env, "real_stage06_mined_trace", "hidden_oracle_failed", quality))
                continue
            mined.append(
                _sample(
                    env=env,
                    trajectory_type="real_stage06_mined_trace_reconstructed",
                    messages=messages,
                    final_code=final_code,
                    quality=quality,
                    metadata={
                        "source_run_dir": run_path.parent.name,
                        "source_trace_reconstructed": True,
                        "source_eval_failure_reasons": eval_row.get("failure_reasons") or [],
                        "uses_hidden_oracle_for_filter_only": True,
                    },
                )
            )
    return mined


def _new_runtime_and_messages(
    env: ExecutableEnvSpec,
    cfg: AppConfig,
    tool_pool_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
) -> tuple[ToolRuntime, list[dict[str, Any]]]:
    level = _env_level(env)
    stage06_budget_cfg = (budget_cfg or {}).get("stage06_tool_agent", {})
    runtime = ToolRuntime(
        env,
        cfg,
        budget=_budget_for_level(level=level, agent_cfg=agent_cfg, stage06_budget_cfg=stage06_budget_cfg),
        allowed_tools=stage06_tool_names(tool_pool_cfg),
        submit_excluded_from_total=bool(stage06_budget_cfg.get("submit_final_code_excluded_from_total", True)),
    )
    return runtime, [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": _public_user_prompt(env)}]


def _call_tool(
    messages: list[dict[str, Any]],
    runtime: ToolRuntime,
    name: str,
    arguments: dict[str, Any],
    *,
    append_terminated_result: bool = False,
) -> dict[str, Any]:
    call_id = f"call_{len(messages)}_{name}"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ],
        }
    )
    result = runtime.execute(name, arguments)
    if append_terminated_result or not result.get("terminated"):
        messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(result, ensure_ascii=False)})
    return result


def _submit_and_quality(
    messages: list[dict[str, Any]],
    runtime: ToolRuntime,
    final_code: str,
    *,
    include_visible_failed_submit: bool,
    user_feedback: bool = False,
) -> dict[str, Any]:
    result = _call_tool(messages, runtime, "submit_final_code", {"code": final_code}, append_terminated_result=False)
    if user_feedback and not result.get("terminated") and include_visible_failed_submit:
        messages.append(
            {
                "role": "user",
                "content": _final_code_repair_prompt(result),
            }
        )
    eval_report = runtime.final_eval or {}
    if not eval_report:
        return {
            "compile_passed": False,
            "execution_passed": False,
            "hidden_oracle_passed": False,
            "evaluation_case_source": "preflight_failed",
            "passed_cases": 0,
            "total_cases": 0,
            "failure_reasons": list(result.get("errors") or [result.get("error") or "SUBMIT_PREFLIGHT_FAILED"]),
            "submit_result": {key: value for key, value in result.items() if key != "validation"},
            "tool_budget_used": {"total": runtime.total_calls, "per_tool": dict(runtime.call_counts)},
        }
    case_reports = eval_report.get("case_reports") or []
    passed_cases = sum(1 for row in case_reports if row.get("passed"))
    return {
        "compile_passed": bool(eval_report.get("compile_passed")),
        "execution_passed": bool(eval_report.get("execution_passed")),
        "hidden_oracle_passed": bool(eval_report.get("compile_passed") and eval_report.get("execution_passed")),
        "evaluation_case_source": str(eval_report.get("evaluation_case_source") or ""),
        "passed_cases": passed_cases,
        "total_cases": len(case_reports),
        "failure_reasons": list(eval_report.get("failure_reasons") or []),
        "tool_budget_used": {"total": runtime.total_calls, "per_tool": dict(runtime.call_counts)},
    }


def _collect_bug_feedback(
    messages: list[dict[str, Any]],
    runtime: ToolRuntime,
    buggy_code: str,
    bug_type: str,
    public_test_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    feedback = []
    validation = _call_tool(messages, runtime, "validate_candidate_code", {"code": buggy_code})
    feedback.append({"tool_name": "validate_candidate_code", "result": validation})
    if not validation.get("ok"):
        return feedback
    plan = public_test_plan or _default_public_test_plan(bug_type)
    for file_item in _safe_public_test_files(plan.get("public_test_files") or []):
        created = _call_tool(messages, runtime, "create_test_file", file_item)
        feedback.append({"tool_name": "create_test_file", "result": created})
    snippet = str(plan.get("public_test_snippet") or "").strip()
    if not snippet:
        snippet = _default_public_test_plan(bug_type).get("public_test_snippet", "")
    if snippet.strip():
        result = _call_tool(messages, runtime, "run_custom_test", {"code": buggy_code, "test_snippet": snippet, "timeout_seconds": 5})
        feedback.append({"tool_name": "run_custom_test", "result": result})
    return feedback


def _maybe_run_public_test(messages: list[dict[str, Any]], runtime: ToolRuntime, final_code: str, teacher: dict[str, Any]) -> None:
    for file_item in _safe_public_test_files(teacher.get("public_test_files") or []):
        _call_tool(messages, runtime, "create_test_file", file_item)
    snippet = str(teacher.get("public_test_snippet") or "").strip()
    if snippet:
        _call_tool(messages, runtime, "run_custom_test", {"code": final_code, "test_snippet": snippet, "timeout_seconds": 5})


def _teacher_clean_solution(
    env: ExecutableEnvSpec,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    *,
    fallback_code: str,
) -> dict[str, Any]:
    if llm_client.mode in {"mock", "local"}:
        return {"final_code": fallback_code, "notes": ["mock clean trajectory uses Stage05 reference code"], "public_test_files": [], "public_test_snippet": ""}
    prompt = prompt_runner.render("tool_sft_clean_teacher.jinja", **_prompt_vars(env))
    response = llm_client.complete_json(
        task_name="stage07_tool_sft_clean_teacher",
        prompt=prompt,
        context={"env_id": env.env_id},
        mock_builder=lambda context: {"final_code": fallback_code, "notes": ["mock clean trajectory"], "public_test_files": [], "public_test_snippet": ""},
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    payload.setdefault("final_code", fallback_code)
    return payload


def _teacher_flawed_draft(
    env: ExecutableEnvSpec,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    *,
    fallback_code: str,
) -> dict[str, Any]:
    if llm_client.mode in {"mock", "local"}:
        return {
            "bug_type": "unavailable_import",
            "buggy_code": "import my_module\n" + fallback_code,
            "public_test_files": [{"path": "data/public_probe.txt", "content": "probe\n"}],
            "public_test_snippet": "from pathlib import Path\nassert Path('data/public_probe.txt').read_text() == 'probe\\n'\n",
            "notes": ["mock flawed draft adds an unavailable import"],
        }
    prompt = prompt_runner.render("tool_sft_flawed_draft_teacher.jinja", **_prompt_vars(env))
    response = llm_client.complete_json(
        task_name="stage07_tool_sft_flawed_draft_teacher",
        prompt=prompt,
        context={"env_id": env.env_id},
        mock_builder=lambda context: {},
    )
    return response.payload if isinstance(response.payload, dict) else {}


def _teacher_repair(
    env: ExecutableEnvSpec,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    *,
    bug_type: str,
    buggy_code: str,
    feedback: list[dict[str, Any]],
    fallback_code: str,
) -> dict[str, Any]:
    if llm_client.mode in {"mock", "local"}:
        return {"final_code": fallback_code, "notes": [f"mock repair fixes {bug_type} with Stage05 reference code"]}
    prompt = prompt_runner.render(
        "tool_sft_repair_teacher.jinja",
        **_prompt_vars(env),
        bug_type=bug_type,
        buggy_code=buggy_code,
        tool_feedback_json=json.dumps(_compact_feedback(feedback), ensure_ascii=False, indent=2),
    )
    response = llm_client.complete_json(
        task_name="stage07_tool_sft_repair_teacher",
        prompt=prompt,
        context={"env_id": env.env_id, "bug_type": bug_type},
        mock_builder=lambda context: {"final_code": fallback_code, "notes": ["mock repair"]},
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    payload.setdefault("final_code", fallback_code)
    return payload


def _prompt_vars(env: ExecutableEnvSpec) -> dict[str, Any]:
    return {
        "public_task": env.user_prompt or env.problem,
        "signature": env.signature or "",
        "context": env.context or "",
        "resource_manifest_json": json.dumps(env.resource_manifest or [{"path": path} for path in env.resource_files], ensure_ascii=False, indent=2),
    }


def _reference_code(env: ExecutableEnvSpec) -> str:
    return str(env.scaled_executable_gold_code or env.scaled_gold_solution or env.gold_solution or "")


def _bug_types_for_env(env: ExecutableEnvSpec, stage07_cfg: dict[str, Any]) -> list[str]:
    configured = stage07_cfg.get("bug_types_by_level") or {}
    level = _env_level(env)
    if level in configured and isinstance(configured[level], list):
        return [str(item) for item in configured[level]]
    count_by_level = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    cap = int((stage07_cfg.get("bug_repairs_per_level") or {}).get(level, count_by_level.get(level, 2)))
    candidates = ["missing_target", "unavailable_import", "markdown_wrapped", "fixture_backed_public_test"]
    if level in {"M2", "M3", "M4"}:
        candidates.insert(1, "seed_regression_prompt_trap")
    return candidates[: max(0, cap)]


def _inject_bug(reference: str, env: ExecutableEnvSpec, bug_type: str) -> str:
    if bug_type == "missing_target":
        target = _target_name(env.signature or "")
        if target:
            return re.sub(rf"(\bdef\s+){re.escape(target)}(\s*\()", rf"\1{target}_buggy\2", reference, count=1)
        return reference + "\n# missing_target injection could not find target\n"
    if bug_type == "unavailable_import":
        return "import my_module\n" + reference
    if bug_type == "markdown_wrapped":
        return "```python\n" + reference + "\n```"
    if bug_type == "fixture_backed_public_test":
        return "import my_module\n" + reference
    if bug_type == "seed_regression_prompt_trap":
        target = _target_name(env.signature or "")
        if target:
            return re.sub(rf"(\bdef\s+){re.escape(target)}(\s*\([^)]*\)\s*:)", rf"\1{target}\2\n    return None", reference, count=1)
        return reference + "\nraise RuntimeError('seed regression broken')\n"
    return reference


def _target_name(signature: str) -> str:
    match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature or "")
    return match.group(1) if match else ""


def _default_public_test_plan(bug_type: str) -> dict[str, Any]:
    if bug_type == "fixture_backed_public_test":
        return {
            "public_test_files": [{"path": "data/public_probe.txt", "content": "probe\n"}],
            "public_test_snippet": "from pathlib import Path\nassert Path('data/public_probe.txt').read_text() == 'probe\\n'\n",
        }
    return {"public_test_files": [], "public_test_snippet": "assert True\n"}


def _safe_public_test_files(files: list[Any]) -> list[dict[str, str]]:
    safe = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        content = str(item.get("content") or "")
        if path and len(content.encode("utf-8")) <= 65536:
            safe.append({"path": path, "content": content})
    return safe[:3]


def _has_useful_failure(feedback: list[dict[str, Any]]) -> bool:
    for item in feedback:
        result = item.get("result") or {}
        if result.get("ok") is False:
            return True
        if result.get("terminated") is False and result.get("preflight_passed") is False:
            return True
    return False


def _trace_has_public_failure(trace: dict[str, Any]) -> bool:
    for step in trace.get("tool_trace") or []:
        name = str(step.get("tool_name") or "")
        result = step.get("result") or {}
        if name in {"validate_candidate_code", "run_custom_test", "submit_final_code"} and result.get("ok") is False:
            return True
    return False


def _sample(
    *,
    env: ExecutableEnvSpec,
    trajectory_type: str,
    messages: list[dict[str, Any]],
    final_code: str,
    quality: dict[str, Any],
    bug_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = stable_hash({"env_id": env.env_id, "trajectory_type": trajectory_type, "bug_type": bug_type, "messages": messages})[:16]
    return {
        "sample_id": f"{env.env_id}_{trajectory_type}_{sample_id}",
        "env_id": env.env_id,
        "original_task_id": env.original_task_id,
        "level": _env_level(env),
        "trajectory_type": trajectory_type,
        "bug_type": bug_type,
        "tool_protocol": "stage06",
        "messages": messages,
        "final_code": final_code,
        "tool_budget": quality.get("tool_budget_used", {}),
        "quality": quality,
        "metadata": metadata or {},
    }


def _reject(env: ExecutableEnvSpec, source: str, reason: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"env_id": env.env_id, "original_task_id": env.original_task_id, "source": source, "reason": reason, "detail": detail or {}}


def _dedupe_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get("env_id"), row.get("trajectory_type"), row.get("bug_type"), stable_hash(row.get("final_code", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _split_envs_by_original_task(environments: list[ExecutableEnvSpec], splits_cfg: dict[str, Any]) -> dict[str, list[ExecutableEnvSpec]]:
    grouped: dict[str, list[ExecutableEnvSpec]] = defaultdict(list)
    for env in environments:
        grouped[str(env.original_task_id or env.env_id)].append(env)
    groups = seeded_shuffle(list(grouped), int(splits_cfg.get("seed", 1337)))
    n = len(groups)
    train_count = int(n * float(splits_cfg.get("train", 0.7)))
    dev_count = int(n * float(splits_cfg.get("dev", 0.1)))
    if n >= 3 and dev_count == 0:
        dev_count = 1
    test_count = max(0, n - train_count - dev_count)
    if n >= 3 and test_count == 0:
        test_count = 1
        train_count = max(1, train_count - 1)
    if n and train_count == 0:
        train_count = 1
    train_end = min(train_count, n)
    dev_end = min(train_end + dev_count, n)
    split_keys = {
        "train": set(groups[:train_end]),
        "dev": set(groups[train_end:dev_end]),
        "test": set(groups[dev_end:]),
    }
    result: dict[str, list[ExecutableEnvSpec]] = {"train": [], "dev": [], "test": []}
    for split_name, keys in split_keys.items():
        for key in keys:
            result[split_name].extend(grouped[key])
    return result


def _build_quality_report(rows: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row.get("trajectory_type") for row in rows)
    reject_counts = Counter(row.get("reason") for row in rejected)
    return [
        {
            "summary": {
                "accepted_samples": len(rows),
                "rejected_samples": len(rejected),
                "trajectory_types": dict(counts),
                "reject_reasons": dict(reject_counts),
                "samples_with_create_test_file": sum(_sample_uses_tool(row, "create_test_file") for row in rows),
                "hidden_oracle_passed": sum(bool((row.get("quality") or {}).get("hidden_oracle_passed")) for row in rows),
            }
        },
        *rejected,
    ]


def _sample_uses_tool(row: dict[str, Any], tool_name: str) -> bool:
    for message in row.get("messages") or []:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == tool_name:
                return True
    return False


def _compact_feedback(feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in feedback:
        result = dict(item.get("result") or {})
        for key in ("stdout_tail", "stderr_tail", "traceback_tail"):
            if key in result:
                result[key] = str(result[key])[-1000:]
        compact.append({"tool_name": item.get("tool_name"), "result": result})
    return compact


def _load_dataset_yaml(cfg: AppConfig, filename: str) -> dict[str, Any]:
    path = cfg.dataset_config_path(filename)
    from medenvscale.utils import load_yaml

    return load_yaml(path) if path.exists() else {}


def _load_budget_cfg(cfg: AppConfig) -> dict[str, Any]:
    from medenvscale.utils import load_yaml

    path = cfg.dataset_config_path_with_fallback("m_level_budgets_4axis.yaml", "m_level_budgets_7axis.yaml")
    return load_yaml(path) if path.exists() else {}
