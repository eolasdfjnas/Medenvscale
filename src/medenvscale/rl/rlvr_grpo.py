from __future__ import annotations

import json
import math
import re
from inspect import signature
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm

from medenvscale.agent.runner import _budget_for_level, _env_level, _public_user_prompt, _system_prompt, run_tool_agent_for_env
from medenvscale.agent.tool_runtime import _agent_python_bin, _evaluation_cases_for_env
from medenvscale.agent.tool_runtime import ToolRuntime
from medenvscale.agent.tool_schemas import stage06_tool_names
from medenvscale.config import AppConfig
from medenvscale.distributed import barrier, distributed_metadata, global_rank, is_distributed, is_main_process, setup_torch_distributed_device
from medenvscale.llm import LLMClient
from medenvscale.llm.json_repair import parse_json_payload
from medenvscale.rubrics import score_requirement_rubrics
from medenvscale.scaling.case_execution import run_scaled_gold_on_validated_oracle_cases
from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.train.checkpoints import latest_trainer_checkpoint
from medenvscale.utils import append_jsonl, load_yaml, read_jsonl, slugify, write_jsonl


def run_stage09_rlvr_grpo(
    *,
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    eval_environments: list[ExecutableEnvSpec] | None = None,
    llm_client: LLMClient,
    output_dir: Path,
    experiment_dir: Path,
    limit: int | None = None,
    rollout_only: bool = True,
    train: bool = False,
    collect_rollouts: bool = False,
    user_feedback: bool = False,
    use_rubric_reward: bool | None = None,
    resume: bool = False,
    use_existing_rollouts: bool = False,
) -> dict[str, Any]:
    stage_cfg = cfg.values.get("stage09_rlvr_grpo", {}) or {}
    eval_environments = list(eval_environments or [])
    if use_rubric_reward is None:
        use_rubric_reward = bool((stage_cfg.get("rubric_reward") or {}).get("enabled", False))
    rollouts_per_env = max(1, int(stage_cfg.get("rollouts_per_env", 4)))
    selected = environments[:limit] if limit is not None else environments
    eval_selected = eval_environments[:limit] if limit is not None else eval_environments
    policy_slug = _policy_slug(stage_cfg=stage_cfg, llm_client=llm_client)
    model_output_dir = output_dir / policy_slug
    model_experiment_dir = experiment_dir / "tool_rl_grpo_lora" / policy_slug
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_experiment_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = model_output_dir / "rl_rollouts.jsonl"
    reward_path = model_output_dir / "reward_report.jsonl"
    summary_path = model_output_dir / "summary.json"
    manifest_path = model_experiment_dir / "train_manifest.json"
    adapter_dir = model_experiment_dir / "adapter"
    if resume and train and not rollout_only and (adapter_dir / "adapter_config.json").exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        return {
            "rollouts": read_jsonl(rollout_path),
            "reward_report": read_jsonl(reward_path),
            "summary": summary,
            "manifest": manifest,
        }

    use_existing_rollouts = bool(use_existing_rollouts or (resume and rollout_path.exists() and reward_path.exists()))
    should_collect_rollouts = bool((rollout_only or collect_rollouts) and not use_existing_rollouts)
    if should_collect_rollouts:
        if is_main_process():
            existing_rollouts = _dedupe_rollout_rows(read_jsonl(rollout_path)) if resume and rollout_path.exists() else []
            if not resume:
                write_jsonl(rollout_path, [])
            rollout_rows = _collect_rollouts(
                cfg=cfg,
                environments=selected,
                llm_client=llm_client,
                stage_cfg=stage_cfg,
                rollouts_per_env=rollouts_per_env,
                user_feedback=user_feedback,
                existing_rows=existing_rollouts,
                checkpoint_path=rollout_path,
            )
            rollout_rows = _dedupe_rollout_rows(rollout_rows)
            reward_rows = [
                _reward_row(
                    row,
                    reward_cfg=stage_cfg.get("reward") or {},
                    use_rubric_reward=bool(use_rubric_reward),
                )
                for row in rollout_rows
            ]
            reward_rows = attach_group_advantages(reward_rows)
            rollout_rows = _attach_rewards_to_rollouts(rollout_rows, reward_rows)
            write_jsonl(rollout_path, rollout_rows)
            write_jsonl(reward_path, reward_rows)
        barrier()
        if not is_main_process():
            rollout_rows = read_jsonl(rollout_path) if rollout_path.exists() else []
            reward_rows = read_jsonl(reward_path) if reward_path.exists() else []
    elif use_existing_rollouts and rollout_path.exists():
        rollout_rows = read_jsonl(rollout_path)
        if reward_path.exists():
            reward_rows = read_jsonl(reward_path)
        else:
            reward_rows = [
                _reward_row(
                    row,
                    reward_cfg=stage_cfg.get("reward") or {},
                    use_rubric_reward=bool(use_rubric_reward),
                )
                for row in rollout_rows
            ]
            reward_rows = attach_group_advantages(reward_rows)
            rollout_rows = _attach_rewards_to_rollouts(rollout_rows, reward_rows)
    else:
        rollout_rows = []
        reward_rows = []
    summary = build_rlvr_summary(reward_rows=reward_rows, stage_cfg=stage_cfg, policy_slug=policy_slug, train_requested=train)
    summary["collect_rollouts"] = should_collect_rollouts
    summary["split"] = stage_cfg.get("split")
    summary["split_source"] = stage_cfg.get("split_source")
    summary["split_env_count"] = stage_cfg.get("split_env_count")
    summary.update(distributed_metadata())

    if is_main_process():
        if not should_collect_rollouts:
            write_jsonl(rollout_path, rollout_rows)
            write_jsonl(reward_path, reward_rows)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    barrier()

    train_result: dict[str, Any] = {
        "status": "rollout_only" if rollout_only or not train else "pending",
        "output_dir": str(model_experiment_dir),
        "adapter_dir": "",
        "num_train_rollouts": 0,
    }
    if train and not rollout_only:
        train_result = _run_trl_grpo_update(
            cfg=cfg,
            stage_cfg=stage_cfg,
            environments=selected,
            eval_environments=eval_selected,
            output_dir=model_experiment_dir,
            use_rubric_reward=bool(use_rubric_reward),
            resume=resume,
        )
    manifest = {
        "algorithm": "rlvr_grpo",
        "trainer": "trl_grpo",
        "trainer_adapter_mode": "interactive_tools_final_code_reward",
        "tool_format_version": "simplified_tool_json_v1",
        "policy_slug": policy_slug,
        "base_model": stage_cfg.get("base_model"),
        "sft_adapter": stage_cfg.get("sft_adapter"),
        "rollouts_per_env": rollouts_per_env,
        "collect_rollouts": should_collect_rollouts,
        "use_existing_rollouts": use_existing_rollouts,
        "split": stage_cfg.get("split"),
        "split_source": stage_cfg.get("split_source"),
        "split_env_count": stage_cfg.get("split_env_count"),
        "eval_split": stage_cfg.get("eval_split"),
        "eval_split_source": stage_cfg.get("eval_split_source"),
        "eval_split_env_count": stage_cfg.get("eval_split_env_count"),
        "eval_steps": stage_cfg.get("eval_steps"),
        "use_rubric_reward": bool(use_rubric_reward),
        "num_envs": len(selected),
        "num_rollouts": len(rollout_rows),
        "rollout_path": str(rollout_path),
        "reward_path": str(reward_path),
        "summary_path": str(summary_path),
        "train": train_result,
        **distributed_metadata(),
    }
    if is_main_process():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    barrier()
    return {
        "rollouts": rollout_rows,
        "reward_report": reward_rows,
        "summary": summary,
        "manifest": manifest,
    }


def _collect_rollouts(
    *,
    cfg: AppConfig,
    environments: list[ExecutableEnvSpec],
    llm_client: LLMClient,
    stage_cfg: dict[str, Any],
    rollouts_per_env: int,
    user_feedback: bool,
    existing_rows: list[dict[str, Any]] | None = None,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    tool_pool_cfg = load_yaml(cfg.dataset_config_path("tool_pool.yaml"))
    budget_cfg = load_yaml(
        cfg.dataset_config_path_with_fallback("m_level_budgets_4axis.yaml", "m_level_budgets_7axis.yaml")
    )
    stage06_cfg = cfg.values.get("stage06", {}) or {}
    agent_cfg = stage06_cfg.get("tool_agent", stage06_cfg) or {}
    expected_ids = {f"{env.env_id}::r{rollout_index}" for env in environments for rollout_index in range(rollouts_per_env)}
    rows = [row for row in _dedupe_rollout_rows(existing_rows or []) if str(row.get("rollout_id") or "") in expected_ids]
    completed_ids = {str(row.get("rollout_id") or "") for row in rows}
    progress = tqdm(total=len(environments) * rollouts_per_env, desc="Stage09 RLVR Rollout", unit="rollout", leave=True)
    if completed_ids:
        progress.update(min(len(completed_ids), len(environments) * rollouts_per_env))
    try:
        for env in environments:
            for rollout_index in range(rollouts_per_env):
                rollout_id = f"{env.env_id}::r{rollout_index}"
                if rollout_id in completed_ids:
                    continue
                try:
                    row = run_tool_agent_for_env(
                        env=env,
                        cfg=cfg,
                        llm_client=llm_client,
                        agent_cfg=agent_cfg,
                        tool_pool_cfg=tool_pool_cfg,
                        budget_cfg=budget_cfg,
                        user_feedback=user_feedback,
                        task_name="stage09_rlvr_grpo_rollout",
                        context_extra={
                            "rollout_index": rollout_index,
                            "rollouts_per_env": rollouts_per_env,
                            "temperature": stage_cfg.get("temperature"),
                            "top_p": stage_cfg.get("top_p"),
                        },
                        include_messages=True,
                    )
                except Exception as exc:
                    row = _failed_rollout_row(env=env, rollout_index=rollout_index, exc=exc)
                rollout_row = {
                    "rollout_id": rollout_id,
                    "group_id": env.env_id,
                    "env_id": env.env_id,
                    "rollout_index": rollout_index,
                    "level": _env_level(env),
                    "run": row["run"],
                    "trace": row["trace"],
                    "eval": row["eval"],
                    "rubrics": env.rubrics or [],
                }
                rows.append(rollout_row)
                completed_ids.add(rollout_id)
                if checkpoint_path is not None:
                    append_jsonl(checkpoint_path, rollout_row)
                progress.update(1)
    finally:
        progress.close()
    return rows


def _dedupe_rollout_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        rollout_id = str(row.get("rollout_id") or "")
        if not rollout_id:
            continue
        if rollout_id not in by_id:
            order.append(rollout_id)
        by_id[rollout_id] = row
    return [by_id[rollout_id] for rollout_id in order]


def _failed_rollout_row(*, env: ExecutableEnvSpec, rollout_index: int, exc: Exception) -> dict[str, dict[str, Any]]:
    message = str(exc)
    error_type = type(exc).__name__
    run = {
        "env_id": env.env_id,
        "original_task_id": env.original_task_id,
        "difficulty": env.difficulty.model_dump() if env.difficulty else {},
        "final_code": "",
        "notes": [message[:2000]],
        "tool_budget_used": {"total": 0, "per_tool": {}},
        "passed": False,
    }
    trace = {
        "env_id": env.env_id,
        "tool_trace": [
            {
                "tool_name": "llm_client.complete_with_tools",
                "arguments": {"rollout_index": rollout_index},
                "result": {"ok": False, "error_type": error_type, "error_message": message[:4000]},
            }
        ],
        "message_count": 0,
        "messages": [],
    }
    eval_row = {
        "env_id": env.env_id,
        "level": _env_level(env),
        "compile_passed": False,
        "execution_passed": False,
        "evaluation_case_source": "llm_error",
        "passed_cases": 0,
        "total_cases": 0,
        "failure_reasons": [f"LLM_API_ERROR:{error_type}"],
        "case_reports": [],
    }
    return {"run": run, "trace": trace, "eval": eval_row}


def compute_rlvr_reward(
    eval_row: dict[str, Any],
    run: dict[str, Any] | None = None,
    reward_cfg: dict[str, Any] | None = None,
    *,
    trace: dict[str, Any] | None = None,
    rubric_score: float = 0.0,
    use_rubric_reward: bool = False,
) -> dict[str, Any]:
    reward_cfg = reward_cfg or {}
    run = run or {}
    trace = trace or {}
    total_cases = int(eval_row.get("total_cases") or 0)
    passed_cases = int(eval_row.get("passed_cases") or 0)
    case_pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
    sample_pass = 1.0 if bool(eval_row.get("execution_passed")) and total_cases > 0 else 0.0
    valid_submit = 1.0 if total_cases > 0 or bool(eval_row.get("compile_passed")) else 0.0
    no_submit = 1.0 if "MAX_TURNS_WITHOUT_VALID_FINAL_CODE" in (eval_row.get("failure_reasons") or []) else 0.0
    runtime_error = 1.0 if _has_runtime_error(eval_row) else 0.0
    budget_violation_count = _budget_violation_count(run, trace)
    budget_violation = 1.0 if budget_violation_count > 0 else 0.0
    budget_ok = 1.0 if budget_violation_count == 0 else 0.0
    capped_budget_violations = min(
        budget_violation_count,
        int(reward_cfg.get("tool_budget_penalty_cap", 3)),
    )
    reward = (
        float(reward_cfg.get("sample_pass_weight", 1.0)) * sample_pass
        + float(reward_cfg.get("case_pass_rate_weight", 0.5)) * case_pass_rate
        + (float(reward_cfg.get("rubric_score_weight", 0.4)) * float(rubric_score) if use_rubric_reward else 0.0)
        + float(reward_cfg.get("valid_submit_weight", 0.1)) * valid_submit
        + float(reward_cfg.get("within_budget_success_bonus", 0.1)) * sample_pass * budget_ok
        - float(reward_cfg.get("no_submit_penalty", 0.2)) * no_submit
        - float(reward_cfg.get("runtime_error_penalty", 0.1)) * runtime_error
        - float(reward_cfg.get("tool_budget_penalty", 0.1)) * capped_budget_violations
    )
    return {
        "reward": round(float(reward), 6),
        "sample_pass": bool(sample_pass),
        "case_pass_rate": round(case_pass_rate, 6),
        "rubric_score": round(float(rubric_score), 6),
        "use_rubric_reward": bool(use_rubric_reward),
        "valid_submit": bool(valid_submit),
        "no_submit": bool(no_submit),
        "runtime_error": bool(runtime_error),
        "tool_budget_violation": bool(budget_violation),
        "tool_budget_violation_count": budget_violation_count,
        "budget_ok": bool(budget_ok),
    }


def _reward_row(row: dict[str, Any], reward_cfg: dict[str, Any], *, use_rubric_reward: bool) -> dict[str, Any]:
    eval_row = row.get("eval") or {}
    run = row.get("run") or {}
    rubric_result = score_requirement_rubrics(row.get("rubrics") or [], eval_row.get("case_reports") or [])
    reward = compute_rlvr_reward(
        eval_row=eval_row,
        run=run,
        trace=row.get("trace") or {},
        reward_cfg=reward_cfg,
        rubric_score=float(rubric_result["rubric_score"]),
        use_rubric_reward=use_rubric_reward,
    )
    return {
        "rollout_id": row["rollout_id"],
        "group_id": row["group_id"],
        "env_id": row["env_id"],
        "rollout_index": row["rollout_index"],
        "level": row.get("level") or str(eval_row.get("level") or "unknown"),
        "reward": reward["reward"],
        "advantage": 0.0,
        "sample_pass": reward["sample_pass"],
        "case_pass_rate": reward["case_pass_rate"],
        "rubric_score": reward["rubric_score"],
        "rubric_scores": rubric_result["rubric_scores"],
        "rubric_by_category": rubric_result["rubric_by_category"],
        "scored_rubrics": rubric_result["scored_rubrics"],
        "total_rubrics": rubric_result["total_rubrics"],
        "use_rubric_reward": reward["use_rubric_reward"],
        "valid_submit": reward["valid_submit"],
        "no_submit": reward["no_submit"],
        "runtime_error": reward["runtime_error"],
        "tool_budget_violation": reward["tool_budget_violation"],
        "tool_budget_violation_count": reward["tool_budget_violation_count"],
        "budget_ok": reward["budget_ok"],
        "compile_passed": bool(eval_row.get("compile_passed")),
        "execution_passed": bool(eval_row.get("execution_passed")),
        "passed_cases": int(eval_row.get("passed_cases") or 0),
        "total_cases": int(eval_row.get("total_cases") or 0),
        "evaluation_case_source": str(eval_row.get("evaluation_case_source") or ""),
        "failure_reasons": list(eval_row.get("failure_reasons") or []),
        "messages": (row.get("trace") or {}).get("messages") or [],
    }


def attach_group_advantages(reward_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        grouped[str(row.get("group_id") or row.get("env_id") or "")].append(row)
    for rows in grouped.values():
        rewards = [float(row.get("reward") or 0.0) for row in rows]
        mean = sum(rewards) / max(len(rewards), 1)
        variance = sum((value - mean) ** 2 for value in rewards) / max(len(rewards), 1)
        std = math.sqrt(variance)
        for row in rows:
            row["group_reward_mean"] = round(mean, 6)
            row["group_reward_std"] = round(std, 6)
            row["advantage"] = round((float(row.get("reward") or 0.0) - mean) / std, 6) if std > 1e-8 else 0.0
    return reward_rows


def _attach_rewards_to_rollouts(rollout_rows: list[dict[str, Any]], reward_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["rollout_id"]: row for row in reward_rows}
    updated = []
    for row in rollout_rows:
        reward = by_id.get(row["rollout_id"], {})
        cloned = dict(row)
        cloned["reward"] = {
            key: reward.get(key)
            for key in (
                "reward",
                "advantage",
                "sample_pass",
                "case_pass_rate",
                "budget_ok",
                "tool_budget_violation",
                "tool_budget_violation_count",
            )
        }
        updated.append(cloned)
    return updated


def build_rlvr_summary(*, reward_rows: list[dict[str, Any]], stage_cfg: dict[str, Any], policy_slug: str, train_requested: bool) -> dict[str, Any]:
    total = len(reward_rows)
    passed_cases = sum(int(row.get("passed_cases") or 0) for row in reward_rows)
    total_cases = sum(int(row.get("total_cases") or 0) for row in reward_rows)
    by_level: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        grouped[str(row.get("level") or "unknown")].append(row)
    for level, rows in sorted(grouped.items()):
        level_cases = sum(int(row.get("total_cases") or 0) for row in rows)
        level_passed_cases = sum(int(row.get("passed_cases") or 0) for row in rows)
        by_level[level] = {
            "rollouts": len(rows),
            "mean_reward": _mean(float(row.get("reward") or 0.0) for row in rows),
            "mean_rubric_score": _mean(
                float(row.get("rubric_score") or 0.0)
                for row in rows
                if int(row.get("scored_rubrics") or 0) > 0
            ),
            "rubric_scored_rollouts": sum(int(row.get("scored_rubrics") or 0) > 0 for row in rows),
            "sample_pass_rate": _rate(sum(bool(row.get("execution_passed")) for row in rows), len(rows)),
            "case_pass_rate": _rate(level_passed_cases, level_cases),
            "budget_ok_rate": _rate(sum(bool(row.get("budget_ok")) for row in rows), len(rows)),
            "tool_budget_violation_rate": _rate(sum(bool(row.get("tool_budget_violation")) for row in rows), len(rows)),
            "tool_budget_violation_count": sum(int(row.get("tool_budget_violation_count") or 0) for row in rows),
            "passed_cases": level_passed_cases,
            "total_cases": level_cases,
        }
    failure_counter: Counter[str] = Counter()
    for row in reward_rows:
        for reason in row.get("failure_reasons") or []:
            failure_counter[str(reason)] += 1
    return {
        "algorithm": "rlvr_grpo",
        "policy_slug": policy_slug,
        "train_requested": bool(train_requested),
        "base_model": stage_cfg.get("base_model"),
        "sft_adapter": stage_cfg.get("sft_adapter"),
        "rollouts_per_env": int(stage_cfg.get("rollouts_per_env", 4)),
        "num_rollouts": total,
        "mean_reward": _mean(float(row.get("reward") or 0.0) for row in reward_rows),
        "use_rubric_reward": any(bool(row.get("use_rubric_reward")) for row in reward_rows),
        "mean_rubric_score": _mean(
            float(row.get("rubric_score") or 0.0)
            for row in reward_rows
            if int(row.get("scored_rubrics") or 0) > 0
        ),
        "rubric_scored_rollouts": sum(int(row.get("scored_rubrics") or 0) > 0 for row in reward_rows),
        "sample_pass_rate": _rate(sum(bool(row.get("execution_passed")) for row in reward_rows), total),
        "case_pass_rate": _rate(passed_cases, total_cases),
        "budget_ok_rate": _rate(sum(bool(row.get("budget_ok")) for row in reward_rows), total),
        "tool_budget_violation_rate": _rate(sum(bool(row.get("tool_budget_violation")) for row in reward_rows), total),
        "tool_budget_violation_count": sum(int(row.get("tool_budget_violation_count") or 0) for row in reward_rows),
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "by_level": by_level,
        "failure_breakdown": dict(failure_counter.most_common(20)),
    }


def _run_trl_grpo_update(
    *,
    cfg: AppConfig,
    stage_cfg: dict[str, Any],
    environments: list[ExecutableEnvSpec],
    eval_environments: list[ExecutableEnvSpec] | None = None,
    output_dir: Path,
    use_rubric_reward: bool,
    resume: bool,
) -> dict[str, Any]:
    train_prompt_path = output_dir / "prepared_grpo_train.jsonl"
    eval_prompt_path = output_dir / f"prepared_grpo_eval_{stage_cfg.get('eval_split') or 'eval'}.jsonl"
    if is_main_process():
        train_rows = _prepare_trl_grpo_prompt_rows(environments, train_prompt_path)
    else:
        train_rows = []
    eval_environments = list(eval_environments or [])
    if is_main_process() and eval_environments and stage_cfg.get("eval_split"):
        eval_rows = _prepare_trl_grpo_prompt_rows(eval_environments, eval_prompt_path)
    else:
        eval_rows = []
    barrier()
    if not is_main_process():
        train_rows = read_jsonl(train_prompt_path)
        eval_rows = read_jsonl(eval_prompt_path) if eval_prompt_path.exists() else []
    manifest = {
        "status": "skipped",
        "reason": "",
        "trainer": "trl_grpo",
        "adapter_mode": "interactive_tools_final_code_reward",
        "resume": bool(resume),
        "resume_from_checkpoint": str(latest_trainer_checkpoint(output_dir) or "") if resume else "",
        "num_train_envs": len(train_rows),
        "num_eval_envs": len(eval_rows),
        "eval_split": stage_cfg.get("eval_split"),
        "eval_steps": stage_cfg.get("eval_steps"),
        "output_dir": str(output_dir),
        "adapter_dir": "",
    }
    if not train_rows:
        manifest["reason"] = "no_train_environments"
        return manifest
    setup_torch_distributed_device()
    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import PeftModel  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Stage09 --train now uses TRL GRPOTrainer and requires torch, transformers, peft, datasets, and trl. "
            "Install them in the active training environment, then rerun the same command."
        ) from exc

    model_path = str(stage_cfg.get("base_model") or "").strip()
    adapter_path = str(stage_cfg.get("sft_adapter") or "").strip()
    if not model_path or not adapter_path:
        raise ValueError("stage09_rlvr_grpo.base_model and sft_adapter are required for --train.")
    if not Path(adapter_path).is_absolute():
        adapter_path = str(cfg.root / adapter_path)

    trust_remote_code = bool(stage_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_cfg = str(stage_cfg.get("torch_dtype", "auto"))
    torch_dtype: Any = "auto"
    if dtype_cfg not in {"", "auto"}:
        torch_dtype = getattr(torch, dtype_cfg)
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if not is_distributed():
        model_kwargs["device_map"] = stage_cfg.get("device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    grpo_args = _build_grpo_config(GRPOConfig, stage_cfg=stage_cfg, output_dir=output_dir)
    reward_trace_path = output_dir / (
        f"trl_reward_trace_rank{global_rank():03d}.jsonl" if is_distributed() else "trl_reward_trace.jsonl"
    )
    if reward_trace_path.exists():
        reward_trace_path.unlink()
    reward_func = _make_trl_reward_func(
        cfg=cfg,
        stage_cfg=stage_cfg,
        environments=[*environments, *eval_environments],
        reward_trace_path=reward_trace_path,
        use_rubric_reward=use_rubric_reward,
    )
    tool_environment_factory = _make_tool_environment_factory(
        cfg=cfg,
        stage_cfg=stage_cfg,
        environments=[*environments, *eval_environments],
    )
    trainer_kwargs = _filter_kwargs_for_callable(
        GRPOTrainer,
        {
            "model": model,
            "args": grpo_args,
            "reward_funcs": [reward_func],
            "train_dataset": dataset,
            "eval_dataset": eval_dataset,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
            "environment_factory": tool_environment_factory,
        },
    )
    trainer = GRPOTrainer(**trainer_kwargs)
    resume_checkpoint = latest_trainer_checkpoint(output_dir) if resume else None
    manifest["resume_from_checkpoint"] = str(resume_checkpoint or "")
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()
    adapter_dir = output_dir / "adapter"
    if is_main_process():
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    barrier()
    manifest.update(
        {
            "status": "completed",
            "reason": "",
            "adapter_dir": str(adapter_dir),
            "max_steps": int(stage_cfg.get("max_steps", 100)),
            "num_generations": int(stage_cfg.get("num_generations", stage_cfg.get("rollouts_per_env", 4))),
            "reward_trace_path": str(reward_trace_path),
            "eval_split": stage_cfg.get("eval_split"),
            "eval_steps": stage_cfg.get("eval_steps"),
            "num_eval_envs": len(eval_rows),
        }
    )
    return manifest


def _prepare_trl_grpo_prompt_rows(environments: list[ExecutableEnvSpec], output_path: Path) -> list[dict[str, Any]]:
    rows = []
    for env in environments:
        cases, case_source = _evaluation_cases_for_env(env)
        if not cases:
            continue
        rows.append(
            {
                "env_id": env.env_id,
                "level": _env_level(env),
                "evaluation_case_source": case_source,
                "prompt": _trl_grpo_prompt(env),
            }
        )
    write_jsonl(output_path, rows)
    return rows


def _trl_grpo_prompt(env: ExecutableEnvSpec) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _public_user_prompt(env)},
    ]


def _build_grpo_config(config_cls: type, *, stage_cfg: dict[str, Any], output_dir: Path) -> Any:
    num_generations = int(stage_cfg.get("num_generations", stage_cfg.get("rollouts_per_env", 4)))
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(stage_cfg.get("learning_rate", 5e-6)),
        "per_device_train_batch_size": int(stage_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(stage_cfg.get("gradient_accumulation_steps", 4)),
        "max_steps": int(stage_cfg.get("max_steps", 100)),
        "logging_steps": int(stage_cfg.get("logging_steps", 5)),
        "save_steps": int(stage_cfg.get("save_steps", stage_cfg.get("max_steps", 100))),
        "report_to": stage_cfg.get("report_to", "none"),
        "num_generations": num_generations,
        "max_prompt_length": int(stage_cfg.get("max_prompt_length", stage_cfg.get("max_seq_length", 4096))),
        "max_completion_length": int(stage_cfg.get("max_completion_length", stage_cfg.get("max_new_tokens", 2048))),
        "max_tool_calling_iterations": int(stage_cfg.get("max_tool_calling_iterations", _stage09_max_tool_iterations(stage_cfg))),
        "temperature": float(stage_cfg.get("temperature", 0.7)),
        "top_p": float(stage_cfg.get("top_p", 0.95)),
        "beta": float(stage_cfg.get("kl_beta", stage_cfg.get("beta", 0.02))),
        "ddp_find_unused_parameters": bool(stage_cfg.get("ddp_find_unused_parameters", False)),
        "gradient_checkpointing": bool(stage_cfg.get("gradient_checkpointing", False)),
    }
    if stage_cfg.get("eval_split"):
        eval_steps = int(stage_cfg.get("eval_steps") or stage_cfg.get("logging_steps") or 25)
        strategy_key = _strategy_arg_name(config_cls)
        kwargs.update(
            {
                strategy_key: "steps",
                "eval_steps": eval_steps,
                "do_eval": True,
            }
        )
    return config_cls(**_filter_kwargs_for_callable(config_cls, kwargs))


def _stage09_max_tool_iterations(stage_cfg: dict[str, Any]) -> int:
    explicit = stage_cfg.get("max_turns")
    if explicit is not None:
        return int(explicit)
    return 12


def _strategy_arg_name(config_cls: type) -> str:
    params = signature(config_cls).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


def _filter_kwargs_for_callable(target: Callable[..., Any] | type, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = signature(target).parameters
    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in params and value is not None}


def _make_trl_reward_func(
    *,
    cfg: AppConfig,
    stage_cfg: dict[str, Any],
    environments: list[ExecutableEnvSpec],
    reward_trace_path: Path,
    use_rubric_reward: bool,
) -> Callable[..., list[float]]:
    env_by_id = {env.env_id: env for env in environments}

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        env_ids = _coerce_reward_env_ids(kwargs.get("env_id"), len(completions))
        tool_envs = kwargs.get("environments") or []
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            env_id = env_ids[index] if index < len(env_ids) else ""
            env = env_by_id.get(str(env_id))
            text = _completion_to_text(completion)
            tool_env = tool_envs[index] if index < len(tool_envs) else None
            if env is None:
                reward_value = -0.2
                trace_row = {
                    "env_id": env_id,
                    "reward": reward_value,
                    "failure_reasons": ["MISSING_ENV_FOR_REWARD"],
                    "completion_chars": len(text),
                }
            elif isinstance(tool_env, BiocoderGRPOToolEnvironment) and tool_env.env_id == str(env_id):
                result = tool_env._reward_result(
                    completion_text=text,
                    reward_cfg=stage_cfg.get("reward") or {},
                    use_rubric_reward=use_rubric_reward,
                )
                reward_value = float(result["reward"])
                trace_row = result
            else:
                result = _evaluate_completion_reward(
                    cfg=cfg,
                    env=env,
                    completion_text=text,
                    reward_cfg=stage_cfg.get("reward") or {},
                    use_rubric_reward=use_rubric_reward,
                )
                reward_value = float(result["reward"])
                trace_row = result
            append_jsonl(reward_trace_path, trace_row)
            rewards.append(float(reward_value))
        return rewards

    return reward_func


class BiocoderGRPOToolEnvironment:
    def __init__(
        self,
        *,
        cfg: AppConfig,
        env_by_id: dict[str, ExecutableEnvSpec],
        agent_cfg: dict[str, Any],
        stage06_budget_cfg: dict[str, Any],
        allowed_tools: set[str],
    ) -> None:
        self.cfg = cfg
        self.env_by_id = env_by_id
        self.agent_cfg = agent_cfg
        self.stage06_budget_cfg = stage06_budget_cfg
        self.allowed_tools = allowed_tools
        self.env: ExecutableEnvSpec | None = None
        self.env_id = ""
        self.runtime: ToolRuntime | None = None

    def reset(self, env_id: str = "", **kwargs: Any) -> None:
        self.env_id = str(env_id or kwargs.get("id") or "")
        self.env = self.env_by_id.get(self.env_id)
        if self.env is None:
            self.runtime = None
            return None
        level = _env_level(self.env)
        env_budget = _budget_for_level(level=level, agent_cfg=self.agent_cfg, stage06_budget_cfg=self.stage06_budget_cfg)
        self.runtime = ToolRuntime(
            self.env,
            self.cfg,
            budget=env_budget,
            allowed_tools=self.allowed_tools,
            submit_excluded_from_total=bool(self.stage06_budget_cfg.get("submit_final_code_excluded_from_total", True)),
        )
        return None

    def get_task_context(self, window: int = 4000) -> str:
        """Read public task context, scaffold, signature, and requirements.

        Args:
            window: Maximum characters of long text fields to return.

        Returns:
            A JSON string with public task context.
        """
        return self._execute_json("get_task_context", {"window": window})

    def validate_candidate_code(self, code: str) -> str:
        """Check syntax, imports, target signature, and static safety for candidate code.

        Args:
            code: Complete executable Python program.

        Returns:
            A JSON string with validation results and repair hints.
        """
        return self._execute_json("validate_candidate_code", {"code": code})

    def create_test_file(self, path: str, content: str) -> str:
        """Create a small public fixture file for later custom tests.

        Args:
            path: Safe relative path such as input.csv or data/example.tsv.
            content: UTF-8 text content to write.

        Returns:
            A JSON string describing the created fixture.
        """
        return self._execute_json("create_test_file", {"path": path, "content": content})

    def run_custom_test(self, code: str, test_snippet: str, timeout_seconds: int = 5) -> str:
        """Run a self-authored public test snippet against candidate code.

        Args:
            code: Complete executable Python program.
            test_snippet: Public Python assertions or calls exercising the candidate.
            timeout_seconds: Maximum runtime for this test.

        Returns:
            A JSON string with stdout, stderr, artifacts, and failures.
        """
        return self._execute_json(
            "run_custom_test",
            {"code": code, "test_snippet": test_snippet, "timeout_seconds": timeout_seconds},
        )

    def submit_final_code(self, code: str) -> str:
        """Submit the final complete Python program for oracle-backed evaluation.

        Args:
            code: Final complete executable Python program.

        Returns:
            A JSON string with aggregate oracle evaluation results. Hidden oracle cases are not revealed.
        """
        return self._execute_json("submit_final_code", {"code": code})

    def _reward_result(self, *, completion_text: str, reward_cfg: dict[str, Any], use_rubric_reward: bool) -> dict[str, Any]:
        if self.env is None:
            return {
                "env_id": self.env_id,
                "reward": -0.2,
                "failure_reasons": ["MISSING_TOOL_ENV"],
                "completion_chars": len(completion_text),
            }
        if self.runtime is None:
            return _evaluate_completion_reward(
                cfg=self.cfg,
                env=self.env,
                completion_text=completion_text,
                reward_cfg=reward_cfg,
                use_rubric_reward=use_rubric_reward,
            )
        if self.runtime.final_eval is None:
            fallback = _evaluate_completion_reward(
                cfg=self.cfg,
                env=self.env,
                completion_text=completion_text,
                reward_cfg=reward_cfg,
                use_rubric_reward=use_rubric_reward,
                run={"notes": [], "tool_budget_used": {"total": self.runtime.total_calls, "per_tool": dict(self.runtime.call_counts)}},
                trace={"tool_trace": self.runtime.trace},
            )
            fallback["tool_budget_used"] = {"total": self.runtime.total_calls, "per_tool": dict(self.runtime.call_counts)}
            fallback["tool_call_count"] = len(self.runtime.trace)
            fallback["submitted_via_tool"] = False
            return fallback
        case_reports = self.runtime.final_eval.get("case_reports") or []
        rubric_result = score_requirement_rubrics(self.env.rubrics or [], case_reports)
        eval_row = {
            "env_id": self.env.env_id,
            "level": _env_level(self.env),
            "compile_passed": bool(self.runtime.final_eval.get("compile_passed")),
            "execution_passed": bool(self.runtime.final_eval.get("execution_passed")) and len(case_reports) > 0,
            "evaluation_case_source": str(self.runtime.final_eval.get("evaluation_case_source") or ""),
            "passed_cases": sum(1 for report in case_reports if report.get("passed")),
            "total_cases": len(case_reports),
            "failure_reasons": list(self.runtime.final_eval.get("failure_reasons") or []),
            "case_reports": case_reports,
        }
        reward = compute_rlvr_reward(
            eval_row=eval_row,
            run={"notes": [], "tool_budget_used": {"total": self.runtime.total_calls, "per_tool": dict(self.runtime.call_counts)}},
            trace={"tool_trace": self.runtime.trace},
            reward_cfg=reward_cfg,
            rubric_score=float(rubric_result["rubric_score"]),
            use_rubric_reward=use_rubric_reward,
        )
        return {
            "env_id": self.env.env_id,
            "level": _env_level(self.env),
            "reward": reward["reward"],
            "case_pass_rate": reward["case_pass_rate"],
            "rubric_score": reward["rubric_score"],
            "use_rubric_reward": reward["use_rubric_reward"],
            "tool_budget_violation": reward["tool_budget_violation"],
            "tool_budget_violation_count": reward["tool_budget_violation_count"],
            "budget_ok": reward["budget_ok"],
            "compile_passed": eval_row["compile_passed"],
            "execution_passed": eval_row["execution_passed"],
            "passed_cases": eval_row["passed_cases"],
            "total_cases": eval_row["total_cases"],
            "evaluation_case_source": eval_row["evaluation_case_source"],
            "failure_reasons": eval_row["failure_reasons"][:10],
            "final_code_chars": len(self.runtime.final_code or ""),
            "completion_chars": len(completion_text),
            "tool_budget_used": {"total": self.runtime.total_calls, "per_tool": dict(self.runtime.call_counts)},
            "tool_call_count": len(self.runtime.trace),
            "submitted_via_tool": True,
        }

    def _execute_json(self, name: str, arguments: dict[str, Any]) -> str:
        if self.runtime is None:
            return json.dumps({"ok": False, "error": "environment_not_initialized"}, ensure_ascii=False)
        result = self.runtime.execute(name, arguments)
        return json.dumps(result, ensure_ascii=False)


def _make_tool_environment_factory(
    *,
    cfg: AppConfig,
    stage_cfg: dict[str, Any],
    environments: list[ExecutableEnvSpec],
) -> Callable[[], BiocoderGRPOToolEnvironment]:
    env_by_id = {env.env_id: env for env in environments}
    stage06_cfg = cfg.values.get("stage06", {}) or {}
    agent_cfg = stage06_cfg.get("tool_agent", stage06_cfg) or {}
    tool_pool_cfg = load_yaml(cfg.dataset_config_path("tool_pool.yaml"))
    budget_cfg = load_yaml(cfg.dataset_config_path_with_fallback("m_level_budgets_4axis.yaml", "m_level_budgets_7axis.yaml"))
    stage06_budget_cfg = (budget_cfg or {}).get("stage06_tool_agent", {})
    allowed_tools = stage06_tool_names(tool_pool_cfg)
    if stage_cfg.get("allowed_tools"):
        allowed_tools = {name for name in allowed_tools if name in set(stage_cfg.get("allowed_tools") or [])}

    def factory() -> BiocoderGRPOToolEnvironment:
        return BiocoderGRPOToolEnvironment(
            cfg=cfg,
            env_by_id=env_by_id,
            agent_cfg=agent_cfg,
            stage06_budget_cfg=stage06_budget_cfg,
            allowed_tools=allowed_tools,
        )

    return factory


def _coerce_reward_env_ids(raw: Any, length: int) -> list[str]:
    if isinstance(raw, list):
        values = [str(item) for item in raw]
    elif raw is None:
        values = [""]
    else:
        values = [str(raw)]
    if len(values) >= length:
        return values
    return [values[index % len(values)] for index in range(length)] if values else [""] * length


def _evaluate_completion_reward(
    *,
    cfg: AppConfig,
    env: ExecutableEnvSpec,
    completion_text: str,
    reward_cfg: dict[str, Any],
    use_rubric_reward: bool,
    run: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_code = _extract_final_code_from_completion(completion_text)
    cases, case_source = _evaluation_cases_for_env(env)
    if not final_code.strip() or not cases:
        eval_row = {
            "env_id": env.env_id,
            "level": _env_level(env),
            "compile_passed": False,
            "execution_passed": False,
            "evaluation_case_source": case_source if cases else "none",
            "passed_cases": 0,
            "total_cases": len(cases),
            "failure_reasons": ["NO_FINAL_CODE"] if not final_code.strip() else ["NO_EVALUATION_CASES"],
            "case_reports": [],
        }
        rubric_result = score_requirement_rubrics(env.rubrics or [], [])
    else:
        execution = run_scaled_gold_on_validated_oracle_cases(
            env,
            final_code,
            cases,
            python_bin=_agent_python_bin(cfg),
        )
        case_reports = execution.get("case_reports") or []
        rubric_result = score_requirement_rubrics(env.rubrics or [], case_reports)
        eval_row = {
            "env_id": env.env_id,
            "level": _env_level(env),
            "compile_passed": bool(execution.get("compile_passed")),
            "execution_passed": bool(execution.get("execution_passed")) and len(case_reports) > 0,
            "evaluation_case_source": case_source,
            "passed_cases": sum(1 for report in case_reports if report.get("passed")),
            "total_cases": len(case_reports),
            "failure_reasons": list(execution.get("failure_reasons") or []),
            "case_reports": case_reports,
        }
    reward = compute_rlvr_reward(
        eval_row=eval_row,
        run=run or {"notes": []},
        trace=trace or {},
        reward_cfg=reward_cfg,
        rubric_score=float(rubric_result["rubric_score"]),
        use_rubric_reward=use_rubric_reward,
    )
    return {
        "env_id": env.env_id,
        "level": _env_level(env),
        "reward": reward["reward"],
        "case_pass_rate": reward["case_pass_rate"],
        "rubric_score": reward["rubric_score"],
        "use_rubric_reward": reward["use_rubric_reward"],
        "tool_budget_violation": reward["tool_budget_violation"],
        "tool_budget_violation_count": reward["tool_budget_violation_count"],
        "budget_ok": reward["budget_ok"],
        "compile_passed": eval_row["compile_passed"],
        "execution_passed": eval_row["execution_passed"],
        "passed_cases": eval_row["passed_cases"],
        "total_cases": eval_row["total_cases"],
        "evaluation_case_source": eval_row["evaluation_case_source"],
        "failure_reasons": eval_row["failure_reasons"][:10],
        "final_code_chars": len(final_code),
        "completion_chars": len(completion_text),
    }


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or json.dumps(completion, ensure_ascii=False))
    return str(completion or "")


def _extract_final_code_from_completion(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        payload = parse_json_payload(raw)
        if isinstance(payload, dict) and str(payload.get("final_code") or "").strip():
            return str(payload.get("final_code") or "")
    except Exception:
        pass
    fenced = _extract_first_code_fence(raw)
    if fenced.strip():
        try:
            payload = parse_json_payload(fenced)
            if isinstance(payload, dict) and str(payload.get("final_code") or "").strip():
                return str(payload.get("final_code") or "")
        except Exception:
            pass
        return fenced
    return raw


def _extract_first_code_fence(text: str) -> str:
    match = re.search(r"```(?:python|py|json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _has_runtime_error(eval_row: dict[str, Any]) -> bool:
    reasons = "\n".join(str(item) for item in (eval_row.get("failure_reasons") or [])).lower()
    if any(marker in reasons for marker in ("runtime", "traceback", "exception", "timeout")):
        return True
    for case in eval_row.get("case_reports") or []:
        if not isinstance(case, dict):
            continue
        text = json.dumps(case, ensure_ascii=False).lower()
        if any(marker in text for marker in ("traceback", "exception", "runtime", "timeout")):
            return True
    return False


def _budget_violation_count(run: dict[str, Any], trace: dict[str, Any]) -> int:
    note_violation = 1 if _has_budget_violation(run) else 0
    trace_violations = 0
    for event in trace.get("tool_trace") or []:
        if not isinstance(event, dict):
            continue
        if event.get("budget_violation"):
            trace_violations += 1
            continue
        result = event.get("result")
        if isinstance(result, dict):
            text = json.dumps(result, ensure_ascii=False).lower()
        else:
            text = str(result or "").lower()
        budget_error = str(event.get("budget_error") or "").lower()
        if "tool_budget_exhausted" in text or "tool_budget_exceeded" in text or "tool_budget_exceeded" in budget_error:
            trace_violations += 1
    return max(note_violation, trace_violations)


def _has_budget_violation(run: dict[str, Any]) -> bool:
    notes = "\n".join(str(item) for item in (run.get("notes") or [])).lower()
    if "tool_budget" in notes:
        return True
    return False


def _policy_slug(*, stage_cfg: dict[str, Any], llm_client: LLMClient) -> str:
    adapter = str(stage_cfg.get("sft_adapter") or "").strip()
    model = str(stage_cfg.get("base_model") or ((llm_client.config.get("local") or {}).get("model_path") or "policy"))
    adapter_name = Path(adapter).name if adapter else ""
    model_name = Path(model).name if model else "policy"
    split_name = str(stage_cfg.get("split") or "all").strip() or "all"
    return slugify("_".join(part for part in [model_name, adapter_name, f"{split_name}_split", "rlvr_grpo"] if part), max_length=96)


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 6) if items else 0.0
