from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from medenvscale.config import AppConfig
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.utils import append_jsonl, read_jsonl, stable_hash
from tqdm.auto import tqdm


_WRAPPER_NAME = "run_candidate.py"
_CANDIDATE_NAME = "candidate.py"
_RESULT_NAME = "execution_result.json"
_SEED_CASE_ID = "seed_case_main"


def validate_and_repair_code_rows(
    rows: list[dict[str, Any]],
    cfg: AppConfig,
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
    parallel_workers: int | None = None,
    resume: bool = False,
    checkpoint_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    worker_count = max(1, int(parallel_workers or 1))
    if llm_client is not None and llm_client.mode == "local" and worker_count > 1:
        print("Stage00 workers is forced to 1 for local LLM mode to avoid concurrent model.generate calls.")
        worker_count = 1
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed = _load_code_validation_checkpoint(checkpoint) if resume and checkpoint is not None else {}
    results: list[tuple[int, dict[str, Any], bool, dict[str, Any]]] = []
    direct_pass_count = 0
    repaired_pass_count = 0
    pending: list[tuple[int, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        key = _code_validation_checkpoint_key(row, index)
        if key in completed:
            item = completed[key]
            results.append((index, dict(item.get("prepared") or {}), bool(item.get("accepted")), dict(item.get("audit") or {})))
        else:
            pending.append((index, key, row))

    progress = tqdm(
        total=len(rows),
        initial=len(results),
        desc="Stage00 Prepare",
        unit="task",
        leave=True,
    ) if rows else None
    try:
        if worker_count == 1:
            for index, key, row in pending:
                prepared, accepted, audit = prepare_executable_row(
                    row=row,
                    cfg=cfg,
                    llm_client=llm_client,
                    prompt_runner=prompt_runner,
                )
                results.append((index, prepared, accepted, audit))
                _append_code_validation_checkpoint(checkpoint, key, prepared, accepted, audit)
                if progress is not None:
                    progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        prepare_executable_row,
                        row=row,
                        cfg=cfg,
                        llm_client=llm_client,
                        prompt_runner=prompt_runner,
                    ): (index, key)
                    for index, key, row in pending
                }
                for future in as_completed(futures):
                    index, key = futures[future]
                    prepared, accepted, audit = future.result()
                    results.append((index, prepared, accepted, audit))
                    _append_code_validation_checkpoint(checkpoint, key, prepared, accepted, audit)
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for _, prepared, accepted, audit in sorted(results, key=lambda item: item[0]):
        if accepted:
            accepted_rows.append(prepared)
            if prepared.get("repair_succeeded"):
                repaired_pass_count += 1
            else:
                direct_pass_count += 1
        else:
            rejected_rows.append(audit)

    summary = {
        "input_rows": len(rows),
        "accepted_rows": len(accepted_rows),
        "rejected_rows": len(rejected_rows),
        "direct_pass_rows": direct_pass_count,
        "repaired_pass_rows": repaired_pass_count,
    }
    return accepted_rows, rejected_rows, summary


def _code_validation_checkpoint_key(row: dict[str, Any], index: int) -> str:
    explicit_id = row.get("task_id") or row.get("id") or row.get("instance_id") or row.get("question_id") or row.get("idx")
    source_split = row.get("source_split") or "unknown"
    if explicit_id:
        return f"{source_split}:{explicit_id}"
    return f"{source_split}:row_{index}:{stable_hash(row)}"


def _load_code_validation_checkpoint(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("task_key") or "")
        if not key:
            continue
        completed[key] = row
    if completed:
        print(f"Stage00 resume: loaded {len(completed)} checkpoint rows from {path}")
    return completed


def _append_code_validation_checkpoint(
    path: Path | None,
    key: str,
    prepared: dict[str, Any],
    accepted: bool,
    audit: dict[str, Any],
) -> None:
    if path is None:
        return
    append_jsonl(
        path,
        {
            "task_key": key,
            "accepted": bool(accepted),
            "prepared": prepared,
            "audit": audit,
        },
    )


def prepare_executable_row(
    row: dict[str, Any],
    cfg: AppConfig,
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    execution_cfg = _execution_config(cfg)
    max_repair_attempts = max(0, int(execution_cfg.get("max_repair_attempts", 3)))
    original_code = str(row.get("code") or row.get("full_code") or row.get("reference_code") or "").strip()
    task_id = str(row.get("task_id") or row.get("idx") or "unknown_task")
    base_audit = {
        "task_id": task_id,
        "idx": row.get("idx"),
        "source_split": row.get("source_split"),
    }

    if not original_code:
        audit = dict(base_audit)
        audit.update(
            {
                "status": "reject",
                "reason": "missing_code",
                "repair_attempts": 0,
            }
        )
        return dict(row), False, audit

    current_code = original_code
    repair_attempts = 0
    repair_history: list[dict[str, Any]] = []
    initial_execution = execute_code_for_ground_truth(current_code, cfg)
    last_execution = initial_execution

    while not _execution_passed(last_execution) and repair_attempts < max_repair_attempts:
        if llm_client is None or prompt_runner is None:
            break
        repair_attempts += 1
        repair_result = repair_code_sample(
            row=row,
            current_code=current_code,
            execution_result=last_execution,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
        )
        repair_history.append(
            {
                "attempt": repair_attempts,
                "summary": repair_result.get("repair_summary", ""),
                "llm_source": repair_result.get("llm_source"),
                "error": repair_result.get("error"),
            }
        )
        repaired_code = str(repair_result.get("repaired_code") or "").strip()
        if not repaired_code:
            continue
        current_code = repaired_code
        last_execution = execute_code_for_ground_truth(current_code, cfg)

    if _execution_passed(last_execution):
        prepared = dict(row)
        prepared["code"] = current_code
        if current_code != original_code:
            prepared["wrong_code"] = original_code
        prepared["ground_truth_output_signature"] = last_execution["output_signature"]
        prepared["code_execution_result"] = _public_execution_result(last_execution)
        seed_case, seed_case_audit = build_seed_execution_case(
            row=prepared,
            code=current_code,
            ground_truth_output_signature=last_execution["output_signature"],
            cfg=cfg,
            llm_client=llm_client,
            prompt_runner=prompt_runner,
        )
        prepared["seed_execution_case"] = seed_case or {}
        prepared["seed_case_audit"] = seed_case_audit
        if str(seed_case_audit.get("status") or "").strip() != "pass":
            audit = dict(base_audit)
            audit.update(
                {
                    "status": "reject",
                    "reason": "seed_case_admission_failed",
                    "seed_case_failure_reason": seed_case_audit.get("failure_reason"),
                    "seed_case_mismatch_reasons": list(seed_case_audit.get("mismatch_reasons") or []),
                    "seed_case_audit": seed_case_audit,
                    "repair_attempts": repair_attempts,
                    "repair_succeeded": current_code != original_code,
                    "code_execution_result": _public_execution_result(last_execution),
                }
            )
            rejected_row = dict(prepared)
            rejected_row["execution_status"] = "reject"
            return rejected_row, False, audit
        prepared["execution_status"] = "pass"
        prepared["repair_attempts"] = repair_attempts
        prepared["repair_succeeded"] = current_code != original_code
        prepared["repair_history"] = repair_history
        prepared.setdefault("ground_truth", prepared["ground_truth_output_signature"])
        audit = dict(base_audit)
        audit.update(
            {
                "status": "accept",
                "repair_attempts": repair_attempts,
                "repair_succeeded": prepared["repair_succeeded"],
            }
        )
        return prepared, True, audit

    audit = dict(base_audit)
    audit.update(
        {
            "status": "reject",
            "reason": last_execution.get("failure_reason") or "execution_failed",
            "repair_attempts": repair_attempts,
            "repair_history": repair_history,
            "last_execution": _public_execution_result(last_execution),
            "wrong_code": original_code,
            "last_code": current_code,
        }
    )
    rejected_row = dict(row)
    rejected_row["execution_status"] = "reject"
    return rejected_row, False, audit


def build_seed_execution_case(
    row: dict[str, Any],
    code: str,
    ground_truth_output_signature: dict[str, Any],
    cfg: AppConfig,
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempts: list[tuple[str, dict[str, Any] | None]] = [
        ("rule_based", _extract_seed_case_rule_based(row, code)),
        ("heuristic", _extract_seed_case_heuristic(row, code)),
    ]
    if llm_client is not None and prompt_runner is not None:
        attempts.append(
            (
                "llm_reconstructed",
                _reconstruct_seed_case_with_llm(
                    row=row,
                    code=code,
                    ground_truth_output_signature=ground_truth_output_signature,
                    llm_client=llm_client,
                    prompt_runner=prompt_runner,
                ),
            )
        )

    last_audit: dict[str, Any] | None = None
    for extraction_method, candidate in attempts:
        if candidate is None:
            continue
        replay = replay_seed_execution_case(code, candidate, cfg)
        if not _execution_passed(replay):
            last_audit = {
                "status": "soft_fail",
                "failure_reason": replay.get("failure_reason") or "seed_case_replay_failed",
                "extraction_method": extraction_method,
                "validated_against_ground_truth": False,
                "replay_result": _public_execution_result(replay),
            }
            continue

        replay_output = dict(replay.get("output_signature") or {})
        matched, mismatch_reasons = _compare_seed_case_to_ground_truth(replay_output, ground_truth_output_signature)
        if not matched:
            last_audit = {
                "status": "soft_fail",
                "failure_reason": "seed_case_ground_truth_mismatch",
                "mismatch_reasons": mismatch_reasons,
                "extraction_method": extraction_method,
                "validated_against_ground_truth": False,
                "replay_result": _public_execution_result(replay),
            }
            continue

        assumptions = candidate.get("assumptions")
        if not isinstance(assumptions, list):
            assumptions = []
        final_case = {
            "case_id": str(candidate.get("case_id") or _SEED_CASE_ID),
            "description": str(candidate.get("description") or candidate.get("case_description") or "Replayable seed execution case."),
            "setup_code": str(candidate.get("setup_code") or "").strip(),
            "call_code": str(candidate.get("call_code") or "").strip(),
            "expected_output_signature": _build_expected_output_signature(replay_output),
            "observed_output_signature": replay_output,
            "extraction_method": extraction_method,
            "validated_against_ground_truth": True,
            "assumptions": [str(item) for item in assumptions if str(item).strip()],
        }
        return final_case, {
            "status": "pass",
            "failure_reason": None,
            "extraction_method": extraction_method,
            "validated_against_ground_truth": True,
            "replay_result": _public_execution_result(replay),
        }

    if last_audit is not None:
        return None, last_audit
    return None, {
        "status": "soft_fail",
        "failure_reason": "seed_case_extraction_failed",
        "extraction_method": None,
        "validated_against_ground_truth": False,
    }


def execute_code_for_ground_truth(code: str, cfg: AppConfig) -> dict[str, Any]:
    execution_cfg = _execution_config(cfg)
    backend = str(execution_cfg.get("backend", "docker")).strip().lower()
    timeout_seconds = max(1, int(execution_cfg.get("timeout_seconds", 30)))

    with tempfile.TemporaryDirectory(prefix="medenvscale-stage00-") as temp_dir:
        workspace = Path(temp_dir)
        candidate_path = workspace / _CANDIDATE_NAME
        wrapper_path = workspace / _WRAPPER_NAME
        result_path = workspace / _RESULT_NAME
        candidate_path.write_text(code, encoding="utf-8")
        wrapper_path.write_text(_wrapper_source(), encoding="utf-8")

        if backend == "local":
            python_bin = str(execution_cfg.get("local_python_bin") or sys.executable)
            command = [python_bin, _WRAPPER_NAME]
        else:
            docker_image = str(execution_cfg.get("docker_image") or "medenvscale-biocoder:latest")
            python_bin = str(execution_cfg.get("python_bin") or "python")
            command = [
                "docker",
                "run",
                "--rm",
                "-w",
                "/workspace",
                "-v",
                f"{workspace}:/workspace",
                docker_image,
                python_bin,
                _WRAPPER_NAME,
            ]

        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            return _failed_execution_result(
                backend=backend,
                failure_reason="execution_backend_missing",
                error_message=str(exc),
            )
        except subprocess.TimeoutExpired:
            return _failed_execution_result(
                backend=backend,
                failure_reason="execution_timeout",
                error_message=f"Execution exceeded {timeout_seconds} seconds.",
            )

        if not result_path.exists():
            return _failed_execution_result(
                backend=backend,
                failure_reason="wrapper_result_missing",
                error_message=(completed.stderr or completed.stdout or "").strip(),
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _failed_execution_result(
                backend=backend,
                failure_reason="wrapper_result_invalid_json",
                error_message=str(exc),
            )

    stdout = _normalize_stdout(str(payload.get("stdout") or ""))
    stderr = _normalize_stdout(str(payload.get("stderr") or ""))
    return_value = _normalize_jsonish(payload.get("return_value"))
    file_artifacts = _normalize_artifact_summary(payload.get("artifacts"))
    output_signature = {
        "return_value": return_value,
        "stdout": stdout,
        "file_artifacts": file_artifacts,
    }
    observable = _has_observable_output(output_signature)
    status = str(payload.get("status") or "fail").lower()
    passed = status == "pass" and observable
    failure_reason = None
    if status != "pass":
        failure_reason = "script_runtime_error"
    elif not observable:
        failure_reason = "no_observable_output_signature"

    return {
        "status": "pass" if passed else "fail",
        "backend": backend,
        "stdout": stdout,
        "stderr": stderr,
        "return_value": return_value,
        "file_artifacts": file_artifacts,
        "output_signature": output_signature,
        "failure_reason": failure_reason,
        "runtime_error": {
            "error_type": payload.get("error_type"),
            "error_message": payload.get("error_message"),
            "traceback": payload.get("traceback"),
        },
    }


def repair_code_sample(
    row: dict[str, Any],
    current_code: str,
    execution_result: dict[str, Any],
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
) -> dict[str, Any]:
    prompt = prompt_runner.render(
        "stage00_code_repair.jinja",
        problem=str(row.get("problem") or row.get("instruction") or ""),
        signature=str(row.get("signature") or ""),
        context=str(row.get("context") or ""),
        solution=str(row.get("solution") or row.get("ground_truth") or ""),
        current_code=current_code,
        failure_reason=str(execution_result.get("failure_reason") or ""),
        stdout=str(execution_result.get("stdout") or ""),
        stderr=str(execution_result.get("stderr") or ""),
        runtime_error=json.dumps(execution_result.get("runtime_error") or {}, ensure_ascii=False, indent=2),
    )
    context = {
        "task_id": row.get("task_id") or row.get("idx"),
        "signature": row.get("signature"),
        "failure_reason": execution_result.get("failure_reason"),
        "current_code": current_code,
    }
    response = llm_client.complete_json(
        task_name="stage00_code_repair",
        prompt=prompt,
        context=context,
        mock_builder=_mock_code_repair_builder,
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    return {
        "repaired_code": str(payload.get("repaired_code") or "").strip(),
        "repair_summary": str(payload.get("repair_summary") or ""),
        "llm_source": response.source,
        "error": str(payload.get("error") or ""),
    }


def _mock_code_repair_builder(context: dict[str, Any]) -> dict[str, Any]:
    code = str(context.get("current_code") or "")
    repaired = code.replace("prin(", "print(").replace("retun ", "return ")
    if repaired == code and "print answer" in code:
        repaired = code.replace("print answer", "print(answer)")
    return {
        "repaired_code": repaired,
        "repair_summary": "Applied simple mock syntax repairs for stage00 code execution.",
    }


def _execution_config(cfg: AppConfig) -> dict[str, Any]:
    dataset_cfg = cfg.values.get("dataset", {})
    return dict(dataset_cfg.get("code_execution", {}) or {})


def _execution_passed(execution_result: dict[str, Any]) -> bool:
    return str(execution_result.get("status") or "") == "pass"


def _public_execution_result(execution_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": execution_result.get("status"),
        "backend": execution_result.get("backend"),
        "stdout": execution_result.get("stdout"),
        "stderr": execution_result.get("stderr"),
        "return_value": execution_result.get("return_value"),
        "file_artifacts": execution_result.get("file_artifacts"),
        "output_signature": execution_result.get("output_signature"),
        "failure_reason": execution_result.get("failure_reason"),
        "runtime_error": execution_result.get("runtime_error"),
    }


def replay_seed_execution_case(code: str, seed_case: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    execution_cfg = _execution_config(cfg)
    backend = str(execution_cfg.get("backend", "docker")).strip().lower()
    timeout_seconds = max(1, int(execution_cfg.get("timeout_seconds", 30)))

    with tempfile.TemporaryDirectory(prefix="medenvscale-seed-case-") as temp_dir:
        workspace = Path(temp_dir)
        candidate_path = workspace / _CANDIDATE_NAME
        wrapper_path = workspace / "run_seed_case.py"
        result_path = workspace / _RESULT_NAME
        seed_case_path = workspace / "seed_case.json"
        candidate_path.write_text(code, encoding="utf-8")
        seed_case_path.write_text(json.dumps(seed_case, ensure_ascii=False), encoding="utf-8")
        wrapper_path.write_text(_seed_case_wrapper_source(), encoding="utf-8")

        if backend == "local":
            python_bin = str(execution_cfg.get("local_python_bin") or sys.executable)
            command = [python_bin, wrapper_path.name]
        else:
            docker_image = str(execution_cfg.get("docker_image") or "medenvscale-biocoder:latest")
            python_bin = str(execution_cfg.get("python_bin") or "python")
            command = [
                "docker",
                "run",
                "--rm",
                "-w",
                "/workspace",
                "-v",
                f"{workspace}:/workspace",
                docker_image,
                python_bin,
                wrapper_path.name,
            ]

        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            return _failed_execution_result(
                backend=backend,
                failure_reason="seed_case_backend_missing",
                error_message=str(exc),
            )
        except subprocess.TimeoutExpired:
            return _failed_execution_result(
                backend=backend,
                failure_reason="seed_case_timeout",
                error_message=f"Seed case replay exceeded {timeout_seconds} seconds.",
            )

        if not result_path.exists():
            return _failed_execution_result(
                backend=backend,
                failure_reason="seed_case_result_missing",
                error_message=(completed.stderr or completed.stdout or "").strip(),
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _failed_execution_result(
                backend=backend,
                failure_reason="seed_case_result_invalid_json",
                error_message=str(exc),
            )

    stdout = _normalize_stdout(str(payload.get("stdout") or ""))
    stderr = _normalize_stdout(str(payload.get("stderr") or ""))
    return_value = _normalize_jsonish(payload.get("return_value"))
    file_artifacts = _normalize_artifact_summary(payload.get("artifacts"))
    output_signature = {
        "return_value": return_value,
        "stdout": stdout,
        "file_artifacts": file_artifacts,
    }
    observable = _has_observable_output(output_signature)
    status = str(payload.get("status") or "fail").lower()
    passed = status == "pass" and observable
    failure_reason = None
    if status != "pass":
        error_type = str(payload.get("error_type") or "").strip()
        failure_reason = f"seed_case_replay_error:{error_type or 'RuntimeError'}"
    elif not observable:
        failure_reason = "no_observable_output_signature"

    return {
        "status": "pass" if passed else "fail",
        "backend": backend,
        "stdout": stdout,
        "stderr": stderr,
        "return_value": return_value,
        "file_artifacts": file_artifacts,
        "output_signature": output_signature,
        "failure_reason": failure_reason,
        "runtime_error": {
            "error_type": payload.get("error_type"),
            "error_message": payload.get("error_message"),
            "traceback": payload.get("traceback"),
        },
    }


def _normalize_stdout(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def _normalize_jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_jsonish(item) for key, item in value.items()}
    return repr(value)


def _normalize_artifact_summary(artifacts: Any) -> list[dict[str, Any]]:
    if not isinstance(artifacts, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        normalized.append(
            {
                "path": path,
                "size_bytes": int(item.get("size_bytes") or 0),
                "sha256": str(item.get("sha256") or ""),
            }
        )
    return normalized


def _has_observable_output(output_signature: dict[str, Any]) -> bool:
    return bool(
        output_signature.get("stdout")
        or output_signature.get("file_artifacts")
        or output_signature.get("return_value") is not None
    )


def _failed_execution_result(backend: str, failure_reason: str, error_message: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "backend": backend,
        "stdout": "",
        "stderr": "",
        "return_value": None,
        "file_artifacts": [],
        "output_signature": {"return_value": None, "stdout": "", "file_artifacts": []},
        "failure_reason": failure_reason,
        "runtime_error": {
            "error_type": None,
            "error_message": error_message,
            "traceback": "",
        },
    }


def _build_expected_output_signature(observed_output_signature: dict[str, Any]) -> dict[str, Any]:
    stdout = _normalize_stdout(str(observed_output_signature.get("stdout") or ""))
    stdout_contains = [line.strip() for line in stdout.split("\n") if line.strip()]
    file_artifacts = observed_output_signature.get("file_artifacts")
    normalized_artifacts: list[dict[str, Any]] = []
    if isinstance(file_artifacts, list):
        normalized_artifacts = [
            {"path": str(item.get("path") or "").strip()}
            for item in file_artifacts
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        ]
    return {
        "return_value": _normalize_jsonish(observed_output_signature.get("return_value")),
        "stdout_contains": stdout_contains,
        "file_artifacts": normalized_artifacts,
    }


def _compare_seed_case_to_ground_truth(
    replay_output_signature: dict[str, Any],
    ground_truth_output_signature: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_stdout = _normalize_stdout(str(ground_truth_output_signature.get("stdout") or ""))
    observed_stdout = _normalize_stdout(str(replay_output_signature.get("stdout") or ""))
    if not _stdout_semantically_equal(expected_stdout, observed_stdout):
        reasons.append("stdout_mismatch")

    if ground_truth_output_signature.get("return_value") is not None:
        if _normalize_jsonish(replay_output_signature.get("return_value")) != _normalize_jsonish(ground_truth_output_signature.get("return_value")):
            reasons.append("return_value_mismatch")

    expected_paths = [
        str(item.get("path") or "").strip()
        for item in (ground_truth_output_signature.get("file_artifacts") or [])
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    observed_paths = [
        str(item.get("path") or "").strip()
        for item in (replay_output_signature.get("file_artifacts") or [])
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    ]
    if sorted(expected_paths) != sorted(observed_paths):
        if sorted(_basename(path) for path in expected_paths) != sorted(_basename(path) for path in observed_paths):
            reasons.append("file_artifacts_mismatch")
    return not reasons, reasons


def _stdout_semantically_equal(expected_stdout: str, observed_stdout: str) -> bool:
    expected = _normalize_stdout(expected_stdout)
    observed = _normalize_stdout(observed_stdout)
    if expected == observed:
        return True
    expected_lines = expected.splitlines()
    observed_lines = observed.splitlines()
    if len(expected_lines) != len(observed_lines):
        return False
    return all(_stdout_line_semantically_equal(left, right) for left, right in zip(expected_lines, observed_lines))


def _stdout_line_semantically_equal(expected_line: str, observed_line: str) -> bool:
    expected = expected_line.strip()
    observed = observed_line.strip()
    if expected == observed:
        return True
    expected_literal = _parse_stdout_literal(expected)
    observed_literal = _parse_stdout_literal(observed)
    if expected_literal["parsed"] and observed_literal["parsed"]:
        return expected_literal["prefix"] == observed_literal["prefix"] and expected_literal["value"] == observed_literal["value"]
    return False


def _parse_stdout_literal(line: str) -> dict[str, Any]:
    stripped = line.strip()
    for prefix, literal_text in _stdout_literal_candidates(stripped):
        try:
            value = ast.literal_eval(literal_text)
        except (SyntaxError, ValueError):
            continue
        return {"parsed": True, "prefix": prefix, "value": value}
    return {"parsed": False, "prefix": "", "value": None}


def _stdout_literal_candidates(line: str) -> list[tuple[str, str]]:
    candidates = [("", line)]
    prefix, sep, rest = line.partition(":")
    if sep and rest.strip():
        candidates.append((prefix.strip(), rest.strip()))
    return candidates


def _extract_target_name(signature: str, code: str) -> str | None:
    signature = str(signature or "").strip()
    if signature.startswith("def "):
        fragment = signature[4:]
        return fragment.split("(", 1)[0].strip() or None
    if signature.startswith("class "):
        fragment = signature[6:]
        return fragment.split("(", 1)[0].split(":", 1)[0].strip() or None
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
    return None


def _extract_seed_case_rule_based(row: dict[str, Any], code: str) -> dict[str, Any] | None:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    target_name = _extract_target_name(str(row.get("signature") or ""), code)
    if not target_name:
        return None
    main_fn = next((node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    if main_fn is None:
        return None
    extracted = _extract_case_from_statements(code, main_fn.body, target_name)
    if extracted is None:
        extracted = _extract_case_from_top_level_and_main_flow(code, module, target_name)
    if extracted is None:
        return None
    extracted["description"] = f"Replay the canonical main() seed flow for {target_name}."
    extracted["case_id"] = _SEED_CASE_ID
    extracted["assumptions"] = []
    return extracted


def _extract_seed_case_heuristic(row: dict[str, Any], code: str) -> dict[str, Any] | None:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    target_name = _extract_target_name(str(row.get("signature") or ""), code)
    if not target_name:
        return None
    candidate_statements = [
        node
        for node in module.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
    ]
    assignment_names = _collect_target_call_assignment_names(candidate_statements, target_name)
    if assignment_names:
        result_assignment = (
            f"result = {assignment_names[0]}"
            if len(assignment_names) == 1
            else "result = {" + ", ".join(f"{name!r}: {name}" for name in assignment_names) + "}"
        )
        return {
            "description": f"Replay the discovered top-level seed demo flow for {target_name}.",
            "case_id": _SEED_CASE_ID,
            "setup_code": "",
            "call_code": result_assignment,
            "assumptions": [],
        }
    extracted = _extract_case_from_statements(code, candidate_statements, target_name)
    if extracted is None:
        return None
    extracted["description"] = f"Replay the discovered top-level seed demo flow for {target_name}."
    extracted["case_id"] = _SEED_CASE_ID
    extracted["assumptions"] = []
    return extracted


def _extract_case_from_statements(code: str, statements: list[ast.stmt], target_name: str) -> dict[str, Any] | None:
    if not statements:
        return None
    first_call_index = None
    for index, statement in enumerate(statements):
        if _statement_references_target(statement, target_name):
            first_call_index = index
            break
    if first_call_index is None:
        return None

    setup_statements = statements[:first_call_index]
    call_statements = list(statements[first_call_index:])
    call_assignments = _collect_target_call_assignment_names(call_statements, target_name)
    call_code = _source_for_statements(code, call_statements)
    if not call_code.strip():
        return None
    if "result =" not in call_code:
        if call_assignments:
            if len(call_assignments) == 1:
                call_code = f"{call_code}\nresult = {call_assignments[0]}"
            else:
                mapping = ", ".join(f"{name!r}: {name}" for name in call_assignments)
                call_code = f"{call_code}\nresult = {{{mapping}}}"
        else:
            rewritten = _rewrite_direct_call_to_result(code, call_statements[0], target_name)
            if rewritten is not None:
                remainder = _source_for_statements(code, call_statements[1:])
                call_code = "\n".join(part for part in [rewritten, remainder] if part.strip())
            else:
                call_code = f"{call_code}\nresult = None"

    return {
        "setup_code": _source_for_statements(code, setup_statements),
        "call_code": call_code.strip(),
    }


def _extract_case_from_top_level_and_main_flow(code: str, module: ast.Module, target_name: str) -> dict[str, Any] | None:
    main_fn = next((node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    if main_fn is None:
        return None

    top_level_statements = [
        node
        for node in module.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.If))
    ]
    assignment_names = _collect_target_call_assignment_names(top_level_statements, target_name)
    if not assignment_names:
        return None

    result_assignment = (
        f"result = {assignment_names[0]}"
        if len(assignment_names) == 1
        else "result = {" + ", ".join(f"{name!r}: {name}" for name in assignment_names) + "}"
    )
    call_code = _source_for_statements(code, main_fn.body)
    if not call_code.strip():
        return None
    call_code = f"{call_code}\n{result_assignment}"
    return {
        "setup_code": "",
        "call_code": call_code.strip(),
    }


def _statement_contains_target_call(statement: ast.stmt, target_name: str) -> bool:
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == target_name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == target_name:
                return True
    return False


def _statement_references_target(statement: ast.stmt, target_name: str) -> bool:
    if _statement_contains_target_call(statement, target_name):
        return True
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and node.id == target_name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == target_name:
            return True
    return False


def _collect_target_call_assignment_names(statements: list[ast.stmt], target_name: str) -> list[str]:
    names: list[str] = []
    for statement in statements:
        if isinstance(statement, ast.Assign) and _statement_contains_target_call(statement, target_name):
            for target in statement.targets:
                names.extend(_collect_assignment_target_names(target))
        elif isinstance(statement, ast.AnnAssign) and _statement_contains_target_call(statement, target_name):
            names.extend(_collect_assignment_target_names(statement.target))
    return names


def _collect_assignment_target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_collect_assignment_target_names(elt))
        return names
    return []


def _rewrite_direct_call_to_result(code: str, statement: ast.stmt, target_name: str) -> str | None:
    if not isinstance(statement, ast.Expr):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    func = statement.value.func
    if isinstance(func, ast.Name) and func.id == target_name:
        return f"result = {ast.get_source_segment(code, statement.value) or ast.unparse(statement.value)}"
    if isinstance(func, ast.Attribute) and func.attr == target_name:
        return f"result = {ast.get_source_segment(code, statement.value) or ast.unparse(statement.value)}"
    return None


def _source_for_statements(code: str, statements: list[ast.stmt]) -> str:
    if not statements:
        return ""
    parts: list[str] = []
    for statement in statements:
        segment = ast.get_source_segment(code, statement)
        if segment is None:
            lines = code.splitlines()
            start = getattr(statement, "lineno", None)
            end = getattr(statement, "end_lineno", None)
            if start is None or end is None:
                continue
            segment = "\n".join(lines[start - 1 : end])
        parts.append(_normalize_statement_source(segment, int(getattr(statement, "col_offset", 0) or 0)))
    return "\n".join(parts).strip()


def _normalize_statement_source(segment: str, base_indent: int) -> str:
    lines = segment.splitlines()
    if not lines:
        return ""
    normalized = [lines[0].lstrip()]
    for line in lines[1:]:
        if not line.strip():
            normalized.append("")
            continue
        if base_indent > 0 and len(line) >= base_indent:
            normalized.append(line[base_indent:])
        else:
            normalized.append(line.lstrip())
    return "\n".join(normalized)


def _reconstruct_seed_case_with_llm(
    row: dict[str, Any],
    code: str,
    ground_truth_output_signature: dict[str, Any],
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
) -> dict[str, Any] | None:
    prompt = prompt_runner.render(
        "seed_execution_case_reconstruct.jinja",
        seed_problem=str(row.get("problem") or row.get("instruction") or ""),
        signature_info=str(row.get("signature") or ""),
        seed_context=str(row.get("context") or ""),
        seed_executable_code=code,
        observed_ground_truth_output_signature=json.dumps(ground_truth_output_signature, ensure_ascii=False, indent=2),
    )
    response = llm_client.complete_json(
        task_name="seed_execution_case_reconstruct",
        prompt=prompt,
        context={
            "task_id": row.get("task_id") or row.get("idx"),
            "signature": row.get("signature"),
        },
        mock_builder=_mock_seed_execution_case_builder,
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    case = payload.get("seed_execution_case")
    if not isinstance(case, dict):
        return None
    return case


def _mock_seed_execution_case_builder(context: dict[str, Any]) -> dict[str, Any]:
    target_name = _extract_target_name(str(context.get("signature") or ""), "")
    call_code = f"result = {target_name}()" if target_name else "result = None"
    return {
        "seed_execution_case": {
            "case_id": _SEED_CASE_ID,
            "description": "Mock-reconstructed seed execution case.",
            "setup_code": "",
            "call_code": call_code,
            "expected_output_signature": {
                "return_value": None,
                "stdout_contains": [],
                "file_artifacts": [],
            },
            "extraction_method": "llm_reconstructed",
            "validated_against_ground_truth": False,
            "assumptions": [],
        }
    }


def _snapshot_files(workdir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workdir).as_posix()
        data = path.read_bytes()
        files[rel] = {
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return files


def _artifact_diff(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for rel_path, meta in sorted(after.items()):
        if before.get(rel_path) != meta:
            changed.append({"path": rel_path, **meta})
    return changed


def _basename(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").rstrip("/")
    return normalized.split("/")[-1] if normalized else ""


def _wrapper_source() -> str:
    return """from __future__ import annotations

import contextlib
import hashlib
import io
import json
import runpy
import traceback
from pathlib import Path

CANDIDATE = "candidate.py"
RESULT = "execution_result.json"
WRAPPER = "run_candidate.py"
EXCLUDED = {CANDIDATE, RESULT, WRAPPER}


def _normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return repr(value)


def _snapshot_files():
    files = {}
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(".").as_posix()
        if rel in EXCLUDED or rel.startswith("__pycache__/"):
            continue
        data = path.read_bytes()
        files[rel] = {
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return files


def main():
    before = _snapshot_files()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    payload = {"status": "fail", "return_value": None}
    namespace = {}
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            namespace = runpy.run_path(CANDIDATE, run_name="__main__")
        payload["status"] = "pass"
        if "answer" in namespace:
            payload["return_value"] = _normalize(namespace["answer"])
        elif "result" in namespace:
            payload["return_value"] = _normalize(namespace["result"])
    except Exception as exc:
        payload["error_type"] = exc.__class__.__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    finally:
        payload["stdout"] = stdout_buffer.getvalue()
        payload["stderr"] = stderr_buffer.getvalue()
        after = _snapshot_files()
        payload["artifacts"] = []
        for rel, meta in sorted(after.items()):
            if before.get(rel) != meta:
                payload["artifacts"].append({"path": rel, **meta})
    Path(RESULT).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
"""


def _seed_case_wrapper_source() -> str:
    return """from __future__ import annotations

import contextlib
import hashlib
import io
import json
import traceback
from pathlib import Path

CANDIDATE = "candidate.py"
RESULT = "execution_result.json"
SEED_CASE = "seed_case.json"
WRAPPER = "run_seed_case.py"
EXCLUDED = {CANDIDATE, RESULT, SEED_CASE, WRAPPER}


def _normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return repr(value)


def _snapshot_files():
    files = {}
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(".").as_posix()
        if rel in EXCLUDED or rel.startswith("__pycache__/"):
            continue
        data = path.read_bytes()
        files[rel] = {
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return files


def main():
    seed_case = json.loads(Path(SEED_CASE).read_text(encoding="utf-8"))
    before = _snapshot_files()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    payload = {"status": "fail", "return_value": None}
    namespace = {"__name__": "__seed_case_runtime__"}
    try:
        source = Path(CANDIDATE).read_text(encoding="utf-8")
        compiled = compile(source, CANDIDATE, "exec")
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compiled, namespace, namespace)
            exec(str(seed_case.get("setup_code") or ""), namespace, namespace)
            exec(str(seed_case.get("call_code") or ""), namespace, namespace)
        payload["status"] = "pass"
        payload["return_value"] = _normalize(namespace.get("result"))
    except Exception as exc:
        payload["error_type"] = exc.__class__.__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    finally:
        payload["stdout"] = stdout_buffer.getvalue()
        payload["stderr"] = stderr_buffer.getvalue()
        after = _snapshot_files()
        payload["artifacts"] = []
        for rel, meta in sorted(after.items()):
            if before.get(rel) != meta:
                payload["artifacts"].append({"path": rel, **meta})
    Path(RESULT).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
"""
