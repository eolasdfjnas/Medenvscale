from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any

from medenvscale.scaling.output_constraints import check_output_constraints, output_constraints_from_scaled_oracle_cases
from medenvscale.scaling.output_signature import materialize_executable_gold_code
from medenvscale.schemas import ExecutableEnvSpec


def run_scaled_gold_on_validated_oracle_cases(
    env: ExecutableEnvSpec,
    candidate_code: str,
    validated_oracle_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    materialized_code = materialize_executable_gold_code(env, candidate_code)
    compile_error = _compile_candidate(materialized_code, env.env_id)
    if compile_error is not None:
        rows = [
            {
                "env_id": env.env_id,
                "level": str((env.difficulty.global_level if env.difficulty else "") or ""),
                "case_id": str(case.get("case_id") or f"case_{index}"),
                "passed": False,
                "observed_output_signature": {"return_value": None, "stdout": "", "file_artifacts": [], "return_type": None},
                "expected_output_signature": dict(case.get("expected_output_signature") or {}),
                "failure_reasons": [f"SCALED_GOLD_COMPILE_FAILED:{compile_error}"],
            }
            for index, case in enumerate(validated_oracle_cases or [], start=1)
        ]
        return {
            "compile_passed": False,
            "execution_passed": False,
            "case_reports": rows,
            "materialized_code": materialized_code,
            "failure_reasons": [f"SCALED_GOLD_COMPILE_FAILED:{compile_error}"],
        }

    rows: list[dict[str, Any]] = []
    all_passed = True
    for case in validated_oracle_cases or []:
        row = execute_oracle_case(materialized_code, env, case)
        rows.append(row)
        all_passed = all_passed and bool(row["passed"])
    return {
        "compile_passed": True,
        "execution_passed": all_passed,
        "case_reports": rows,
        "materialized_code": materialized_code,
        "failure_reasons": [reason for row in rows for reason in row.get("failure_reasons", [])],
    }


def execute_oracle_case(materialized_code: str, env: ExecutableEnvSpec, case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "oracle_case")
    expected = dict(case.get("expected_output_signature") or {})
    compiled = compile(materialized_code, f"{env.env_id}_{case_id}_candidate.py", "exec")
    level = str((env.difficulty.global_level if env.difficulty else "") or "")

    with tempfile.TemporaryDirectory(prefix="medenvscale-case-") as temp_dir:
        workdir = Path(temp_dir)
        namespace: dict[str, Any] = {"__name__": "__case_runtime__"}
        original_cwd = Path.cwd()
        try:
            os.chdir(workdir)
            exec(compiled, namespace, namespace)
            before_files = _snapshot_files(workdir)
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(str(case.get("setup_code") or ""), namespace, namespace)
                exec(str(case.get("call_code") or ""), namespace, namespace)
            after_files = _snapshot_files(workdir)
        except Exception as exc:
            if Path.cwd() != original_cwd:
                os.chdir(original_cwd)
            return {
                "env_id": env.env_id,
                "level": level,
                "case_id": case_id,
                "passed": False,
                "observed_output_signature": {"return_value": None, "stdout": "", "file_artifacts": [], "return_type": None},
                "expected_output_signature": expected,
                "failure_reasons": [f"CASE_EXECUTION_ERROR:{exc.__class__.__name__}:{exc}"],
            }
        finally:
            if Path.cwd() != original_cwd:
                os.chdir(original_cwd)

    observed = {
        "return_value": _normalize_runtime_value(namespace.get("result")),
        "return_type": type(namespace.get("result")).__name__ if "result" in namespace else None,
        "stdout": _normalize_stdout(stdout_buffer.getvalue()),
        "stderr": _normalize_stdout(stderr_buffer.getvalue()),
        "file_artifacts": _artifact_diff(before_files, after_files),
    }
    spec = output_constraints_from_scaled_oracle_cases([case])
    comparison = check_output_constraints(observed, spec)
    failure_reasons = [f"{item.get('check_id')}:{item.get('reason')}" for item in comparison.get("failed_checks", [])]
    return {
        "env_id": env.env_id,
        "level": level,
        "case_id": case_id,
        "passed": comparison.get("passed", False),
        "observed_output_signature": observed,
        "expected_output_signature": expected,
        "failure_reasons": failure_reasons,
    }


def _compile_candidate(materialized_code: str, env_id: str) -> str | None:
    try:
        compile(materialized_code, f"{env_id}_scaled_gold.py", "exec")
        return None
    except SyntaxError as exc:
        return exc.msg


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


def _normalize_stdout(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _normalize_runtime_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_runtime_value(item) for key, item in value.items()}
    return repr(value)
