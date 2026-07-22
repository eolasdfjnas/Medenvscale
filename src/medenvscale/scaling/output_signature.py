from __future__ import annotations

import contextlib
import hashlib
import io
import numbers
import os
import tempfile
from pathlib import Path
from typing import Any

from medenvscale.execution_lock import WORKING_DIRECTORY_LOCK
from medenvscale.schemas import ExecutableEnvSpec


def materialize_executable_gold_code(env: ExecutableEnvSpec, candidate_solution: str) -> str:
    return str(candidate_solution or "").strip()


def execute_materialized_code(materialized_code: str, env_id: str) -> dict[str, Any]:
    try:
        compiled = compile(materialized_code, f"{env_id}_scaled_gold.py", "exec")
    except SyntaxError as exc:
        return _failed_result(
            failure_reason="compile_failed",
            runtime_error={"error_type": "SyntaxError", "error_message": exc.msg, "traceback": ""},
        )

    with WORKING_DIRECTORY_LOCK:
        with tempfile.TemporaryDirectory(prefix="medenvscale-stage05-") as temp_dir:
            workdir = Path(temp_dir)
            before_files = _snapshot_files(workdir)
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            namespace: dict[str, Any] = {"__name__": "__main__"}
            original_cwd = Path.cwd()
            try:
                os.chdir(workdir)
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    exec(compiled, namespace, namespace)
            except Exception as exc:
                after_files = _snapshot_files(workdir)
                os.chdir(original_cwd)
                return {
                    "status": "fail",
                    "compile_passed": True,
                    "execution_passed": False,
                    "output_signature": _build_output_signature(
                        namespace=namespace,
                        stdout=stdout_buffer.getvalue(),
                        after_files=after_files,
                        before_files=before_files,
                    ),
                    "runtime_error": {
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                        "traceback": "",
                    },
                    "failure_reason": "execution_failed",
                    "stdout": _normalize_stdout(stdout_buffer.getvalue()),
                    "stderr": _normalize_stdout(stderr_buffer.getvalue()),
                }
            finally:
                if Path.cwd() != original_cwd:
                    os.chdir(original_cwd)

            after_files = _snapshot_files(workdir)
            output_signature = _build_output_signature(
                namespace=namespace,
                stdout=stdout_buffer.getvalue(),
                after_files=after_files,
                before_files=before_files,
            )
        observable = bool(
            output_signature["stdout"]
            or output_signature["file_artifacts"]
            or output_signature["return_value"] is not None
        )
        return {
            "status": "pass" if observable else "fail",
            "compile_passed": True,
            "execution_passed": observable,
            "output_signature": output_signature,
            "runtime_error": {"error_type": None, "error_message": None, "traceback": None},
            "failure_reason": None if observable else "no_observable_output_signature",
            "stdout": output_signature["stdout"],
            "stderr": _normalize_stdout(stderr_buffer.getvalue()),
        }


def execute_candidate_solution(env: ExecutableEnvSpec, candidate_solution: str) -> dict[str, Any]:
    materialized = materialize_executable_gold_code(env, candidate_solution)
    result = execute_materialized_code(materialized, env.env_id)
    result["materialized_code"] = materialized
    result["candidate_solution"] = candidate_solution
    return result


def _build_output_signature(
    namespace: dict[str, Any],
    stdout: str,
    after_files: dict[str, dict[str, Any]],
    before_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return_value = _extract_return_value(namespace)
    return {
        "return_value": _normalize_runtime_value(return_value),
        "return_type": type(return_value).__name__ if return_value is not None else None,
        "stdout": _normalize_stdout(stdout),
        "file_artifacts": _artifact_diff(before_files, after_files),
    }


def _extract_return_value(namespace: dict[str, Any]) -> Any:
    if "answer" in namespace:
        return namespace["answer"]
    if "result" in namespace:
        return namespace["result"]
    return None


def _artifact_diff(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for rel_path, meta in sorted(after.items()):
        if before.get(rel_path) != meta:
            changed.append({"path": rel_path, **meta})
    return changed


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


def _normalize_stdout(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _normalize_runtime_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return _normalize_runtime_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, list):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_runtime_value(item) for key, item in value.items()}
    try:
        return repr(value)
    except Exception:
        return str(value)


def _failed_result(failure_reason: str, runtime_error: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "fail",
        "compile_passed": False,
        "execution_passed": False,
        "output_signature": {
            "return_value": None,
            "return_type": None,
            "stdout": "",
            "file_artifacts": [],
        },
        "runtime_error": runtime_error,
        "failure_reason": failure_reason,
        "stdout": "",
        "stderr": "",
    }
