from __future__ import annotations

import contextlib
import hashlib
import io
import json
import numbers
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from medenvscale.execution_lock import WORKING_DIRECTORY_LOCK
from medenvscale.scaling.output_constraints import check_output_constraints, output_constraints_from_scaled_oracle_cases
from medenvscale.scaling.output_signature import materialize_executable_gold_code
from medenvscale.scaling.runtime_value_sanitizer import stabilize_runtime_value
from medenvscale.schemas import ExecutableEnvSpec


def run_scaled_gold_on_validated_oracle_cases(
    env: ExecutableEnvSpec,
    candidate_code: str,
    validated_oracle_cases: list[dict[str, Any]],
    python_bin: str | None = None,
) -> dict[str, Any]:
    materialized_code = materialize_executable_gold_code(env, candidate_code)
    compile_error = _compile_candidate(materialized_code, env.env_id)
    if compile_error is not None:
        rows = [
            {
                "env_id": env.env_id,
                "level": str((env.difficulty.global_level if env.difficulty else "") or ""),
                "case_id": str(case.get("case_id") or f"case_{index}"),
                "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
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
        row = (
            execute_oracle_case_subprocess(materialized_code, env, case, python_bin=str(python_bin))
            if python_bin and str(python_bin) != sys.executable
            else execute_oracle_case(materialized_code, env, case)
        )
        rows.append(row)
        all_passed = all_passed and bool(row["passed"])
    return {
        "compile_passed": True,
        "execution_passed": all_passed,
        "case_reports": rows,
        "materialized_code": materialized_code,
        "failure_reasons": [reason for row in rows for reason in row.get("failure_reasons", [])],
    }


def execute_oracle_case_subprocess(materialized_code: str, env: ExecutableEnvSpec, case: dict[str, Any], python_bin: str) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "oracle_case")
    expected = dict(case.get("expected_output_signature") or {})
    level = str((env.difficulty.global_level if env.difficulty else "") or "")
    marker = "__MEDENVSCALE_CASE_RESULT__"
    wrapper = _subprocess_case_wrapper(
        materialized_code=materialized_code,
        setup_code=str(case.get("setup_code") or ""),
        call_code=str(case.get("call_code") or ""),
        marker=marker,
    )
    with tempfile.TemporaryDirectory(prefix="medenvscale-case-") as temp_dir:
        workdir = Path(temp_dir)
        script_path = workdir / "case_runner.py"
        script_path.write_text(wrapper, encoding="utf-8")
        try:
            completed = subprocess.run(
                [python_bin, str(script_path)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "env_id": env.env_id,
                "level": level,
                "case_id": case_id,
                "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
                "passed": False,
                "observed_output_signature": {"return_value": None, "stdout": "", "file_artifacts": [], "return_type": None},
                "expected_output_signature": expected,
                "failure_reasons": ["CASE_EXECUTION_ERROR:TimeoutExpired:case timed out"],
            }
    payload = _parse_subprocess_case_result(completed.stdout, marker)
    if completed.returncode != 0 and payload is None:
        return {
            "env_id": env.env_id,
            "level": level,
            "case_id": case_id,
            "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
            "passed": False,
            "observed_output_signature": {"return_value": None, "stdout": "", "file_artifacts": [], "return_type": None},
            "expected_output_signature": expected,
            "failure_reasons": [f"CASE_EXECUTION_ERROR:SubprocessFailed:{_tail(completed.stderr or completed.stdout)}"],
        }
    if not payload or not payload.get("ok"):
        error_type = str((payload or {}).get("error_type") or "SubprocessCaseError")
        error = str((payload or {}).get("error") or "case subprocess did not return an observation")
        return {
            "env_id": env.env_id,
            "level": level,
            "case_id": case_id,
            "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
            "passed": False,
            "observed_output_signature": {"return_value": None, "stdout": "", "file_artifacts": [], "return_type": None},
            "expected_output_signature": expected,
            "failure_reasons": [f"CASE_EXECUTION_ERROR:{error_type}:{error}"],
        }
    observed = payload.get("observed") or {}
    spec = output_constraints_from_scaled_oracle_cases([case])
    comparison = check_output_constraints(observed, spec)
    failure_reasons = [f"{item.get('check_id')}:{item.get('reason')}" for item in comparison.get("failed_checks", [])]
    return {
        "env_id": env.env_id,
        "level": level,
        "case_id": case_id,
        "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
        "passed": comparison.get("passed", False),
        "observed_output_signature": observed,
        "expected_output_signature": expected,
        "failure_reasons": failure_reasons,
    }


def execute_oracle_case(materialized_code: str, env: ExecutableEnvSpec, case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "oracle_case")
    expected = dict(case.get("expected_output_signature") or {})
    compiled = compile(materialized_code, f"{env.env_id}_{case_id}_candidate.py", "exec")
    level = str((env.difficulty.global_level if env.difficulty else "") or "")

    with WORKING_DIRECTORY_LOCK:
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
                    "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
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
        "covered_requirement_ids": list(case.get("covered_requirement_ids") or []),
        "passed": comparison.get("passed", False),
        "observed_output_signature": observed,
        "expected_output_signature": expected,
        "failure_reasons": failure_reasons,
    }


def _subprocess_case_wrapper(*, materialized_code: str, setup_code: str, call_code: str, marker: str) -> str:
    return (
        "import contextlib, hashlib, io, json, os, traceback\n"
        "from pathlib import Path\n\n"
        f"MARKER = {json.dumps(marker)}\n"
        f"MATERIALIZED_CODE = {json.dumps(materialized_code)}\n"
        f"SETUP_CODE = {json.dumps(setup_code)}\n"
        f"CALL_CODE = {json.dumps(call_code)}\n\n"
        "def _snapshot_files(workdir):\n"
        "    files = {}\n"
        "    for path in workdir.rglob('*'):\n"
        "        if not path.is_file():\n"
        "            continue\n"
        "        rel = path.relative_to(workdir).as_posix()\n"
        "        data = path.read_bytes()\n"
        "        files[rel] = {'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}\n"
        "    return files\n\n"
        "def _artifact_diff(before, after):\n"
        "    changed = []\n"
        "    for rel_path, meta in sorted(after.items()):\n"
        "        if before.get(rel_path) != meta:\n"
        "            row = {'path': rel_path}\n"
        "            row.update(meta)\n"
        "            changed.append(row)\n"
        "    return changed\n\n"
        "def _normalize_stdout(text):\n"
        "    normalized = text.replace('\\r\\n', '\\n').replace('\\r', '\\n')\n"
        "    return '\\n'.join(line.rstrip() for line in normalized.split('\\n')).strip()\n\n"
        "def _normalize_runtime_value(value):\n"
        "    if value is None or type(value) in (bool, int, float, str):\n"
        "        return value\n"
        "    if hasattr(value, 'tolist') and not isinstance(value, (bytes, bytearray)):\n"
        "        try:\n"
        "            return _normalize_runtime_value(value.tolist())\n"
        "        except Exception:\n"
        "            pass\n"
        "    if isinstance(value, list):\n"
        "        return [_normalize_runtime_value(item) for item in value]\n"
        "    if isinstance(value, tuple):\n"
        "        return [_normalize_runtime_value(item) for item in value]\n"
        "    if isinstance(value, dict):\n"
        "        return {str(key): _normalize_runtime_value(item) for key, item in value.items()}\n"
        "    try:\n"
        "        return repr(value)\n"
        "    except Exception:\n"
        "        return str(value)\n\n"
        "def _emit(payload):\n"
        "    print(MARKER + json.dumps(payload, ensure_ascii=False, default=repr))\n\n"
        "try:\n"
        "    workdir = Path.cwd()\n"
        "    namespace = {'__name__': '__case_runtime__'}\n"
        "    exec(compile(MATERIALIZED_CODE, 'candidate.py', 'exec'), namespace, namespace)\n"
        "    before_files = _snapshot_files(workdir)\n"
        "    stdout_buffer = io.StringIO()\n"
        "    stderr_buffer = io.StringIO()\n"
        "    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):\n"
        "        exec(SETUP_CODE, namespace, namespace)\n"
        "        exec(CALL_CODE, namespace, namespace)\n"
        "    after_files = _snapshot_files(workdir)\n"
        "    result = namespace.get('result')\n"
        "    observed = {\n"
        "        'return_value': _normalize_runtime_value(result),\n"
        "        'return_type': type(result).__name__ if 'result' in namespace else None,\n"
        "        'stdout': _normalize_stdout(stdout_buffer.getvalue()),\n"
        "        'stderr': _normalize_stdout(stderr_buffer.getvalue()),\n"
        "        'file_artifacts': _artifact_diff(before_files, after_files),\n"
        "    }\n"
        "    _emit({'ok': True, 'observed': observed})\n"
        "except Exception as exc:\n"
        "    _emit({'ok': False, 'error_type': exc.__class__.__name__, 'error': str(exc), 'traceback': traceback.format_exc()[-2000:]})\n"
        "    raise SystemExit(1)\n"
    )


def _parse_subprocess_case_result(stdout: str, marker: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout or "").splitlines()):
        if marker in line:
            _, _, payload = line.partition(marker)
            try:
                parsed = json.loads(payload)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _tail(text: str, limit: int = 1000) -> str:
    return str(text or "")[-limit:]


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
    if value is None or type(value) in (bool, int, float, str):
        return stabilize_runtime_value(value)
    if isinstance(value, numbers.Integral):
        return stabilize_runtime_value(int(value))
    if isinstance(value, numbers.Real):
        return stabilize_runtime_value(float(value))
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return stabilize_runtime_value(_normalize_runtime_value(value.tolist()))
        except Exception:
            pass
    if isinstance(value, list):
        return stabilize_runtime_value([_normalize_runtime_value(item) for item in value])
    if isinstance(value, tuple):
        return stabilize_runtime_value([_normalize_runtime_value(item) for item in value])
    if isinstance(value, dict):
        return stabilize_runtime_value({str(key): _normalize_runtime_value(item) for key, item in value.items()})
    try:
        return stabilize_runtime_value(repr(value))
    except Exception:
        return stabilize_runtime_value(str(value))
