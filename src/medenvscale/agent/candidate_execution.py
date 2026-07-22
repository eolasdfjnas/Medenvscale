from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from medenvscale.scaling.path_safety import analyze_relative_path, extract_code_path_references


FORBIDDEN_IMPORT_ROOTS = {
    "ftplib",
    "http",
    "socket",
    "subprocess",
    "telnetlib",
    "urllib",
}
UNAVAILABLE_IMPORT_ROOTS = {
    "my_module",
    "requests_toolbelt",
}
FORBIDDEN_CALLS = {
    ("os", "popen"),
    ("os", "spawnl"),
    ("os", "spawnle"),
    ("os", "spawnlp"),
    ("os", "spawnlpe"),
    ("os", "spawnv"),
    ("os", "spawnve"),
    ("os", "spawnvp"),
    ("os", "spawnvpe"),
    ("os", "system"),
}


def check_syntax(code: str) -> dict[str, Any]:
    try:
        compile(str(code or ""), "candidate.py", "exec")
        return {"ok": True, "error_type": None, "message": "", "line": None}
    except SyntaxError as exc:
        return {"ok": False, "error_type": "SyntaxError", "message": exc.msg, "line": exc.lineno}
    except Exception as exc:
        return {"ok": False, "error_type": exc.__class__.__name__, "message": str(exc), "line": None}


def check_target_signature(code: str, expected_signature: str | None) -> dict[str, Any]:
    target_name = _target_name(expected_signature or "")
    if not target_name:
        return {"ok": True, "target_name": "", "signature_found": "", "problems": []}
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError as exc:
        return {
            "ok": False,
            "target_name": target_name,
            "signature_found": "",
            "problems": [f"candidate_not_parseable:{exc.msg}"],
        }
    node = _find_target(tree, target_name)
    if node is None:
        return {
            "ok": False,
            "target_name": target_name,
            "signature_found": "",
            "problems": [f"missing_target:{target_name}"],
        }
    found = _signature_from_node(node)
    expected_args = _target_arg_names(expected_signature or "")
    found_args = _arg_names(node)
    problems = []
    if expected_args and expected_args != found_args[: len(expected_args)]:
        problems.append(f"signature_args_mismatch:expected={expected_args}:found={found_args}")
    return {
        "ok": not problems,
        "target_name": target_name,
        "signature_found": found,
        "problems": problems,
    }


def validate_candidate_code(code: str, expected_signature: str | None) -> dict[str, Any]:
    syntax = check_syntax(code)
    signature = check_target_signature(code, expected_signature) if syntax["ok"] else {
        "ok": False,
        "target_name": _target_name(expected_signature or ""),
        "signature_found": "",
        "problems": ["syntax_failed"],
    }
    safety = check_static_safety(code)
    return {
        "ok": bool(syntax["ok"] and signature["ok"] and safety["ok"]),
        "syntax": syntax,
        "signature": signature,
        "safety": safety,
    }


def preflight_final_code(code: str, expected_signature: str | None, python_bin: str | None = None) -> dict[str, Any]:
    value = str(code or "")
    errors: list[str] = []
    repair_hints: list[str] = []

    if not value.strip():
        errors.append("EMPTY_FINAL_CODE")
        repair_hints.append("Submit complete executable Python code in the code argument.")

    if _looks_like_markdown_or_wrapped_text(value):
        errors.append("MARKDOWN_OR_NATURAL_LANGUAGE_OUTPUT")
        repair_hints.append("Submit raw Python source only; remove Markdown fences, prose, and JSON wrappers from final_code.")

    validation = validate_candidate_code(value, expected_signature)
    if not validation["ok"]:
        errors.append("VALIDATION_FAILED")
        repair_hints.append("Fix syntax, target signature, static safety, or unsafe path issues before final submission.")

    import_check = check_import_availability(value, python_bin=python_bin)
    if not import_check["ok"]:
        errors.extend(import_check["problems"])
        repair_hints.extend(import_check["repair_hints"])

    deduped_errors = list(dict.fromkeys(errors))
    deduped_hints = list(dict.fromkeys(repair_hints))
    return {
        "ok": not deduped_errors,
        "errors": deduped_errors,
        "repair_hints": deduped_hints,
        "validation": validation,
        "import_check": import_check,
    }


def run_custom_test(
    code: str,
    test_snippet: str,
    timeout_seconds: int | float = 5,
    fixture_files: list[dict[str, Any]] | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    safety = check_static_safety(str(code or "") + "\n" + str(test_snippet or ""))
    if not safety["ok"]:
        return {
            "ok": False,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "traceback_tail": "",
            "artifacts": [],
            "failure_reasons": safety["problems"],
        }
    script = f"{code}\n\n# agent custom test\n{test_snippet}\n"
    return _run_script(script, timeout_seconds=timeout_seconds, fixture_files=fixture_files, python_bin=python_bin)


def run_candidate_code(code: str, timeout_seconds: int | float = 5, python_bin: str | None = None) -> dict[str, Any]:
    safety = check_static_safety(code)
    if not safety["ok"]:
        return {
            "ok": False,
            "exit_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "traceback_tail": "",
            "artifacts": [],
            "failure_reasons": safety["problems"],
        }
    return _run_script(str(code or ""), timeout_seconds=timeout_seconds, python_bin=python_bin)


def check_import_availability(code: str, python_bin: str | None = None) -> dict[str, Any]:
    problems: list[str] = []
    repair_hints: list[str] = []
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return {"ok": True, "problems": [], "repair_hints": []}

    for module, is_relative in _iter_imported_modules(tree):
        if is_relative:
            problems.append("RELATIVE_IMPORT_NOT_ALLOWED")
            repair_hints.append("Do not rely on relative or hidden local modules; inline the required helper behavior.")
            continue
        root = module.split(".", 1)[0]
        if root in UNAVAILABLE_IMPORT_ROOTS:
            problems.append(f"unavailable_import:{module}")
            repair_hints.append(
                f"Module {module!r} is not available to the agent environment. "
                "Remove this import and inline the required helper behavior or use an available dependency."
            )
            continue
        available = _module_available(root, python_bin=python_bin)
        if not available:
            problems.append(f"unavailable_import:{module}")
            repair_hints.append(
                f"Module {module!r} could not be imported in the execution environment. "
                "Use standard-library code, visible task resources, or inline the needed helper logic."
            )

    return {
        "ok": not problems,
        "problems": list(dict.fromkeys(problems)),
        "repair_hints": list(dict.fromkeys(repair_hints)),
    }


def _module_available(root: str, python_bin: str | None = None) -> bool:
    target_python = str(python_bin or sys.executable)
    if not target_python or target_python == sys.executable:
        try:
            return importlib.util.find_spec(root) is not None
        except (ImportError, AttributeError, TypeError, ValueError):
            return False
    try:
        completed = subprocess.run(
            [
                target_python,
                "-c",
                "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                root,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def check_static_safety(code: str) -> dict[str, Any]:
    problems: list[str] = []
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return {"ok": True, "problems": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    problems.append(f"forbidden_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                problems.append(f"forbidden_import:{node.module}")
        elif isinstance(node, ast.Call):
            call = _call_name(node)
            if call in FORBIDDEN_CALLS:
                problems.append(f"forbidden_call:{'.'.join(call)}")
    for reference in extract_code_path_references(str(code or "")):
        result = analyze_relative_path(reference.path, artifact=reference.operation in {"path_write_text", "path_write_bytes"})
        if not result.safe:
            problems.append(f"unsafe_path:{reference.path}:{result.reason}")
    return {"ok": not problems, "problems": problems}


def _looks_like_markdown_or_wrapped_text(code: str) -> bool:
    stripped = str(code or "").lstrip()
    if "```" in stripped:
        return True
    lower = stripped[:80].lower()
    prose_prefixes = (
        "here is",
        "here's",
        "sure",
        "the solution",
        "final code",
        "json",
    )
    return any(lower.startswith(prefix) for prefix in prose_prefixes)


def _iter_imported_modules(tree: ast.AST) -> list[tuple[str, bool]]:
    modules: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append((alias.name, False))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                modules.append((node.module or "", True))
            elif node.module:
                modules.append((node.module, False))
    return modules


def _run_script(
    script: str,
    timeout_seconds: int | float,
    fixture_files: list[dict[str, Any]] | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    timeout = max(1, min(float(timeout_seconds or 5), 15.0))
    with tempfile.TemporaryDirectory(prefix="medenvscale-agent-") as temp_dir:
        workdir = Path(temp_dir)
        script_path = workdir / "candidate_run.py"
        fixture_result = _materialize_fixture_files(workdir, fixture_files or [])
        if not fixture_result["ok"]:
            return {
                "ok": False,
                "exit_code": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "traceback_tail": "",
                "diagnosis": "Agent-created fixture files could not be materialized safely.",
                "artifacts": [],
                "failure_reasons": fixture_result["errors"],
            }
        before = _snapshot_files(workdir)
        script_path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(python_bin or sys.executable), str(script_path)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            after = _snapshot_files(workdir)
            traceback_tail = _traceback_tail(completed.stderr)
            return {
                "ok": completed.returncode == 0,
                "exit_code": completed.returncode,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
                "traceback_tail": traceback_tail,
                "diagnosis": _failure_diagnosis(traceback_tail or completed.stderr),
                "artifacts": _artifact_diff(before, after, exclude={"candidate_run.py"}),
                "failure_reasons": [] if completed.returncode == 0 else ["nonzero_exit"],
            }
        except subprocess.TimeoutExpired as exc:
            after = _snapshot_files(workdir)
            return {
                "ok": False,
                "exit_code": None,
                "stdout_tail": _tail(exc.stdout or ""),
                "stderr_tail": _tail(exc.stderr or ""),
                "traceback_tail": "",
                "diagnosis": "Execution timed out. Reduce long-running loops, network calls, or expensive computation in the candidate code.",
                "artifacts": _artifact_diff(before, after, exclude={"candidate_run.py"}),
                "failure_reasons": ["timeout"],
            }


def _materialize_fixture_files(workdir: Path, fixture_files: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(fixture_files) > 10:
        errors.append("too_many_fixture_files")
        return {"ok": False, "errors": errors}
    total_bytes = 0
    for item in fixture_files:
        path = str(item.get("path") or "")
        content = str(item.get("content") or "")
        safety = analyze_relative_path(path, artifact=True)
        if not safety.safe:
            errors.append(f"unsafe_fixture_path:{path}:{safety.reason}")
            continue
        data = content.encode("utf-8")
        total_bytes += len(data)
        if len(data) > 65536:
            errors.append(f"fixture_file_too_large:{safety.normalized_path}")
            continue
        if total_bytes > 262144:
            errors.append("fixture_files_total_too_large")
            continue
        target = workdir / safety.normalized_path
        try:
            target.relative_to(workdir)
        except ValueError:
            errors.append(f"unsafe_fixture_path:{path}:resolved_outside_workdir")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"ok": not errors, "errors": errors}


def _target_name(signature: str) -> str:
    match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", signature or "")
    return match.group(1) if match else ""


def _target_arg_names(signature: str) -> list[str]:
    if not signature.strip().startswith("def "):
        return []
    try:
        node = ast.parse(signature.rstrip(":") + ":\n    pass").body[0]
    except SyntaxError:
        return []
    return _arg_names(node) if isinstance(node, ast.FunctionDef) else []


def _find_target(tree: ast.AST, target_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target_name:
            return node
    return None


def _signature_from_node(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    args = ", ".join(_arg_names(node))
    return f"def {node.name}({args})"


def _arg_names(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    return [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]


def _call_name(node: ast.Call) -> tuple[str, str] | tuple[str]:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return (node.func.value.id, node.func.attr)
    if isinstance(node.func, ast.Name):
        return (node.func.id,)
    return ("",)


def _snapshot_files(workdir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workdir).as_posix()
        data = path.read_bytes()
        files[rel] = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return files


def _artifact_diff(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    changed = []
    for rel_path, meta in sorted(after.items()):
        if rel_path in excluded:
            continue
        if before.get(rel_path) != meta:
            changed.append({"path": rel_path, **meta})
    return changed


def _tail(text: str, limit: int = 4000) -> str:
    return str(text or "")[-limit:]


def _traceback_tail(stderr: str, limit: int = 4000) -> str:
    text = str(stderr or "")
    marker = "Traceback (most recent call last):"
    index = text.rfind(marker)
    return text[index:][-limit:] if index >= 0 else text[-limit:]


def _failure_diagnosis(stderr_or_traceback: str) -> str:
    text = str(stderr_or_traceback or "")
    missing = re.search(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", text)
    if missing:
        module = missing.group(1)
        return (
            f"Import failed because module {module!r} is unavailable in the execution environment. "
            "Remove this import and inline the required helper behavior, or replace it with a standard-library "
            "or visibly available dependency alternative before submitting final_code."
        )
    if "ImportError:" in text:
        return (
            "An import failed in the execution environment. Do not rely on hidden project-local modules or optional "
            "packages unless a tool check confirms availability; inline the needed helper behavior instead."
        )
    return ""
