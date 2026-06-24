from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import os
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec

"""
Hidden-test execution guards for trusted raw gold solutions.

Project-wide policy:
- Raw dataset gold solutions are treated as trusted reference answers.
- The pipeline does not run a gold-compatibility gate or emit gold compatibility reports.
- Stage 05 still checks whether the current environment and its hidden tests can compile
  and execute against the trusted reference solution.
- Execution output from task scaffolds is suppressed so verifier checks do not pollute
  the stage progress logs.
"""


@dataclass
class HiddenTestCheckResult:
    status: str
    errors: list[str]
    passed_tests: list[str]
    failed_tests: list[str]


def run_hidden_test_execution_check(env: ExecutableEnvSpec) -> HiddenTestCheckResult:
    materialized = materialize_solution_context(env)
    try:
        compile(materialized, f"{env.env_id}.py", "exec")
    except SyntaxError as exc:
        return HiddenTestCheckResult(
            status="fail",
            errors=[f"hidden_test_context_compile_error: {exc.msg}"],
            passed_tests=[],
            failed_tests=["context_compile_check"],
        )

    namespace: dict[str, Any] = {"__name__": "__main__", "inserted_solution_code": env.gold_solution}
    try:
        with _suppress_process_output():
            exec(materialized, namespace, namespace)
    except Exception as exc:
        return HiddenTestCheckResult(
            status="fail",
            errors=[f"hidden_test_context_execution_error: {exc.__class__.__name__}: {exc}"],
            passed_tests=["context_compile_check"],
            failed_tests=["context_execution_check"],
        )

    errors: list[str] = []
    passed_tests: list[str] = ["context_compile_check", "context_execution_check"]
    failed_tests: list[str] = []
    seen_test_ids: set[str] = set()

    for index, test in enumerate(env.hidden_tests, start=1):
        if not isinstance(test, dict):
            errors.append("Hidden test must be a dict")
            failed_tests.append(f"hidden_test_{index}_shape")
            continue
        test_id = str(test.get("test_id") or test.get("name") or f"hidden_test_{index}")
        if test_id in seen_test_ids:
            errors.append(f"duplicate_hidden_test_id: {test_id}")
            failed_tests.append(f"{test_id}_duplicate")
            continue
        seen_test_ids.add(test_id)
        code = str(test.get("code") or test.get("assertion_code") or "").strip()
        if not code:
            errors.append(f"hidden_test_missing_code: {test_id}")
            failed_tests.append(f"{test_id}_missing_code")
            continue
        try:
            compile(code, f"{env.env_id}_{test_id}.py", "exec")
        except SyntaxError as exc:
            errors.append(f"hidden_test_compile_error:{test_id}: {exc.msg}")
            failed_tests.append(f"{test_id}_compile")
            continue
        try:
            test_namespace = dict(namespace)
            with _suppress_process_output():
                exec(code, test_namespace, test_namespace)
        except Exception as exc:
            errors.append(f"hidden_test_execution_error:{test_id}: {exc.__class__.__name__}: {exc}")
            failed_tests.append(f"{test_id}_execute")
            continue
        passed_tests.append(test_id)

    return HiddenTestCheckResult(
        status="pass" if not errors else "fail",
        errors=errors,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
    )


def materialize_solution_context(env: ExecutableEnvSpec) -> str:
    placeholder = env.visible_state.get("placeholder_token", "<<insert solution here>>")
    executable_code = str(env.scaled_executable_gold_code or "").strip()
    if executable_code:
        return executable_code
    gold_solution = str(env.gold_solution or "").strip()
    if looks_like_complete_program(gold_solution, placeholder=placeholder):
        return gold_solution
    return insert_solution(
        context=env.context,
        solution=env.gold_solution,
        placeholder=placeholder,
    )


def insert_solution(context: str, solution: str, placeholder: str) -> str:
    if placeholder not in context:
        return context
    lines = context.splitlines()
    for line in lines:
        if placeholder not in line:
            continue
        indent = line[: line.index(placeholder)]
        if line.strip() == placeholder:
            indented_solution = _indent_block(solution, indent)
            return context.replace(placeholder, indented_solution, 1)
        return context.replace(placeholder, solution, 1)
    return context.replace(placeholder, solution, 1)


def looks_like_complete_program(code: str, placeholder: str = "<<insert solution here>>") -> bool:
    stripped = str(code or "").strip()
    if not stripped:
        return False
    if placeholder and placeholder in stripped:
        return False
    if 'if __name__ == "__main__"' in stripped or "if __name__ == '__main__'" in stripped:
        return True
    if stripped.startswith(("import ", "from ", "def ", "class ")) and "\n" in stripped:
        return True
    if "\n" in stripped and "return " not in stripped.splitlines()[0]:
        return True
    return False


def merge_candidate_into_seed_scaffold(
    scaffold: str,
    candidate: str,
    placeholder: str = "<<insert solution here>>",
    fallback_context: str = "",
) -> str:
    scaffold = str(scaffold or "").strip()
    candidate = str(candidate or "").strip()
    if not candidate:
        return fallback_context or scaffold or candidate
    if not scaffold or placeholder in candidate:
        return candidate
    merged = _replace_top_level_definition(scaffold, candidate)
    if merged:
        return merged
    if fallback_context and placeholder in fallback_context:
        return insert_solution(fallback_context, candidate, placeholder)
    return candidate


def _replace_top_level_definition(scaffold: str, candidate: str) -> str | None:
    candidate_name, candidate_kind = _extract_primary_definition_signature(candidate)
    if not candidate_name or not candidate_kind:
        return None
    scaffold_lines = scaffold.splitlines()
    prefix = f"{candidate_kind} {candidate_name}"
    start = None
    indent = None
    for index, line in enumerate(scaffold_lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if current_indent != 0:
            continue
        if stripped.startswith(prefix) and stripped[len(prefix) : len(prefix) + 1] in {"(", ":"}:
            start = index
            indent = current_indent
            break
    if start is None or indent is None:
        return None
    end = len(scaffold_lines)
    for index in range(start + 1, len(scaffold_lines)):
        line = scaffold_lines[index]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if not stripped:
            continue
        if current_indent == 0 and (stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("if __name__")):
            end = index
            break
    replacement = candidate.splitlines()
    merged_lines = scaffold_lines[:start] + replacement + scaffold_lines[end:]
    return "\n".join(merged_lines)


def _extract_primary_definition_signature(code: str) -> tuple[str | None, str | None]:
    for line in str(code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            name = stripped[4:].split("(", 1)[0].strip()
            return name or None, "def"
        if stripped.startswith("class "):
            name = stripped[6:].split("(", 1)[0].split(":", 1)[0].strip()
            return name or None, "class"
    return None, None


def _indent_block(block: str, indent: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    return "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines)


@contextmanager
def _suppress_process_output():
    sink = io.StringIO()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
