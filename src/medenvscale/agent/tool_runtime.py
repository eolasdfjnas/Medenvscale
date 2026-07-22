from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from medenvscale.agent.candidate_execution import preflight_final_code, run_custom_test, validate_candidate_code
from medenvscale.config import AppConfig
from medenvscale.scaling.case_execution import run_scaled_gold_on_validated_oracle_cases
from medenvscale.scaling.path_safety import analyze_relative_path
from medenvscale.schemas import ExecutableEnvSpec


class ToolRuntime:
    def __init__(
        self,
        env: ExecutableEnvSpec,
        cfg: AppConfig,
        budget: dict[str, Any] | None = None,
        allowed_tools: set[str] | None = None,
        submit_excluded_from_total: bool = True,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.budget = _normalize_budget(budget or {})
        self.allowed_tools = allowed_tools or set(self.budget["max_calls_per_tool"])
        self.submit_excluded_from_total = submit_excluded_from_total
        self.trace: list[dict[str, Any]] = []
        self.call_counts: dict[str, int] = {}
        self.total_calls = 0
        self.final_code: str | None = None
        self.final_eval: dict[str, Any] | None = None
        self.test_files: dict[str, str] = {}
        self.python_bin = _agent_python_bin(cfg)
        self.terminated = False

    def execute(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        args = _coerce_args(arguments)
        allowed = self._budget_allows(name)
        if not allowed["ok"]:
            result = allowed
        elif name == "get_task_context":
            result = self.get_task_context(window=int(args.get("window", 4000)))
        elif name == "validate_candidate_code":
            result = validate_candidate_code(str(args.get("code") or ""), self.env.signature)
        elif name == "create_test_file":
            result = self.create_test_file(
                path=str(args.get("path") or ""),
                content=str(args.get("content") or ""),
            )
        elif name == "run_custom_test":
            result = run_custom_test(
                code=str(args.get("code") or ""),
                test_snippet=str(args.get("test_snippet") or ""),
                timeout_seconds=int(args.get("timeout_seconds", 5) or 5),
                fixture_files=self._test_fixture_files(),
                python_bin=self.python_bin,
            )
        elif name == "submit_final_code":
            result = self.submit_final_code(str(args.get("code") or ""))
        else:
            result = {"ok": False, "error": f"unknown_tool:{name}"}
        if allowed.get("budget_violation") and isinstance(result, dict):
            result = dict(result)
            result["budget_violation"] = True
            result["budget_error"] = allowed.get("budget_error") or "tool_budget_exceeded"
        if name == "submit_final_code" and allowed.get("ok") and not result.get("terminated", False):
            self.call_counts[name] = max(0, self.call_counts.get(name, 0) - 1)
            if allowed.get("counted_total") and not self.submit_excluded_from_total:
                self.total_calls = max(0, self.total_calls - 1)
        trace_row = {
            "tool_name": name,
            "arguments": _redact_args(args),
            "result": _redact_result(name, result),
        }
        if allowed.get("budget_violation"):
            trace_row["budget_violation"] = True
            trace_row["budget_error"] = allowed.get("budget_error") or "tool_budget_exceeded"
        self.trace.append(trace_row)
        return result

    def get_task_context(self, window: int = 4000) -> dict[str, Any]:
        limit = max(500, min(int(window or 4000), 12000))
        return {
            "ok": True,
            "env_id": self.env.env_id,
            "problem": _truncate(self.env.problem, limit),
            "user_prompt": _truncate(self.env.user_prompt or self.env.problem, limit),
            "signature": self.env.signature or "",
            "context": _truncate(self.env.context, limit),
            "solution_form": self.env.solution_form,
            "resource_manifest": self._public_resource_manifest(),
            "public_requirements": list(self.env.output_requirements or []),
            "difficulty": (self.env.difficulty.model_dump() if self.env.difficulty else {}),
            "agent_created_test_files": sorted(self.test_files),
        }

    def create_test_file(self, path: str, content: str) -> dict[str, Any]:
        safety = analyze_relative_path(path, artifact=True)
        if not safety.safe:
            return {"ok": False, "error": f"unsafe_path:{safety.reason}", "path": path}
        normalized = safety.normalized_path
        data = str(content or "").encode("utf-8")
        if len(data) > 65536:
            return {"ok": False, "error": "file_too_large", "path": normalized, "max_bytes": 65536}
        current_total = sum(len(value.encode("utf-8")) for value in self.test_files.values())
        if current_total + len(data) > 262144:
            return {"ok": False, "error": "test_files_total_too_large", "max_total_bytes": 262144}
        self.test_files[normalized] = str(content or "")
        return {
            "ok": True,
            "path": normalized,
            "bytes_written": len(data),
            "total_test_files": len(self.test_files),
            "available_to": ["run_custom_test"],
            "note": "This file is available only to later run_custom_test calls, not to final oracle evaluation.",
        }

    def submit_final_code(self, code: str) -> dict[str, Any]:
        preflight = preflight_final_code(code, self.env.signature, python_bin=self.python_bin)
        if not preflight["ok"]:
            return {
                "ok": False,
                "terminated": False,
                "preflight_passed": False,
                "errors": preflight["errors"],
                "repair_hints": preflight["repair_hints"],
                "validation": preflight["validation"],
                "import_check": preflight["import_check"],
            }
        self.final_code = code
        evaluation_cases, case_source = _evaluation_cases_for_env(self.env)
        self.final_eval = run_scaled_gold_on_validated_oracle_cases(
            self.env,
            code,
            evaluation_cases,
            python_bin=self.python_bin,
        )
        self.final_eval["evaluation_case_source"] = case_source
        if not evaluation_cases:
            self.final_eval["execution_passed"] = False
            failure_reasons = list(self.final_eval.get("failure_reasons", []) or [])
            failure_reasons.append("NO_EVALUATION_CASES")
            self.final_eval["failure_reasons"] = failure_reasons
        self.terminated = True
        reports = self.final_eval.get("case_reports", [])
        passed_count = sum(1 for report in reports if report.get("passed"))
        return {
            "ok": bool(self.final_eval.get("compile_passed") and self.final_eval.get("execution_passed")),
            "terminated": True,
            "preflight_passed": True,
            "compile_passed": bool(self.final_eval.get("compile_passed")),
            "execution_passed": bool(self.final_eval.get("execution_passed")),
            "passed_cases": passed_count,
            "total_cases": len(reports),
            "evaluation_case_source": case_source,
            "failure_count": len(self.final_eval.get("failure_reasons", [])),
        }

    def mark_no_final_code(self, reason: str = "NO_FINAL_CODE_SUBMITTED") -> dict[str, Any]:
        self.final_code = ""
        self.final_eval = {
            "compile_passed": False,
            "execution_passed": False,
            "evaluation_case_source": "none",
            "failure_reasons": [reason],
            "case_reports": [],
        }
        self.terminated = True
        return {
            "ok": False,
            "terminated": True,
            "compile_passed": False,
            "execution_passed": False,
            "passed_cases": 0,
            "total_cases": 0,
            "evaluation_case_source": "none",
            "failure_count": 1,
        }

    def _budget_allows(self, name: str) -> dict[str, Any]:
        if name not in self.allowed_tools:
            return {"ok": False, "error": f"tool_not_allowed:{name}"}
        if self.terminated:
            return {"ok": False, "error": "episode_already_terminated"}
        if name == "submit_final_code" and self.submit_excluded_from_total:
            violation = self.call_counts.get(name, 0) >= int(self.budget["max_calls_per_tool"].get(name, 1))
            if self.call_counts.get(name, 0) >= 1:
                violation = True
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            return _budget_status(violation, "tool_budget_exceeded:submit_final_code", counted_total=False)
        max_total = int(self.budget["max_total_tool_calls"])
        per_tool = self.budget["max_calls_per_tool"]
        violation = self.total_calls >= max_total
        error = "tool_budget_exceeded" if violation else ""
        if self.call_counts.get(name, 0) >= int(per_tool.get(name, max_total)):
            violation = True
            error = f"tool_budget_exceeded:{name}"
        self.total_calls += 1
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        return _budget_status(violation, error, counted_total=True)

    def _public_resource_manifest(self) -> list[dict[str, Any]]:
        rows = []
        for item in self.env.resource_manifest or []:
            if isinstance(item, dict) and item.get("path"):
                normalized = analyze_relative_path(item["path"])
                if normalized.safe:
                    rows.append({"path": normalized.normalized_path})
        for path in self.env.resource_files or []:
            normalized = analyze_relative_path(path)
            if normalized.safe:
                rows.append({"path": normalized.normalized_path})
        deduped = {}
        for row in rows:
            if row["path"]:
                deduped[row["path"]] = row
        return list(deduped.values())

    def _test_fixture_files(self) -> list[dict[str, Any]]:
        return [{"path": path, "content": content} for path, content in sorted(self.test_files.items())]


def _normalize_budget(raw: dict[str, Any]) -> dict[str, Any]:
    per_tool = {
        "get_task_context": 1,
        "create_test_file": 3,
        "validate_candidate_code": 2,
        "run_custom_test": 3,
        "submit_final_code": 1,
    }
    supplied = raw.get("max_calls_per_tool") or {}
    per_tool.update({str(key): int(value) for key, value in supplied.items()})
    return {
        "max_total_tool_calls": int(raw.get("max_total_tool_calls", 7)),
        "max_calls_per_tool": per_tool,
    }


def _budget_status(violation: bool, error: str, *, counted_total: bool) -> dict[str, Any]:
    status = {"ok": True, "counted_total": bool(counted_total)}
    if violation:
        status["budget_violation"] = True
        status["budget_error"] = error or "tool_budget_exceeded"
    return status


def _agent_python_bin(cfg: AppConfig) -> str:
    code_execution = ((cfg.values.get("dataset") or {}).get("code_execution") or {}) if cfg is not None else {}
    configured = str(code_execution.get("local_python_bin") or code_execution.get("python_bin") or "").strip()
    return configured or sys.executable


def _evaluation_cases_for_env(env: ExecutableEnvSpec) -> tuple[list[dict[str, Any]], str]:
    validated = [case for case in (env.validated_oracle_cases or []) if isinstance(case, dict)]
    if validated:
        return validated, "validated_oracle_cases"
    seed_case = env.seed_execution_case if isinstance(env.seed_execution_case, dict) else {}
    if seed_case:
        return [seed_case], "seed_execution_case"
    return [], "none"


def _coerce_args(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            payload = json.loads(arguments)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _truncate(text: str | None, limit: int) -> str:
    value = str(text or "")
    return value[:limit]


def _redact_args(args: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(args)
    for key in ("code", "test_snippet", "content"):
        if key in redacted:
            redacted[key] = {"sha256_prefix": __import__("hashlib").sha256(str(redacted[key]).encode("utf-8")).hexdigest()[:12], "chars": len(str(redacted[key]))}
    return redacted


def _redact_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if name != "submit_final_code":
        return result
    return {key: value for key, value in result.items() if key not in {"case_reports", "materialized_code"}}
