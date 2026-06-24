from __future__ import annotations

import json
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
        self.terminated = False

    def execute(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        args = _coerce_args(arguments)
        allowed = self._budget_allows(name)
        if not allowed["ok"]:
            result = allowed
        elif name == "get_task_context":
            result = self.get_task_context(window=int(args.get("window", 4000)))
        elif name == "read_resource_file":
            result = self.read_resource_file(
                path=str(args.get("path") or ""),
                offset=int(args.get("offset", 0) or 0),
                max_bytes=int(args.get("max_bytes", 4000) or 4000),
            )
        elif name == "validate_candidate_code":
            result = validate_candidate_code(str(args.get("code") or ""), self.env.signature)
        elif name == "run_custom_test":
            result = run_custom_test(
                code=str(args.get("code") or ""),
                test_snippet=str(args.get("test_snippet") or ""),
                timeout_seconds=int(args.get("timeout_seconds", 5) or 5),
            )
        elif name == "submit_final_code":
            result = self.submit_final_code(str(args.get("code") or ""))
        else:
            result = {"ok": False, "error": f"unknown_tool:{name}"}
        if name == "submit_final_code" and not result.get("terminated", False):
            self.call_counts[name] = max(0, self.call_counts.get(name, 0) - 1)
            if allowed.get("ok") and not self.submit_excluded_from_total:
                self.total_calls = max(0, self.total_calls - 1)
        self.trace.append({"tool_name": name, "arguments": _redact_args(args), "result": _redact_result(name, result)})
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
        }

    def read_resource_file(self, path: str, offset: int = 0, max_bytes: int = 4000) -> dict[str, Any]:
        safety = analyze_relative_path(path)
        if not safety.safe:
            return {"ok": False, "error": f"unsafe_path:{safety.reason}", "path": path}
        normalized = safety.normalized_path
        allowed = {item["path"] for item in self._public_resource_manifest()}
        if allowed and normalized not in allowed and path not in allowed:
            return {"ok": False, "error": "resource_not_in_manifest", "path": normalized, "available_paths": sorted(allowed)}
        resolved = self._resolve_resource(normalized)
        if resolved is None:
            return {"ok": False, "error": "resource_not_found", "path": normalized}
        start = max(0, int(offset or 0))
        limit = max(1, min(int(max_bytes or 4000), 20000))
        data = resolved.read_bytes()
        chunk = data[start : start + limit]
        return {
            "ok": True,
            "path": normalized,
            "offset": start,
            "bytes_read": len(chunk),
            "total_bytes": len(data),
            "content": chunk.decode("utf-8", errors="replace"),
            "truncated": start + len(chunk) < len(data),
        }

    def submit_final_code(self, code: str) -> dict[str, Any]:
        preflight = preflight_final_code(code, self.env.signature)
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
            if self.call_counts.get(name, 0) >= 1:
                return {"ok": False, "error": "tool_budget_exhausted:submit_final_code"}
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            return {"ok": True}
        max_total = int(self.budget["max_total_tool_calls"])
        if self.total_calls >= max_total:
            return {"ok": False, "error": "tool_budget_exhausted"}
        per_tool = self.budget["max_calls_per_tool"]
        if self.call_counts.get(name, 0) >= int(per_tool.get(name, max_total)):
            return {"ok": False, "error": f"tool_budget_exhausted:{name}"}
        self.total_calls += 1
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        return {"ok": True}

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

    def _resolve_resource(self, normalized_path: str) -> Path | None:
        candidates = [
            self.cfg.root / normalized_path,
            self.cfg.output_dirs["raw"] / normalized_path,
            self.cfg.output_dirs["raw"] / "source" / normalized_path,
        ]
        for candidate in candidates:
            try:
                candidate.relative_to(self.cfg.root)
            except ValueError:
                continue
            if candidate.exists() and candidate.is_file():
                return candidate
        return None


def _normalize_budget(raw: dict[str, Any]) -> dict[str, Any]:
    per_tool = {
        "get_task_context": 1,
        "read_resource_file": 3,
        "validate_candidate_code": 2,
        "run_custom_test": 3,
        "submit_final_code": 1,
    }
    supplied = raw.get("max_calls_per_tool") or {}
    per_tool.update({str(key): int(value) for key, value in supplied.items()})
    return {
        "max_total_tool_calls": int(raw.get("max_total_tool_calls", 6)),
        "max_calls_per_tool": per_tool,
    }


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
    for key in ("code", "test_snippet"):
        if key in redacted:
            redacted[key] = {"sha256_prefix": __import__("hashlib").sha256(str(redacted[key]).encode("utf-8")).hexdigest()[:12], "chars": len(str(redacted[key]))}
    return redacted


def _redact_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if name != "submit_final_code":
        return result
    return {key: value for key, value in result.items() if key not in {"case_reports", "materialized_code"}}
