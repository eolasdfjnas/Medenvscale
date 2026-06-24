from __future__ import annotations

from typing import Any


HARD_FAIL = "hard_fail"
SOFT_FAIL = "soft_fail"
PASS_WITH_WARNING = "pass_with_warning"
PASS = "pass"


def build_gate_result(
    gate_name: str,
    checks: dict[str, bool],
    failure_reasons: list[str],
    warnings: list[str],
    evidence: dict[str, Any] | None = None,
    hard_fail_codes: set[str] | None = None,
) -> dict[str, Any]:
    failure_reasons = _dedupe(failure_reasons)
    warnings = _dedupe(warnings)
    severity = classify_severity(failure_reasons, warnings, hard_fail_codes=hard_fail_codes or set())
    passed = severity in {PASS, PASS_WITH_WARNING}
    return {
        "gate_name": gate_name,
        "passed": passed,
        "severity": severity,
        "score": _score_for_severity(severity),
        "checks": checks,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "evidence": evidence or {},
    }


def classify_severity(failure_reasons: list[str], warnings: list[str], hard_fail_codes: set[str]) -> str:
    if any(_matches_code(reason, hard_fail_codes) for reason in failure_reasons):
        return HARD_FAIL
    if failure_reasons:
        return SOFT_FAIL
    if warnings:
        return PASS_WITH_WARNING
    return PASS


def summarize_operator_report(operator_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not operator_reports:
        return {
            "passed": False,
            "severity": HARD_FAIL,
            "failure_reasons": ["OPERATOR_REALIZATION_REPORT_MISSING"],
            "warnings": [],
            "operator_ids": [],
        }
    failures: list[str] = []
    warnings: list[str] = []
    severities = [str(report.get("severity") or PASS) for report in operator_reports]
    for report in operator_reports:
        failures.extend(str(item) for item in report.get("failure_reasons", []) or [])
        warnings.extend(str(item) for item in report.get("warnings", []) or [])
    if HARD_FAIL in severities:
        severity = HARD_FAIL
    elif SOFT_FAIL in severities:
        severity = SOFT_FAIL
    elif PASS_WITH_WARNING in severities or warnings:
        severity = PASS_WITH_WARNING
    else:
        severity = PASS
    return {
        "passed": severity in {PASS, PASS_WITH_WARNING},
        "severity": severity,
        "failure_reasons": _dedupe(failures),
        "warnings": _dedupe(warnings),
        "operator_ids": [str(report.get("operator_id") or "") for report in operator_reports],
    }


def decide_final_decision(operator_summary: dict[str, Any], gate_results: list[dict[str, Any]]) -> tuple[str, bool]:
    if str(operator_summary.get("severity") or PASS) in {SOFT_FAIL, HARD_FAIL}:
        return "rejected", False
    if any(str(result.get("severity") or PASS) in {SOFT_FAIL, HARD_FAIL} for result in gate_results):
        return "rejected", False
    return "clean", True


def collect_rejection_reasons(operator_summary: dict[str, Any], gate_results: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if str(operator_summary.get("severity") or PASS) in {SOFT_FAIL, HARD_FAIL}:
        reasons.extend(str(item) for item in operator_summary.get("failure_reasons", []) or [])
        reasons.append(f"OPERATOR_REALIZATION_{str(operator_summary.get('severity') or '').upper()}")
    for result in gate_results:
        if str(result.get("severity") or PASS) in {SOFT_FAIL, HARD_FAIL}:
            reasons.extend(str(item) for item in result.get("failure_reasons", []) or [])
    return _dedupe(reasons)


def _matches_code(reason: str, codes: set[str]) -> bool:
    return any(reason == code or reason.startswith(f"{code}:") for code in codes)


def _score_for_severity(severity: str) -> float:
    return {
        PASS: 1.0,
        PASS_WITH_WARNING: 0.8,
        SOFT_FAIL: 0.3,
        HARD_FAIL: 0.0,
    }.get(severity, 0.0)


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
