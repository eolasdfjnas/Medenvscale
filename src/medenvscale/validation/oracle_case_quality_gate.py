from __future__ import annotations

import re
from typing import Any

from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.scaling.requirement_registry import requirements_match

from .report_schema import build_gate_result

HARD_FAIL_CODES = {
    "NO_VALIDATED_ORACLE_CASES",
    "ORACLE_CASE_VALIDATION_FAILED",
    "NO_CASE_COVERS_NEW_REQUIREMENT",
    "PARTIAL_CASE_REQUIREMENT_COVERAGE",
    "PLACEHOLDER_CASE_PRESENT",
    "CASE_MISSING_CALL_CODE",
    "CASE_MISSING_EXPECTED_OUTPUT_SIGNATURE",
}

GENERIC_REQUIREMENT_PATTERNS = [
    "scaled executable cases must expose the new requirement introduced by this operator",
    "scaled executable cases must expose the new computation and contract constraints introduced by this operator",
    "handle scaled input/data variants from the oracle cases",
    "scaled cases introduce stronger computation",
    "respect new parameter boundary ordering or return contract constraints",
    "new requirement introduced by this operator",
]

GENERIC_REQUIREMENT_TOKENS = {
    "a",
    "an",
    "additional",
    "and",
    "any",
    "are",
    "as",
    "answer",
    "be",
    "boundary",
    "case",
    "cases",
    "computation",
    "constraint",
    "constraints",
    "contract",
    "data",
    "edge",
    "expected",
    "executable",
    "expose",
    "format",
    "from",
    "function",
    "global",
    "handle",
    "handles",
    "input",
    "introduced",
    "must",
    "new",
    "operator",
    "or",
    "oracle",
    "output",
    "parameter",
    "parameters",
    "provided",
    "requirement",
    "requirements",
    "respect",
    "return",
    "returns",
    "scaled",
    "should",
    "solution",
    "still",
    "stronger",
    "task",
    "test",
    "tests",
    "the",
    "to",
    "using",
    "value",
    "values",
    "variant",
    "variants",
}


def run_oracle_case_quality_gate(sample: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    scaled_env = _load_env(sample, "scaled_task")
    config = config or {}
    validated_cases = list(scaled_env.validated_oracle_cases or [])
    validation_report = list(scaled_env.oracle_case_validation_report or [])
    output_requirements = list(scaled_env.output_requirements or [])
    output_requirement_metadata = [
        row for row in (scaled_env.output_requirement_metadata or []) if isinstance(row, dict)
    ]
    difficulty_level = str((scaled_env.difficulty.global_level if scaled_env.difficulty else "") or "M1")
    stage05_cfg = config.get("stage05_cfg") or {}
    recommended_cases_cfg = stage05_cfg.get("min_validated_oracle_cases", {})
    recommended_case_count = int(recommended_cases_cfg.get(difficulty_level, 0))

    checks = {
        "validated_case_count_passed": len(validated_cases) >= 1,
        "validation_report_passed": all(bool(row.get("valid")) for row in validation_report),
        "requirement_coverage_passed": True,
        "placeholder_filter_passed": True,
        "input_output_shape_passed": True,
    }
    failure_reasons: list[str] = []
    warnings: list[str] = []

    if not validated_cases:
        failure_reasons.append("NO_VALIDATED_ORACLE_CASES")
    if recommended_case_count and len(validated_cases) < recommended_case_count:
        warnings.append(f"RECOMMENDED_VALIDATED_CASES_TOO_FEW:{len(validated_cases)}/{recommended_case_count}")
    invalid_rows = [row for row in validation_report if not bool(row.get("valid"))]
    if invalid_rows:
        failure_reasons.append("ORACLE_CASE_VALIDATION_FAILED")
    placeholder_case_ids = []
    covered_requirements = set()
    covered_requirement_ids = set()
    bound_requirement_evidence: dict[str, list[str]] = {}
    coverage_evidence = set()
    for case in validated_cases:
        if not isinstance(case, dict):
            continue
        if "assert True" in str(case.get("assertion_code") or ""):
            placeholder_case_ids.append(str(case.get("case_id") or "unknown_case"))
        if not str(case.get("call_code") or "").strip():
            checks["input_output_shape_passed"] = False
            failure_reasons.append("CASE_MISSING_CALL_CODE")
        if not isinstance(case.get("expected_output_signature"), dict) or not case.get("expected_output_signature"):
            checks["input_output_shape_passed"] = False
            failure_reasons.append("CASE_MISSING_EXPECTED_OUTPUT_SIGNATURE")
        for requirement in ((case.get("covered_requirements") or case.get("covers_requirements")) or []):
            text = str(requirement).strip()
            if text:
                covered_requirements.add(text)
                coverage_evidence.add(text)
        for req_id in case.get("covered_requirement_ids") or []:
            text = str(req_id).strip()
            if text and not output_requirement_metadata:
                covered_requirement_ids.add(text)
        for check in case.get("bound_requirement_checks") or []:
            if not isinstance(check, dict):
                continue
            req_id = str(check.get("requirement_id") or "")
            if req_id and check.get("passed"):
                covered_requirement_ids.add(req_id)
                bound_requirement_evidence.setdefault(req_id, []).append(str(case.get("case_id") or ""))
        for evidence in _case_requirement_evidence(case):
            coverage_evidence.add(evidence)
    if placeholder_case_ids:
        checks["placeholder_filter_passed"] = False
        failure_reasons.extend(f"PLACEHOLDER_CASE_PRESENT:{case_id}" for case_id in placeholder_case_ids)
    coverage_targets: list[str] = []
    coverage_target_ids: list[str] = []
    missing_requirement_ids: list[str] = []
    skipped_generic_requirements: list[str] = []
    if output_requirement_metadata:
        for row in output_requirement_metadata:
            req_id = str(row.get("requirement_id") or "")
            req = str(row.get("text") or "").strip()
            if not req_id or not req or not bool(row.get("required_coverage", True)):
                continue
            if _is_generic_requirement(req):
                skipped_generic_requirements.append(req)
                continue
            coverage_targets.append(req)
            coverage_target_ids.append(req_id)
        missing_requirement_ids = [req_id for req_id in coverage_target_ids if req_id not in covered_requirement_ids]
        if coverage_target_ids:
            matched = len(coverage_target_ids) - len(missing_requirement_ids)
            if matched == 0:
                checks["requirement_coverage_passed"] = False
                failure_reasons.append("NO_CASE_COVERS_NEW_REQUIREMENT")
            elif missing_requirement_ids:
                checks["requirement_coverage_passed"] = False
                failure_reasons.append("PARTIAL_CASE_REQUIREMENT_COVERAGE")
    elif output_requirements:
        for requirement in output_requirements:
            req = str(requirement).strip()
            if not req:
                continue
            if _is_generic_requirement(req):
                skipped_generic_requirements.append(req)
            else:
                coverage_targets.append(req)

    if coverage_targets:
        matched = 0
        for req in coverage_targets:
            if any(_requirements_match(req, covered) for covered in coverage_evidence):
                matched += 1
        if matched == 0:
            checks["requirement_coverage_passed"] = False
            failure_reasons.append("NO_CASE_COVERS_NEW_REQUIREMENT")
        elif matched < len(coverage_targets):
            checks["requirement_coverage_passed"] = False
            failure_reasons.append("PARTIAL_CASE_REQUIREMENT_COVERAGE")
    elif skipped_generic_requirements:
        warnings.append("GENERIC_OUTPUT_REQUIREMENTS_SKIPPED")

    return build_gate_result(
        gate_name="oracle_case_quality_gate",
        checks=checks,
        failure_reasons=failure_reasons,
        warnings=warnings,
        evidence={
            "validated_case_ids": [str(case.get("case_id") or "") for case in validated_cases if isinstance(case, dict)],
            "invalid_case_ids": [str(row.get("case_id") or "") for row in invalid_rows],
            "covered_requirements": sorted(covered_requirements),
            "covered_requirement_ids": sorted(covered_requirement_ids),
            "coverage_evidence": sorted(coverage_evidence),
            "coverage_targets": coverage_targets,
            "coverage_target_ids": coverage_target_ids,
            "missing_requirement_ids": missing_requirement_ids,
            "missing_requirements": [
                str(row.get("text") or "")
                for row in output_requirement_metadata
                if str(row.get("requirement_id") or "") in set(missing_requirement_ids)
            ],
            "bound_requirement_evidence": bound_requirement_evidence,
            "skipped_generic_output_requirements": skipped_generic_requirements,
            "recommended_case_count": recommended_case_count,
        },
        hard_fail_codes=HARD_FAIL_CODES,
    )


def _load_env(sample: dict[str, Any], key: str) -> ExecutableEnvSpec:
    value = sample.get(key)
    if isinstance(value, ExecutableEnvSpec):
        return value
    if isinstance(value, dict):
        return ExecutableEnvSpec.model_validate(value)
    if key == "scaled_task":
        return ExecutableEnvSpec.model_validate(sample)
    raise ValueError(f"Missing environment payload: {key}")


def _is_generic_requirement(text: str) -> bool:
    normalized = _normalize_requirement(text)
    if not normalized:
        return True
    if any(pattern in normalized for pattern in GENERIC_REQUIREMENT_PATTERNS):
        return True
    tokens = _requirement_tokens(normalized)
    if not tokens:
        return True
    specific_tokens = [token for token in tokens if token not in GENERIC_REQUIREMENT_TOKENS]
    return len(specific_tokens) == 0


def _requirements_match(target: str, covered: str) -> bool:
    return requirements_match(target, covered)


def _case_requirement_evidence(case: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for key in ["description", "semantic_intent", "target_constraint", "expected_failure_mode"]:
        text = str(case.get(key) or "").strip()
        if text:
            evidence.append(text)
    expected = case.get("expected_output_signature")
    if isinstance(expected, dict):
        for value in expected.values():
            if isinstance(value, str) and value.strip():
                evidence.append(value.strip())
            elif isinstance(value, list):
                evidence.extend(str(item).strip() for item in value if str(item).strip())
    return evidence


def _specific_tokens(text: str) -> set[str]:
    return {token for token in _requirement_tokens(text) if token not in GENERIC_REQUIREMENT_TOKENS}


def _requirement_tokens(text: str) -> list[str]:
    return [_canonical_requirement_token(token) for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())]


def _canonical_requirement_token(token: str) -> str:
    synonyms = {
        "mutate": "modify",
        "mutated": "modify",
        "mutating": "modify",
        "mutation": "modify",
        "modified": "modify",
        "modifies": "modify",
        "check": "verify",
        "checked": "verify",
        "checks": "verify",
        "prove": "verify",
        "proves": "verify",
        "proving": "verify",
        "test": "verify",
        "tests": "verify",
        "testing": "verify",
        "validated": "validate",
        "validates": "validate",
        "validating": "validate",
    }
    return synonyms.get(token, token)


def _normalize_requirement(text: str) -> str:
    return " ".join(_requirement_tokens(str(text)))
