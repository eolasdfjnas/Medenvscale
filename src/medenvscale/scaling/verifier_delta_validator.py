from __future__ import annotations

from medenvscale.schemas.scaling import DynamicOperatorInstance
from medenvscale.scaling.verifier_delta_normalizer import normalize_verifier_delta


def validate_verifier_delta(op: DynamicOperatorInstance) -> list[str]:
    errors: list[str] = []
    try:
        normalized = normalize_verifier_delta(op.verifier_delta, owner_id=op.operator_id)
        op.verifier_delta = op.verifier_delta.model_validate(normalized)
    except Exception as exc:
        return [f"verifier_delta_normalization_failed: {exc}"]
    delta = op.verifier_delta
    if op.axis == "V" and not (delta.new_hidden_tests or delta.new_checks or op.semantic_test_specs):
        errors.append("V operator must add hidden tests or verifier checks")
    if op.axis in {"D", "C", "V", "A"} and not any(
        [delta.new_hidden_tests, delta.new_checks, delta.static_checks, delta.exception_tests, delta.numeric_tolerance_tests, op.semantic_test_specs]
    ):
        errors.append("Behavior-changing operator must include executable verifier deltas")
    seen_hidden_ids: set[str] = set()
    for test in delta.new_hidden_tests:
        if not isinstance(test, dict):
            errors.append("Hidden test must be a dict")
            continue
        test_id = str(test.get("test_id") or test.get("name") or "").strip()
        if not test_id:
            errors.append("Hidden test missing test_id")
        elif test_id in seen_hidden_ids:
            errors.append(f"duplicate_hidden_test_id: {test_id}")
        else:
            seen_hidden_ids.add(test_id)
        code = str(test.get("code") or "").strip()
        if not code:
            continue
        if "assert" not in code and "def test_" not in code:
            continue
        try:
            compile(code, f"{op.operator_id}_{test_id or 'hidden'}.py", "exec")
        except SyntaxError:
            continue
    for test in delta.numeric_tolerance_tests:
        if not isinstance(test, dict):
            errors.append("numeric_tolerance_test must be a dict")
            continue
        if "expected" not in test or "tolerance" not in test or test.get("expected") is None or test.get("tolerance") is None:
            errors.append("numeric_tolerance_tests missing required fields")
    for test in delta.exception_tests:
        if not isinstance(test, dict):
            errors.append("exception_test must be a dict")
            continue
        if not test.get("expected_exception"):
            errors.append("exception_tests missing required fields")
    for check in delta.static_checks:
        if not isinstance(check, dict):
            errors.append("static_check must be a dict")
            continue
        if not check.get("rule"):
            errors.append("static_checks missing required fields")
    return errors
