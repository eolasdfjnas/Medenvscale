from __future__ import annotations

from medenvscale.schemas import ExecutableEnvSpec
from medenvscale.schemas.scaling import DynamicOperatorInstance
from medenvscale.schemas.verifier import VerifierSpec
from medenvscale.scaling.verifier_delta_normalizer import normalize_verifier_delta
from medenvscale.scaling.verifier_delta_normalizer import is_weak_smoke_test


def build_verifier_spec(
    environment: ExecutableEnvSpec,
    operator_instances: list[DynamicOperatorInstance],
    hidden_tests: list[dict] | None = None,
) -> VerifierSpec:
    checks = [{"name": "compile_check", "kind": "compile"}]
    collected_hidden_tests: list[dict] = [] if hidden_tests is None else list(hidden_tests)
    static_checks = [{"name": "placeholder_removed", "rule": "placeholder must not remain after insertion"}]
    exception_tests = []
    generated_from = []
    secondary_domain_names = [item.domain for item in environment.secondary_domains]
    if secondary_domain_names:
        static_checks.append(
            {
                "name": "secondary_domain_context_available",
                "rule": f"secondary domain semantic context preserved: {', '.join(secondary_domain_names)}",
            }
        )
        checks.append(
            {
                "name": "secondary_domain_grounding",
                "kind": "semantic_context",
                "domains": secondary_domain_names,
            }
        )
    for op in operator_instances:
        try:
            normalized_delta = normalize_verifier_delta(op.verifier_delta, owner_id=op.operator_id)
            op.verifier_delta = op.verifier_delta.model_validate(normalized_delta)
        except Exception:
            continue
        generated_from.append(op.operator_id)
        checks.extend(_sanitize_dict_items(op.verifier_delta.new_checks))
        if hidden_tests is None:
            collected_hidden_tests.extend(_sanitize_hidden_tests(op.verifier_delta.new_hidden_tests))
        static_checks.extend(_sanitize_dict_items(op.verifier_delta.static_checks))
        exception_tests.extend(_sanitize_dict_items(op.verifier_delta.exception_tests))
    final_hidden_tests = _sanitize_hidden_tests(collected_hidden_tests)
    return VerifierSpec(
        verifier_id=f"verifier_{environment.env_id}",
        env_id=environment.env_id,
        verifier_type=environment.verifier_type_hint or "unit_test",
        solution_form=environment.solution_form,
        checks=checks,
        hidden_tests=final_hidden_tests,
        exception_tests=exception_tests,
        static_checks=static_checks,
        rubric_links=[environment.primary_domain, *secondary_domain_names],
        generated_from_operator_ids=generated_from,
    )


def _sanitize_hidden_tests(items: list[dict] | list) -> list[dict]:
    sanitized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("assertion_code") or "").strip()
        if not code:
            continue
        item = dict(item)
        if _is_disallowed_hidden_test(item, code):
            continue
        item["code"] = code
        item.setdefault("name", item.get("test_id"))
        item.setdefault("assertion_code", code)
        sanitized.append(item)
    return sanitized


def _sanitize_dict_items(items: list[dict] | list) -> list[dict]:
    return [item for item in items if isinstance(item, dict)]


def _is_disallowed_hidden_test(item: dict, code: str) -> bool:
    normalized = code.replace(" ", "")
    if bool(item.get("is_placeholder")):
        return True
    if str(item.get("source") or "") == "fallback":
        return True
    if str(item.get("test_tier") or "") == "smoke":
        return True
    if not bool(item.get("counts_as_hidden_test", True)):
        return True
    if not bool(item.get("eligible_for_clean_export", True)):
        return True
    if is_weak_smoke_test(code):
        return True
    if "asserttrue" in normalized.lower():
        return True
    if "callable(globals()" in normalized.lower():
        return True
    return False
