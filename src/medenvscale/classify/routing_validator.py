from __future__ import annotations

from medenvscale.classify.taxonomy import (
    is_known_domain_name,
    is_known_solution_form_name,
    is_known_task_type_name,
    normalize_domain_name,
    normalize_solution_form_name,
    normalize_task_type_name,
)
from medenvscale.schemas import DomainHint, MedAgentGymTask, RoutingResult


def validate_secondary_domains(primary_domain: str, secondary_domains: list[DomainHint]) -> list[str]:
    errors = []

    if len(secondary_domains) > 3:
        errors.append("secondary_domains must contain at most 3 domains")

    seen = set()
    for item in secondary_domains:
        if item.domain == primary_domain:
            errors.append("secondary_domains must not repeat primary_domain")
        if item.domain in seen:
            errors.append(f"duplicate secondary domain: {item.domain}")
        seen.add(item.domain)
        if not (0.0 <= item.relevance <= 1.0):
            errors.append(f"secondary domain relevance out of range: {item.domain}")
        if item.relevance < 0.35:
            errors.append(f"secondary domain relevance too low: {item.domain}")

    return errors


def validate_routing(
    item: MedAgentGymTask,
    routing: dict,
    allowed_domains: list[str],
    allowed_task_types: list[str],
    min_confidence: float,
    review_confidence: float,
    allowed_solution_forms: list[str] | None = None,
) -> RoutingResult:
    routing = dict(routing)
    allowed_solution_forms = allowed_solution_forms or [
        "function_definition",
        "function_body",
        "expression_completion",
        "statement_block_completion",
        "decorated_function_definition",
        "patch_or_bugfix",
    ]

    raw_domain = str(routing.get("primary_domain") or routing.get("domain") or "").strip()
    domain = normalize_domain_name(raw_domain)
    needs_review = False
    if not raw_domain or domain not in allowed_domains or not is_known_domain_name(raw_domain):
        domain = allowed_domains[0]
        needs_review = True

    raw_secondary_domains = routing.get("secondary_domains", []) or []
    if isinstance(raw_secondary_domains, dict):
        raw_secondary_domains = [raw_secondary_domains]
    cleaned_secondary_domains: list[DomainHint] = []
    for secondary_item in raw_secondary_domains:
        if isinstance(secondary_item, str):
            candidate = DomainHint(domain=secondary_item, relevance=0.5)
        else:
            candidate = secondary_item if isinstance(secondary_item, DomainHint) else DomainHint.model_validate(secondary_item)
        if candidate.domain == domain:
            continue
        if candidate.domain not in allowed_domains:
            continue
        if candidate.relevance < 0.35:
            continue
        if any(existing.domain == candidate.domain for existing in cleaned_secondary_domains):
            continue
        cleaned_secondary_domains.append(candidate)
    cleaned_secondary_domains = sorted(cleaned_secondary_domains, key=lambda item: item.relevance, reverse=True)[:3]
    secondary_domain_errors = validate_secondary_domains(domain, cleaned_secondary_domains)
    if secondary_domain_errors:
        needs_review = True

    raw_task_type = str(routing.get("primary_task_type") or routing.get("task_type") or "").strip()
    task_type = normalize_task_type_name(raw_task_type)
    if not raw_task_type or task_type not in allowed_task_types or not is_known_task_type_name(raw_task_type):
        task_type = allowed_task_types[-1]
        needs_review = True

    raw_solution_form = str(routing.get("solution_form") or "").strip()
    solution_form = normalize_solution_form_name(raw_solution_form)
    if not raw_solution_form or solution_form not in allowed_solution_forms or not is_known_solution_form_name(raw_solution_form):
        solution_form = "statement_block_completion"
        needs_review = True

    secondary_task_types = routing.get("secondary_task_types", [])
    if isinstance(secondary_task_types, str):
        secondary_task_types = [secondary_task_types]
    cleaned_secondary = []
    for task in secondary_task_types:
        normalized = normalize_task_type_name(str(task))
        if normalized in allowed_task_types and normalized != task_type and normalized not in cleaned_secondary:
            cleaned_secondary.append(normalized)

    confidence = routing.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
        needs_review = True
    confidence = max(0.0, min(1.0, confidence))
    if confidence < review_confidence:
        needs_review = True
    if confidence < min_confidence:
        needs_review = True

    concepts = routing.get("domain_concepts", [])
    if isinstance(concepts, str):
        concepts = [concepts]
    capabilities = routing.get("required_capabilities", [])
    if isinstance(capabilities, str):
        capabilities = [capabilities]

    verifier_type_hint = routing.get("verifier_type_hint")
    if verifier_type_hint is not None:
        verifier_type_hint = str(verifier_type_hint)

    return RoutingResult(
        task_id=item.task_id,
        source_split=item.source_split,
        primary_domain=domain,
        secondary_domains=cleaned_secondary_domains,
        primary_task_type=task_type,
        solution_form=solution_form,
        secondary_task_types=cleaned_secondary,
        domain_concepts=[str(concept).strip() for concept in concepts if str(concept).strip()][:12],
        required_capabilities=[str(cap).strip() for cap in capabilities if str(cap).strip()][:12],
        verifier_type_hint=verifier_type_hint,
        routing_reason=str(routing.get("routing_reason") or f"Validated routing for {item.task_id}"),
        confidence=confidence,
        needs_review=needs_review,
        routing_trace={**(routing.get("routing_trace") or {}), "secondary_domain_errors": secondary_domain_errors},
    )
