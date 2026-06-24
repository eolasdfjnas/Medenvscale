from __future__ import annotations

from copy import deepcopy
from typing import Any

from medenvscale.classify.taxonomy import normalize_domain_name
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import ClinicalEnvironment


GUIDANCE_KEYS = (
    "risk_hints",
    "evidence_hints",
    "constraint_hints",
    "safety_hints",
    "unsafe_patterns",
    "distractor_history",
)

SECONDARY_DOMAIN_KEYS = (
    "risk_hints",
    "safety_hints",
    "unsafe_patterns",
)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def resolve_domain_guidance(primary_domain: str, secondary_domains: list[str], guidance_cfg: dict[str, Any]) -> dict[str, Any]:
    primary_domain = normalize_domain_name(primary_domain)
    secondary_domains = [normalize_domain_name(domain) for domain in secondary_domains]
    default = guidance_cfg.get("default", {})
    domains = guidance_cfg.get("domains", {})
    primary = domains.get(primary_domain, {})
    secondary_profiles = [domains.get(domain, {}) for domain in secondary_domains]

    merged: dict[str, Any] = {
        "primary_domain": primary_domain,
        "secondary_domains": list(secondary_domains),
    }
    for key in GUIDANCE_KEYS:
        values = list(primary.get(key, []))
        if key in SECONDARY_DOMAIN_KEYS:
            for profile in secondary_profiles:
                values.extend(profile.get(key, []))
        values.extend(default.get(key, []))
        merged[key] = _dedupe(values)
    return merged


def merge_domain_guidance(primary_domain: str, secondary_domains: list[str], static_guidance: dict[str, Any], dynamic_guidance: dict[str, Any] | None = None) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "primary_domain": primary_domain,
        "secondary_domains": list(secondary_domains),
    }
    dynamic_guidance = dynamic_guidance or {}
    for key in GUIDANCE_KEYS:
        values = list(dynamic_guidance.get(key, []))
        values.extend(static_guidance.get(key, []))
        merged[key] = _dedupe([str(value) for value in values if value])
    return merged


def _mock_dynamic_domain_guidance_builder(context: dict[str, Any]) -> dict[str, Any]:
    clinical_topic = context["clinical_topic"]
    primary_domain = context["primary_domain"]
    task_type = context["primary_task_type"]
    secondary_domains = context.get("secondary_domains", [])
    topic_phrase = clinical_topic.lower()
    secondary_phrase = ", ".join(secondary_domains) if secondary_domains else "related cross-domain concerns"
    return {
        "risk_hints": [f"For this {primary_domain} case, explicitly address whether {topic_phrase} could signal a time-sensitive problem."],
        "evidence_hints": [f"Clarify the missing history or examination details that would narrow {topic_phrase}."],
        "constraint_hints": [f"Keep the {task_type} response clinically careful and avoid overstating certainty for {topic_phrase}."],
        "safety_hints": [f"Explain what symptom change in {topic_phrase} should prompt clinician follow-up or urgent care, including {secondary_phrase} when relevant."],
        "unsafe_patterns": [f"Do not claim {topic_phrase} is definitely benign without checking for red flags."],
        "distractor_history": [f"The patient also mentions a background issue that may distract from {topic_phrase}."],
    }


def _coerce_dynamic_domain_guidance(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("domain_guidance") if isinstance(payload.get("domain_guidance"), dict) else payload
    coerced: dict[str, Any] = {}
    for key in GUIDANCE_KEYS:
        values = raw.get(key, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            values = []
        coerced[key] = [str(value).strip() for value in values if str(value).strip()]
    return coerced


def generate_dynamic_domain_guidance(
    environment: ClinicalEnvironment,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
) -> dict[str, Any]:
    prompt = prompt_runner.render(
        "domain_guidance_generate.jinja",
        primary_domain=environment.domain,
        secondary_domains=environment.secondary_domains,
        primary_task_type=environment.primary_task_type,
        secondary_task_types=environment.secondary_task_types,
        clinical_topic=environment.clinical_topic,
        patient_state=environment.patient_state,
        clinical_context=environment.clinical_context,
        evidence_state=environment.evidence_state,
        gold_state=environment.gold_state,
    )
    response = llm_client.complete_json(
        task_name="domain_guidance_generate",
        prompt=prompt,
        context={
            "primary_domain": environment.domain,
            "secondary_domains": environment.secondary_domains,
            "primary_task_type": environment.primary_task_type,
            "secondary_task_types": environment.secondary_task_types,
            "clinical_topic": environment.clinical_topic,
            "patient_state": environment.patient_state,
            "clinical_context": environment.clinical_context,
            "evidence_state": environment.evidence_state,
            "gold_state": environment.gold_state,
        },
        mock_builder=_mock_dynamic_domain_guidance_builder,
    )
    return _coerce_dynamic_domain_guidance(response.payload)


def enrich_environment_with_domain_guidance(
    environment: ClinicalEnvironment,
    guidance_cfg: dict[str, Any],
    dynamic_guidance: dict[str, Any] | None = None,
) -> ClinicalEnvironment:
    static_guidance = resolve_domain_guidance(environment.domain, environment.secondary_domains, guidance_cfg)
    guidance = merge_domain_guidance(environment.domain, environment.secondary_domains, static_guidance, dynamic_guidance)
    clinical_context = deepcopy(environment.clinical_context)
    clinical_context.update(
        {
            "primary_domain": environment.primary_domain,
            "secondary_domains": list(environment.secondary_domains),
            "primary_task_type": environment.primary_task_type,
            "secondary_task_types": list(environment.secondary_task_types),
            "dynamic_domain_guidance_used": bool(dynamic_guidance),
            "domain_risk_hints": guidance["risk_hints"],
            "domain_evidence_hints": guidance["evidence_hints"],
            "domain_constraint_hints": guidance["constraint_hints"],
            "domain_safety_hints": guidance["safety_hints"],
            "domain_unsafe_patterns": guidance["unsafe_patterns"],
            "domain_distractor_history": guidance["distractor_history"],
        }
    )

    gold_state = deepcopy(environment.gold_state)
    must_include = list(gold_state.get("must_include", []))
    for hint in guidance["safety_hints"][:2]:
        if hint not in must_include:
            must_include.append(hint)
    if must_include:
        gold_state["must_include"] = must_include

    return environment.model_copy(update={"clinical_context": clinical_context, "gold_state": gold_state})
