from __future__ import annotations

from typing import Any

from medenvscale.classify.taxonomy import (
    normalize_domain_name,
    normalize_solution_form_name,
    normalize_task_type_name,
)

from .common import StrictBaseModel


class DomainHint(StrictBaseModel):
    domain: str
    relevance: float = 0.0
    reason: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        kwargs["domain"] = normalize_domain_name(kwargs.get("domain"))
        try:
            kwargs["relevance"] = float(kwargs.get("relevance", 0.0))
        except (TypeError, ValueError):
            kwargs["relevance"] = 0.0
        super().__init__(**kwargs)


def normalize_domain_hints(raw_secondary_domains: Any) -> list[DomainHint]:
    if not raw_secondary_domains:
        return []
    raw_items = raw_secondary_domains if isinstance(raw_secondary_domains, list) else [raw_secondary_domains]
    hints: list[DomainHint] = []
    for item in raw_items:
        hints.extend(_coerce_domain_hint_item(item))
    return hints


def _coerce_domain_hint_item(item: Any) -> list[DomainHint]:
    if isinstance(item, DomainHint):
        return [item]
    if isinstance(item, str):
        text = item.strip()
        return [DomainHint(domain=text, relevance=0.5)] if text else []
    if not isinstance(item, dict):
        return []
    if "domain" in item:
        return [DomainHint.model_validate(item)]

    hints: list[DomainHint] = []
    for domain, value in item.items():
        domain_text = str(domain).strip()
        if not domain_text:
            continue
        payload: dict[str, Any] = {"domain": domain_text}
        if isinstance(value, dict):
            payload["relevance"] = value.get("relevance", value.get("score", 0.5))
            if value.get("reason") is not None:
                payload["reason"] = str(value.get("reason"))
        else:
            payload["relevance"] = value
        hints.append(DomainHint(**payload))
    return hints


class RoutingResult(StrictBaseModel):
    task_id: str
    source_split: str
    primary_domain: str
    secondary_domains: list[DomainHint] = []
    primary_task_type: str
    solution_form: str
    secondary_task_types: list[str] = []
    domain_concepts: list[str] = []
    required_capabilities: list[str] = []
    verifier_type_hint: str | None = None
    routing_reason: str
    confidence: float
    needs_review: bool = False
    routing_trace: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        kwargs["primary_domain"] = normalize_domain_name(kwargs.get("primary_domain") or kwargs.get("domain"))
        kwargs["secondary_domains"] = normalize_domain_hints(kwargs.get("secondary_domains", []) or [])
        kwargs["primary_task_type"] = normalize_task_type_name(kwargs.get("primary_task_type") or kwargs.get("task_type"))
        kwargs["solution_form"] = normalize_solution_form_name(kwargs.get("solution_form"))
        kwargs["secondary_task_types"] = [
            normalize_task_type_name(str(task_type))
            for task_type in (kwargs.get("secondary_task_types", []) or [])
            if str(task_type).strip()
        ]
        kwargs.setdefault("domain_concepts", [])
        kwargs.setdefault("required_capabilities", [])
        kwargs.setdefault("routing_trace", {})
        super().__init__(**kwargs)

    @property
    def domain(self) -> str:
        return self.primary_domain

    @property
    def task_type(self) -> str:
        return self.primary_task_type
