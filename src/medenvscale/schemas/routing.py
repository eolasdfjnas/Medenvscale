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
        secondary_domains = kwargs.get("secondary_domains", []) or []
        if isinstance(secondary_domains, dict):
            secondary_domains = [secondary_domains]
        kwargs["secondary_domains"] = [item if isinstance(item, DomainHint) else DomainHint.model_validate(item) for item in secondary_domains]
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
