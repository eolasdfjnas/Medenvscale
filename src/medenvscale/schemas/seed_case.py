from __future__ import annotations

from typing import Any
from typing import Literal

from medenvscale.classify.taxonomy import normalize_domain_name, normalize_task_type_name

from .common import StrictBaseModel


class SeedCase(StrictBaseModel):
    seed_id: str
    source: str
    task_id: str
    medqa_id: str
    original_question: str
    original_options: dict[str, str]
    original_answer_key: str | None = None
    original_answer_text: str | None = None
    original_explanation: str | None = None
    task_family: str | None = None
    category_path: list[str] = []
    system_prompt: str | None = None
    verifier_reference: dict[str, Any] = {}
    resource_files: list[str] = []
    starter_code: str | None = None
    access_tier: str | None = None
    primary_domain: str
    secondary_domains: list[str] = []
    primary_task_type: str
    secondary_task_types: list[str] = []
    clinical_topic: str
    medical_concepts: list[str]
    seed_risk_level: Literal["R1", "R2", "R3", "R4", "R5"]
    seed_constraint_level: Literal["C1", "C2", "C3", "C4", "C5"]
    seed_evidence_level: Literal["E1", "E2", "E3", "E4", "E5"]
    usable_for_generation: bool
    filter_reason: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        if "task_id" not in kwargs and "medqa_id" in kwargs:
            kwargs["task_id"] = kwargs["medqa_id"]
        if "medqa_id" not in kwargs and "task_id" in kwargs:
            kwargs["medqa_id"] = kwargs["task_id"]
        if "primary_domain" not in kwargs and "domain" in kwargs:
            kwargs["primary_domain"] = kwargs.pop("domain")
        if "primary_task_type" not in kwargs and "task_type" in kwargs:
            kwargs["primary_task_type"] = kwargs.pop("task_type")
        kwargs["primary_domain"] = normalize_domain_name(kwargs.get("primary_domain"))
        kwargs["secondary_domains"] = [
            normalize_domain_name(str(domain)) for domain in (kwargs.get("secondary_domains", []) or []) if str(domain).strip()
        ]
        kwargs["primary_task_type"] = normalize_task_type_name(kwargs.get("primary_task_type"))
        kwargs["secondary_task_types"] = [
            normalize_task_type_name(str(task_type))
            for task_type in (kwargs.get("secondary_task_types", []) or [])
            if str(task_type).strip()
        ]
        kwargs.setdefault("secondary_domains", [])
        kwargs.setdefault("secondary_task_types", [])
        kwargs.setdefault("category_path", [])
        kwargs.setdefault("verifier_reference", {})
        kwargs.setdefault("resource_files", [])
        super().__init__(**kwargs)

    @property
    def domain(self) -> str:
        return self.primary_domain

    @property
    def task_type(self) -> str:
        return self.primary_task_type
