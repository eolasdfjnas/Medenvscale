from __future__ import annotations

from medenvscale.schemas import ClinicalEnvironment, SeedCase

from .difficulty_builder import build_difficulty


def build_environment_skeleton(seed: SeedCase, state_payload: dict, level: str = "M1") -> ClinicalEnvironment:
    difficulty = build_difficulty(level, axis_priority=["D", "C", "A", "V"])
    return ClinicalEnvironment(
        env_id=f"env_{seed.seed_id}_{level}",
        seed_id=seed.seed_id,
        task_id=seed.task_id,
        medqa_id=seed.medqa_id,
        primary_domain=seed.primary_domain,
        secondary_domains=seed.secondary_domains,
        primary_task_type=seed.primary_task_type,
        secondary_task_types=seed.secondary_task_types,
        clinical_topic=seed.clinical_topic,
        level=level,
        patient_state=state_payload["patient_state"],
        clinical_context=state_payload["clinical_context"],
        evidence_state=state_payload["evidence_state"],
        gold_state=state_payload["gold_state"],
        user_prompt=state_payload["user_prompt"],
        difficulty=difficulty,
    )
