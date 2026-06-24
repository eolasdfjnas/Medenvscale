from __future__ import annotations

from typing import Any

from .common import StrictBaseModel


class ChatMessage(StrictBaseModel):
    role: str
    content: str


class SFTSample(StrictBaseModel):
    id: str
    env_id: str
    messages: list[ChatMessage]
    domain: str
    secondary_domains: list[dict[str, Any]] = []
    task_type: str
    secondary_task_types: list[str] = []
    solution_form: str
    tool_config: dict[str, Any]
    difficulty: dict[str, Any]
    operator_mode: str
    rubrics: list[str]
    verifier_id: str


class PreferenceSample(StrictBaseModel):
    id: str
    env_id: str
    prompt: str
    chosen: str
    rejected: str
    preference_reason: str
    domain: str
    secondary_domains: list[dict[str, Any]] = []
    task_type: str
    secondary_task_types: list[str] = []
    tool_config: dict[str, Any]
    difficulty: dict[str, Any]
    rubric_deltas: dict[str, Any]
    operator_failure_modes: list[str]
    chosen_score: float
    rejected_score: float


class PRMStep(StrictBaseModel):
    step_id: int
    state: str
    action: dict[str, Any]
    label: str
    rubric_hits: list[str] = []
    score: float
    related_axes: list[str] = []
    related_operator_ids: list[str] = []


class PRMSample(StrictBaseModel):
    env_id: str
    trajectory_id: str
    step_id: int
    state: str
    action: dict[str, Any]
    label: str
    rubric_hits: list[str]
    score: float
    related_axes: list[str]
    related_operator_ids: list[str]


class RLVREnv(StrictBaseModel):
    env_id: str
    initial_observation: str
    action_space: list[str]
    tool_config: dict[str, Any]
    verifier: dict[str, Any]
    reward_fn: dict[str, Any]
    max_steps: int
    difficulty: dict[str, Any]
    secondary_domains: list[dict[str, Any]] = []
    secondary_task_types: list[str] = []
