from __future__ import annotations

from typing import Any

from .common import StrictBaseModel


class MedAgentGymTask(StrictBaseModel):
    task_id: str
    source_split: str
    idx: str | None = None
    problem: str
    solution: str
    context: str
    signature: str | None = None
    code: str | None = None
    wrong_code: str | None = None
    task_family: str | None = None
    category: str | None = None
    category_path: list[str] = []
    resource_files: list[str] = []
    placeholder_token: str = "<<insert solution here>>"
    has_placeholder: bool = True
    context_summary: str | None = None
    ground_truth_output_signature: dict[str, Any] = {}
    code_execution_result: dict[str, Any] = {}
    seed_execution_case: dict[str, Any] = {}
    seed_case_audit: dict[str, Any] = {}
    execution_status: str | None = None
    repair_attempts: int = 0
    repair_succeeded: bool = False
    raw_metadata: dict[str, Any] = {}

    @property
    def medqa_id(self) -> str:
        return self.task_id

    @property
    def question(self) -> str:
        return self.problem

    @property
    def options(self) -> dict[str, str]:
        return {}

    @property
    def answer_key(self) -> str | None:
        return None

    @property
    def answer_text(self) -> str | None:
        return self.solution

    @property
    def explanation(self) -> str | None:
        return None
