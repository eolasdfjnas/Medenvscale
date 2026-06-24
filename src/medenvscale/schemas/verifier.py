from __future__ import annotations

from .common import StrictBaseModel


class VerifierSpec(StrictBaseModel):
    verifier_id: str
    env_id: str
    verifier_type: str
    solution_form: str
    checks: list[dict]
    hidden_tests: list[dict] = []
    exception_tests: list[dict] = []
    numeric_tolerance: float | None = None
    rubric_links: list[str] = []
    static_checks: list[dict] = []
    generated_from_operator_ids: list[str] = []
