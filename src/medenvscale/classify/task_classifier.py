from __future__ import annotations

from medenvscale.classify.taxonomy import normalize_task_type_name


TASK_TYPE_PRIORITY = [
    "biostatistics_calculation",
    "ethics_quality_safety",
    "triage_urgent_management",
    "medication_safety",
    "treatment_planning",
    "diagnostic_workup",
    "evidence_interpretation",
    "diagnosis_reasoning",
    "prevention_screening_counseling",
    "medical_knowledge_mechanism",
]


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _matches_biostatistics(text: str) -> bool:
    return _contains_any(
        text,
        [
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "positive predictive value",
            "negative predictive value",
            "confidence interval",
            "odds ratio",
            "hazard ratio",
            "cohort",
            "case-control",
            "prospective",
            "retrospective",
            "randomized controlled trial",
            "study",
        ],
    )


def _matches_ethics_quality_safety(text: str) -> bool:
    return _contains_any(
        text,
        [
            "closed-loop communication",
            "medical error",
            "patient safety",
            "quality improvement",
            "professional conduct",
            "professionalism",
            "ethics",
            "satisfaction driven healthcare",
            "informed consent",
        ],
    )


def _matches_triage(text: str) -> bool:
    urgent_terms = [
        "emergency",
        "shock",
        "respiratory failure",
        "altered mental status",
        "trauma",
        "overdose",
        "poison",
        "toxic ingestion",
        "cauda equina",
        "stroke",
        "myocardial infarction",
        "acute limb ischemia",
        "urgent management",
        "should i go to hospital",
        "should i go to the hospital",
        "seek urgent",
        "urgent evaluation",
        "emergency department",
        "first step in management",
    ]
    if _contains_any(text, urgent_terms):
        return True
    return _contains_any(text, ["next best step", "most appropriate next step"]) and _contains_any(
        text,
        [
            "acute",
            "sudden",
            "severe",
            "unstable",
            "fever",
            "dyspnea",
            "weakness",
            "confusion",
            "bleeding",
            "chest pain",
            "troponin",
            "ecg",
            "ekg",
            "stroke",
        ],
    )


def _matches_medication_safety(text: str) -> bool:
    return _contains_any(
        text,
        [
            "adverse effect",
            "side effect",
            "contraindication",
            "toxicity",
            "overdose",
            "antidote",
            "drug interaction",
            "safe to take",
            "safe in pregnancy",
            "chemotherapy regimen",
            "which medication",
        ],
    )


def _matches_treatment_planning(text: str) -> bool:
    if _matches_triage(text):
        return False
    return _contains_any(
        text,
        [
            "best treatment",
            "most appropriate treatment",
            "most appropriate management",
            "next step in management",
            "which medication is indicated",
            "which of the following medications is indicated",
            "most appropriate pharmacotherapy",
            "therapy",
            "management after delivery",
        ],
    )


def _matches_diagnostic_workup(text: str) -> bool:
    return _contains_any(
        text,
        [
            "next step in diagnosis",
            "most appropriate next step in diagnosis",
            "best next test",
            "best next step in diagnosis",
            "confirm the probable condition",
            "which test should confirm",
            "most appropriate diagnostic test",
            "confirm the diagnosis",
        ],
    )


def _matches_evidence_interpretation(text: str) -> bool:
    return _contains_any(
        text,
        [
            "ecg",
            "ekg",
            "lab",
            "troponin",
            "x-ray",
            "ct",
            "mri",
            "ultrasound",
            "imaging",
            "test result",
            "results show",
            "biopsy",
            "urinalysis",
            "csf",
            "abg",
            "screening test",
        ],
    )


def _matches_diagnosis_reasoning(text: str) -> bool:
    return _contains_any(
        text,
        [
            "most likely diagnosis",
            "most likely cause",
            "most likely etiology",
            "most likely mechanism",
            "which of the following is the most likely diagnosis",
            "which of the following is the most likely cause",
            "pathophysiology",
            "what is the most likely diagnosis",
        ],
    )


def _matches_prevention_counseling(text: str) -> bool:
    return _contains_any(
        text,
        [
            "prevented",
            "prevention",
            "vaccine",
            "vaccination",
            "screening",
            "counsel",
            "health maintenance",
            "routine checkup",
            "safe sleep",
            "avoid sun exposure",
            "lifestyle",
        ],
    )


def _matches_medical_knowledge(text: str) -> bool:
    return _contains_any(
        text,
        [
            "enzyme",
            "gene",
            "chromosome",
            "embryologic",
            "embryology",
            "ubiquitination",
            "receptor",
            "transporter",
            "kinetics",
            "mechanism",
            "genetic principle",
            "post-translational",
        ],
    )


def infer_task_types(question: str, answer_text: str | None = None) -> tuple[str, list[str]]:
    text = f"{question} {answer_text or ''}".lower()
    matched: list[str] = []
    if _matches_biostatistics(text):
        matched.append("biostatistics_calculation")
    if _matches_ethics_quality_safety(text):
        matched.append("ethics_quality_safety")
    if _matches_triage(text):
        matched.append("triage_urgent_management")
    if _matches_medication_safety(text):
        matched.append("medication_safety")
    if _matches_treatment_planning(text):
        matched.append("treatment_planning")
    if _matches_diagnostic_workup(text):
        matched.append("diagnostic_workup")
    if _matches_evidence_interpretation(text):
        matched.append("evidence_interpretation")
    if _matches_diagnosis_reasoning(text):
        matched.append("diagnosis_reasoning")
    if _matches_prevention_counseling(text):
        matched.append("prevention_screening_counseling")
    if _matches_medical_knowledge(text):
        matched.append("medical_knowledge_mechanism")
    if not matched:
        return "diagnosis_reasoning", []

    ordered = [task_type for task_type in TASK_TYPE_PRIORITY if task_type in matched]
    primary_task_type = ordered[0]
    secondary_task_types = ordered[1:]
    return primary_task_type, secondary_task_types


def infer_task_type(question: str, answer_text: str | None = None) -> str:
    primary_task_type, _ = infer_task_types(question, answer_text)
    return normalize_task_type_name(primary_task_type)
