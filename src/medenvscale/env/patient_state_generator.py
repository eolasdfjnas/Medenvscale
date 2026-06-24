from __future__ import annotations

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import SeedCase


def _is_medagent_seed(seed: SeedCase) -> bool:
    return seed.source == "medagentgym" or bool(seed.task_family or seed.resource_files)


def _clinical_patient_prompt(seed: SeedCase) -> str:
    answer = seed.original_answer_text or "the likely diagnosis"
    if seed.primary_task_type == "triage_urgent_management":
        return f"I am worried about {seed.clinical_topic}. Based on these symptoms, what should I do next?"
    if seed.primary_task_type == "medication_safety":
        return f"I have a question about {seed.clinical_topic}. Is it safe, and what precautions should I take?"
    if seed.primary_task_type == "diagnosis_reasoning":
        return f"I have symptoms related to {seed.clinical_topic}. What could this mean?"
    return f"I have a question related to {seed.clinical_topic}. Please explain what I should know about {answer}."


def _mock_clinical_state_builder(seed: SeedCase) -> dict:
    risk = seed.seed_risk_level
    return {
        "patient_state": {
            "demographics": {
                "age_group": "older_adult" if "58" in seed.original_question or "65" in seed.original_question else "adult",
                "special_population": "pregnant" if "pregnan" in seed.original_question.lower() else "none",
            },
            "symptoms": [seed.clinical_topic],
            "reported_concern": seed.original_question,
            "risk_level": risk,
        },
        "clinical_context": {
            "domain": seed.domain,
            "primary_task_type": seed.primary_task_type,
            "secondary_task_types": seed.secondary_task_types,
            "task_type": seed.primary_task_type,
            "question_origin": "medqa_seed",
        },
        "evidence_state": {
            "known_facts": [seed.original_answer_text or "", *seed.medical_concepts][:4],
            "missing_information": ["duration", "severity", "relevant history"],
        },
        "gold_state": {
            "correct_answer": seed.original_answer_text,
            "recommended_action": "urgent evaluation" if risk in {"R4", "R5"} else "appropriate clinical follow-up",
            "must_include": seed.medical_concepts[:3] or [seed.clinical_topic],
            "risk_level": risk,
        },
        "user_prompt": _clinical_patient_prompt(seed),
    }


def _medagent_user_prompt(seed: SeedCase) -> str:
    lines = [
        seed.system_prompt or "You are a biomedical coding agent. Solve the task carefully and produce verifiable output.",
        "",
        f"Task: {seed.original_question}",
    ]
    if seed.resource_files:
        lines.append("Available resources:")
        lines.extend(f"- {path}" for path in seed.resource_files[:8])
    if seed.starter_code:
        lines.append("Starter code is provided and should be completed rather than replaced wholesale.")
    if seed.original_answer_text:
        lines.append("Return an answer that is consistent with the expected solution pattern and explain the reasoning briefly.")
    return "\n".join(lines)


def _mock_medagent_state_builder(seed: SeedCase) -> dict:
    risk = seed.seed_risk_level
    constraints = [
        "Prefer executable reasoning over free-form speculation.",
        "Use only the provided task resources unless a fallback assumption is explicitly stated.",
    ]
    if seed.resource_files:
        constraints.append("Inspect relevant files before finalizing the answer.")
    if seed.starter_code:
        constraints.append("Preserve the function signature and extend the provided scaffold.")
    return {
        "patient_state": {
            "task_summary": seed.original_question,
            "task_family": seed.task_family or seed.primary_task_type,
            "resource_files": seed.resource_files,
            "starter_code_present": bool(seed.starter_code),
            "access_tier": seed.access_tier or "open",
            "risk_level": risk,
        },
        "clinical_context": {
            "domain": seed.domain,
            "primary_task_type": seed.primary_task_type,
            "secondary_task_types": seed.secondary_task_types,
            "task_type": seed.primary_task_type,
            "question_origin": "medagentgym_seed",
            "execution_mode": "code_agent",
            "task_family": seed.task_family,
            "category_path": seed.category_path,
            "constraints": constraints,
        },
        "evidence_state": {
            "known_facts": [seed.original_answer_text or "", *seed.medical_concepts][:4],
            "resource_manifest": seed.resource_files,
            "missing_information": ["input schema details", "edge-case coverage", "verifier expectations"],
        },
        "gold_state": {
            "correct_answer": seed.original_answer_text or "Produce a validated executable solution",
            "recommended_action": "inspect the resources, reason about the biomedical objective, and produce a verifier-aligned solution",
            "must_include": [
                "inspect relevant resources before concluding",
                "produce an executable or operational answer",
                *seed.medical_concepts[:2],
            ][:4],
            "risk_level": risk,
            "verifier_reference": seed.verifier_reference,
            "resource_files": seed.resource_files,
        },
        "user_prompt": _medagent_user_prompt(seed),
    }


def _default_patient_state_payload(seed: SeedCase) -> dict:
    if _is_medagent_seed(seed):
        return _mock_medagent_state_builder(seed)
    return _mock_clinical_state_builder(seed)


def _coerce_patient_state_payload(payload: dict, seed: SeedCase) -> dict:
    default = _default_patient_state_payload(seed)
    coerced = dict(default)

    if payload.get("patient_state"):
        coerced["patient_state"] = payload["patient_state"]
    if payload.get("clinical_context"):
        clinical_context = dict(default["clinical_context"])
        clinical_context.update(payload["clinical_context"])
        coerced["clinical_context"] = clinical_context
    if payload.get("evidence_state"):
        evidence_state = dict(default["evidence_state"])
        evidence_state.update(payload["evidence_state"])
        coerced["evidence_state"] = evidence_state
    if payload.get("gold_state"):
        gold_state = dict(default["gold_state"])
        gold_state.update(payload["gold_state"])
        coerced["gold_state"] = gold_state
    if payload.get("user_prompt"):
        coerced["user_prompt"] = payload["user_prompt"]

    patient_scenario = payload.get("patient_scenario")
    if isinstance(patient_scenario, dict):
        patient = patient_scenario.get("patient", {})
        age = patient.get("age")
        age_group = "adult"
        if isinstance(age, int) and age >= 65:
            age_group = "older_adult"
        pregnancy_status = str(patient.get("pregnancy_status") or "").lower()
        special_population = "pregnant" if "pregnan" in pregnancy_status or "gestation" in pregnancy_status else "none"
        symptoms = patient.get("symptoms", {})
        symptom_names = [key.replace("_", " ") for key in symptoms.keys()] if isinstance(symptoms, dict) else []
        coerced["patient_state"] = {
            "demographics": {
                "age_group": age_group,
                "special_population": special_population,
            },
            "symptoms": symptom_names or [seed.clinical_topic],
            "reported_concern": seed.original_question,
            "risk_level": seed.seed_risk_level,
        }
        known_facts = [seed.original_answer_text or "", *seed.medical_concepts]
        diagnosis = patient_scenario.get("diagnosis")
        if diagnosis:
            known_facts.append(str(diagnosis))
        recommended_treatment = patient_scenario.get("recommended_treatment")
        if recommended_treatment:
            known_facts.append(str(recommended_treatment))
        coerced["clinical_context"] = {
            "domain": seed.domain,
            "primary_task_type": seed.primary_task_type,
            "secondary_task_types": seed.secondary_task_types,
            "task_type": seed.primary_task_type,
            "question_origin": "medqa_seed",
        }
        coerced["evidence_state"] = {
            "known_facts": [fact for fact in known_facts if fact][:4],
            "missing_information": list(default["evidence_state"]["missing_information"]),
        }
        gold_state = dict(default["gold_state"])
        if diagnosis:
            gold_state["diagnosis"] = str(diagnosis)
        if recommended_treatment:
            gold_state["recommended_treatment"] = str(recommended_treatment)
        coerced["gold_state"] = gold_state
        coerced["user_prompt"] = _clinical_patient_prompt(seed)

    return coerced


def generate_patient_state(seed: SeedCase, llm_client: LLMClient, prompt_runner: PromptRunner) -> dict:
    template_name = "medagentgym_to_task_state.jinja" if _is_medagent_seed(seed) else "medqa_to_patient_state.jinja"
    prompt = prompt_runner.render(
        template_name,
        question=seed.original_question,
        answer_text=seed.original_answer_text,
        clinical_topic=seed.clinical_topic,
        task_type=seed.primary_task_type,
        task_family=seed.task_family,
        resource_files=seed.resource_files,
        system_prompt=seed.system_prompt,
    )
    response = llm_client.complete_json(
        task_name="medagentgym_to_task_state" if _is_medagent_seed(seed) else "medqa_to_patient_state",
        prompt=prompt,
        context={"seed": seed},
        mock_builder=lambda context: _default_patient_state_payload(context["seed"]),
    )
    return _coerce_patient_state_payload(response.payload, seed)
