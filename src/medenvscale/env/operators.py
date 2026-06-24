from __future__ import annotations

from copy import deepcopy

from medenvscale.schemas import ClinicalEnvironment


def _extend_unique(target: list[str], values: list[str]) -> list[str]:
    for value in values:
        if value not in target:
            target.append(value)
    return target


def _domain_hints(environment: ClinicalEnvironment, key: str, fallback: list[str]) -> list[str]:
    values = environment.clinical_context.get(key, [])
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    return fallback


def apply_operator(environment: ClinicalEnvironment, operator_name: str, strength: int = 1) -> ClinicalEnvironment:
    env = environment.model_copy(deep=True)
    prompt = env.user_prompt
    gold_state = deepcopy(env.gold_state)
    flags = list(env.quality_flags)
    strength = max(1, int(strength))
    constraints = list(env.clinical_context.get("constraints", []))
    missing_information = list(env.evidence_state.get("missing_information", []))
    must_include = list(gold_state.get("must_include", []))
    is_code_agent = env.clinical_context.get("execution_mode") == "code_agent"

    if is_code_agent:
        if operator_name == "add_distractor_history":
            notes = list(env.patient_state.get("background_notes", []))
            distractors = _domain_hints(
                env,
                "domain_distractor_history",
                ["Some repository files may be unrelated to the target biomedical decision."],
            )
            notes = _extend_unique(notes, distractors[: min(strength, len(distractors))])
            env.patient_state["background_notes"] = notes
            prompt += "\nThere may be extra files or historical notes that are not all equally relevant."
        elif operator_name == "increase_risk_salience":
            risk_hints = _domain_hints(
                env,
                "domain_risk_hints",
                ["A wrong rule here could miss a high-risk biomedical condition."],
            )
            selected = risk_hints[: min(strength, len(risk_hints))]
            env.patient_state["task_stakes"] = " ".join(selected)
            gold_state["must_mention_urgency"] = True
            prompt += "\nYour output will be checked automatically and should be conservative around safety-critical edge cases."
            must_include = _extend_unique(must_include, selected)
        elif operator_name == "add_information_gap":
            evidence_hints = _domain_hints(env, "domain_evidence_hints", ["schema details", "missing feature definitions"])
            missing_information = _extend_unique(missing_information, evidence_hints[: min(strength, len(evidence_hints))])
            prompt += "\nSome schema details are incomplete, so note what you would inspect before finalizing."
        elif operator_name == "add_evidence_requirement":
            evidence_hints = _domain_hints(env, "domain_evidence_hints", ["inspect the provided resources"])
            must_include.append("justify the answer using the available resources or verifier reference")
            must_include = _extend_unique(must_include, evidence_hints[: min(strength, len(evidence_hints))])
        elif operator_name == "add_constraint":
            constraint_hints = _domain_hints(
                env,
                "domain_constraint_hints",
                ["Keep the solution executable and avoid unsupported assumptions."],
            )
            safety_hints = _domain_hints(env, "domain_safety_hints", [])
            constraints = _extend_unique(constraints, constraint_hints[: min(strength, len(constraint_hints))])
            constraints = _extend_unique(constraints, safety_hints[:1])
            must_include = _extend_unique(must_include, safety_hints[: min(strength, len(safety_hints))])
        elif operator_name == "add_ambiguity":
            prompt += "\nSome field names, labels, or file boundaries may be ambiguous."
            if strength > 1:
                prompt += " Resolve ambiguity by stating the assumption that keeps the task verifiable."
            flags.append("contains_ambiguity")
    elif operator_name == "add_distractor_history":
        history = list(env.patient_state.get("history", []))
        distractors = _domain_hints(
            env,
            "domain_distractor_history",
            ["The patient also mentions a less relevant chronic symptom."],
        )
        history = _extend_unique(history, distractors[: min(strength, len(distractors))])
        env.patient_state["history"] = history
        prompt += " There is also some background information that may or may not matter."
    elif operator_name == "increase_risk_salience":
        risk_hints = _domain_hints(env, "domain_risk_hints", ["Do not miss serious causes."])
        selected = risk_hints[: min(strength, len(risk_hints))]
        env.patient_state["risk_emphasis"] = " ".join(selected)
        gold_state["must_mention_urgency"] = True
        prompt += " I am worried this could be something serious."
        if selected:
            prompt += f" {selected[0]}"
        must_include = _extend_unique(must_include, selected)
    elif operator_name == "add_information_gap":
        evidence_hints = _domain_hints(env, "domain_evidence_hints", ["medication list"])
        missing_information = _extend_unique(missing_information, evidence_hints[: min(strength, len(evidence_hints))])
        prompt += " I do not know all of the relevant details yet."
    elif operator_name == "add_evidence_requirement":
        evidence_hints = _domain_hints(env, "domain_evidence_hints", ["additional context"])
        must_include.append("explain what additional information is needed")
        must_include = _extend_unique(must_include, evidence_hints[: min(strength, len(evidence_hints))])
    elif operator_name == "add_constraint":
        constraint_hints = _domain_hints(
            env,
            "domain_constraint_hints",
            ["The answer should stay patient-friendly and avoid overclaiming."],
        )
        safety_hints = _domain_hints(env, "domain_safety_hints", [])
        constraints = _extend_unique(constraints, constraint_hints[: min(strength, len(constraint_hints))])
        constraints = _extend_unique(constraints, safety_hints[:1])
        must_include = _extend_unique(must_include, safety_hints[: min(strength, len(safety_hints))])
    elif operator_name == "add_ambiguity":
        prompt += " Some parts of the history may be incomplete or ambiguous."
        if strength > 1:
            prompt += " The symptoms may overlap across more than one clinical category."
        flags.append("contains_ambiguity")

    env.user_prompt = prompt
    env.gold_state = gold_state
    if must_include:
        env.gold_state["must_include"] = must_include
    if missing_information:
        env.evidence_state["missing_information"] = missing_information
    if constraints:
        env.clinical_context["constraints"] = constraints
    env.quality_flags = flags
    env.operators_applied = [*env.operators_applied, operator_name]
    return env


def apply_operators(environment: ClinicalEnvironment, operator_names: list[str] | list[dict[str, int | str]]) -> ClinicalEnvironment:
    env = environment
    for operator in operator_names:
        if isinstance(operator, dict):
            env = apply_operator(env, str(operator["name"]), strength=int(operator.get("strength", 1)))
        else:
            env = apply_operator(env, operator)
    return env
