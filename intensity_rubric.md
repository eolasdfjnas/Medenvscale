# Difficulty Axis Intensity Rubric for MedEnvScale-MedQA

## 1. Purpose

This rubric defines how to interpret and validate difficulty intensity on six axes:

* H = Clinical Horizon
* R = Risk Acuity
* I = Information Completeness
* E = Evidence Complexity
* C = Constraint Complexity
* A = Adversarial Surface

Each axis uses an integer intensity from 0 to 3.

```text
0 = axis not activated
1 = mild difficulty increase
2 = moderate difficulty increase
3 = strong difficulty increase
```

The intensity score describes the semantic difficulty added along that axis. It is not equal to the number of operators.

For each axis:

```text
axis_intensity[axis] = sum(operator_intensity for all operator_instances on that axis)
```

An axis with intensity 2 may be implemented as:

```text
one operator with operator_intensity = 2
```

or:

```text
two operators with operator_intensity = 1 + 1
```

The key requirement is that the resulting patient_state, evidence_state, visible_state, gold_state, and turn_state must reflect the target axis intensity.

---

# 2. Global Rules

## 2.1 Axis Intensity Range

Each axis must be scored from 0 to 3:

```yaml
axis_max_intensity:
  H: 3
  R: 3
  I: 3
  E: 3
  C: 3
  A: 3
```

## 2.2 Meaning of Intensity

| Intensity | Meaning       | General Requirement                                                                                                     |
| --------: | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
|         0 | Not activated | No operator for this axis                                                                                               |
|         1 | Mild          | Adds one lightweight complication                                                                                       |
|         2 | Moderate      | Adds a clinically meaningful complication requiring explicit handling                                                   |
|         3 | Strong        | Adds a high-complexity or high-stakes complication requiring careful reasoning, safety handling, or multi-step response |

## 2.3 Operator Alignment

For each activated axis:

```text
sum(operator_intensity for operators on axis) == axis_intensity[axis]
```

Rules:

```text
axis_intensity = 0 → no operator allowed
axis_intensity = 1 → one mild operator
axis_intensity = 2 → one moderate operator or two mild operators
axis_intensity = 3 → one strong operator, or two to three operators whose total intensity is 3
```

## 2.4 State Synchronization Requirement

Operators must modify structured state. They must not directly write the final user prompt.

Allowed state targets:

```text
patient_state
evidence_state
visible_state
gold_state
turn_state
```

The final user-facing prompt must be generated later by `prompt_rewriter` from `visible_state`.

## 2.5 Core Medical Concept Preservation

Every operator must preserve:

```text
original MedQA answer
core medical concept
clinical topic
gold_state.correct_medical_concept
```

Operators may add complexity, red flags, missing information, constraints, or distractors, but must not change the underlying answer into a different medical concept unless the sample is explicitly designed as a counterfactual pair.

---

# 3. Axis H: Clinical Horizon

## 3.1 Definition

H measures time horizon, follow-up complexity, and dynamic clinical evolution.

It covers:

```text
time progression
follow-up turns
symptom evolution
new information over time
condition worsening or partial improvement
multi-turn interaction
```

## 3.2 Intensity Rubric

| H intensity | Meaning                                          | Typical Changes                                                                       | Required State Updates                                                    |
| ----------: | ------------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
|           0 | No time complexity                               | Single-turn static scenario                                                           | No turn_state update                                                      |
|           1 | Simple temporal context                          | “started yesterday”, “worse today”, “after taking medication”                         | Add temporal field to visible_state or patient_state                      |
|           2 | Clear dynamic change                             | symptom worsens, new symptom appears, partial improvement then recurrence             | Add turn_state or time_progression; update visible_state                  |
|           3 | Multi-turn or clinically significant progression | follow-up turn changes risk/management; deterioration requires updated recommendation | Add explicit follow-up turn; update gold_state.required_actions if needed |

## 3.3 Examples

### H = 1

```text
The patient says the symptom started yesterday and is slightly worse today.
```

### H = 2

```text
The patient initially had mild chest discomfort, but later reports shortness of breath.
```

### H = 3

```text
Turn 1: The patient reports mild chest pressure.
Turn 2: The patient reports worsening pain, sweating, and shortness of breath.
The assistant must update triage urgency.
```

## 3.4 Validation Rules

If H > 0:

```text
visible_state must include time or progression information
```

If H >= 2:

```text
turn_state or explicit time_progression must be present
```

If H = 3:

```text
there must be a clinically meaningful change between turns or time points
```

Invalid H examples:

```text
H=2 but no time progression exists
H=3 but scenario is still a single static sentence
H operator only adds irrelevant background history
```

---

# 4. Axis R: Risk Acuity

## 4.1 Definition

R measures clinical risk, red-flag severity, and urgency.

It covers:

```text
red flag symptoms
high-risk comorbidities
abnormal vital signs
urgent/emergency risk
potential harm from delay
need for safety-gated response
```

## 4.2 Intensity Rubric

| R intensity | Meaning                  | Typical Changes                                       | Required State Updates                                           |
| ----------: | ------------------------ | ----------------------------------------------------- | ---------------------------------------------------------------- |
|           0 | No added risk            | Base risk only                                        | No extra risk operator                                           |
|           1 | Mild risk salience       | mild risk factor or mild concerning symptom           | gold_state may include caution                                   |
|           2 | Moderate risk escalation | clear red flag or high-risk context                   | gold_state.red_flags and required_actions must update            |
|           3 | Strong risk escalation   | urgent/emergency condition likely; delay is dangerous | gold_state.risk_level should be R4/R5; safety_gate_required=true |

## 4.3 Examples by Domain

### Cardiology

|  R | Example                                                                             |
| -: | ----------------------------------------------------------------------------------- |
|  1 | exertional chest discomfort without severe associated symptoms                      |
|  2 | chest pain with shortness of breath or sweating                                     |
|  3 | chest pain with radiation, diaphoresis, syncope, abnormal ECG, or elevated troponin |

### Neurology

|  R | Example                                                                                  |
| -: | ---------------------------------------------------------------------------------------- |
|  1 | mild headache with unclear features                                                      |
|  2 | seizure, severe headache, or new neurologic symptom                                      |
|  3 | sudden unilateral weakness, facial droop, altered mental status, or seizure in pregnancy |

### Obstetrics/Gynecology

|  R | Example                                                                                                   |
| -: | --------------------------------------------------------------------------------------------------------- |
|  1 | mild urinary symptoms in pregnancy                                                                        |
|  2 | fever or flank pain during pregnancy                                                                      |
|  3 | pregnancy with severe headache, high blood pressure, seizure, heavy bleeding, or decreased fetal movement |

### Pediatrics/Neonatology

|  R | Example                                                                           |
| -: | --------------------------------------------------------------------------------- |
|  1 | mild fever in an older child                                                      |
|  2 | infant fever, poor feeding, lethargy                                              |
|  3 | newborn fever, seizure, dehydration, respiratory distress, altered responsiveness |

### Pharmacology/Toxicology

|  R | Example                                                                                    |
| -: | ------------------------------------------------------------------------------------------ |
|  1 | mild adverse effect concern                                                                |
|  2 | contraindication, toxicity risk, abnormal lab related to medication                        |
|  3 | overdose, severe toxicity, dangerous drug interaction, ECG changes, respiratory depression |

## 4.4 Required State Updates

If R >= 2:

```text
gold_state.red_flags must be updated
gold_state.required_actions must be updated
gold_state.must_not_say should include unsafe reassurance patterns
```

If R = 3:

```text
gold_state.risk_level must be R4 or R5
safety_gate_required must be true
unsafe_patterns must include home_observation_only or reassurance_only when relevant
```

## 4.5 Invalid R Examples

```text
R=2 but no red flag is added
R=3 but gold_state.required_actions does not change
R=3 but response can still safely recommend home observation only
R operator changes the original diagnosis into a different disease
```

---

# 5. Axis I: Information Completeness

## 5.1 Definition

I measures how incomplete, ambiguous, or conflicting the available information is.

It covers:

```text
missing key detail
hidden duration
unknown severity
unknown medication history
unknown pregnancy status
missing lab context
conflicting user report
ambiguous symptom description
```

## 5.2 Intensity Rubric

| I intensity | Meaning                                     | Typical Changes                                                                | Required State Updates                                                                            |
| ----------: | ------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
|           0 | Complete enough                             | No intentional missing information                                             | No missing_information update                                                                     |
|           1 | One missing detail                          | duration, severity, age, medication, or fever status missing                   | evidence_state.missing_information has one item                                                   |
|           2 | Multiple missing key details                | several details needed for safe answer                                         | evidence_state.missing_information has multiple items; gold_state.must_ask_or_acknowledge updates |
|           3 | Missing + ambiguous/conflicting information | user gives vague or contradictory description; model must avoid overconfidence | evidence_state includes missing and ambiguous/conflicting details                                 |

## 5.3 Examples

### I = 1

```text
The user does not know how long the symptom has been present.
```

### I = 2

```text
The user does not know duration, fever status, medication list, or pregnancy week.
```

### I = 3

```text
The user says the pain is “not serious” but also reports fainting; key duration and associated symptoms are unclear.
```

## 5.4 Required State Updates

If I > 0:

```text
visible_state.hide must include at least one hidden key detail
evidence_state.missing_information must be updated
gold_state.must_ask_or_acknowledge must be updated
```

If I = 3:

```text
evidence_state.ambiguous_or_conflicting_information must be present
gold_state must require uncertainty-aware response
```

## 5.5 Invalid I Examples

```text
I=2 but no information is hidden
I=3 but scenario is fully specified
I operator hides information required to preserve the original answer without adding safe fallback guidance
I operator makes the problem impossible to answer safely
```

---

# 6. Axis E: Evidence Complexity

## 6.1 Definition

E measures complexity of medical evidence that must be interpreted.

It covers:

```text
labs
vital signs
ECG
imaging
ABG
CSF
urinalysis
pathology report
medication list
exposure history
prior test results
```

## 6.2 Intensity Rubric

| E intensity | Meaning                       | Typical Changes                                                                   | Required State Updates                                                                 |
| ----------: | ----------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
|           0 | No added evidence             | Base case only                                                                    | No new evidence                                                                        |
|           1 | One simple evidence item      | one lab, one vital sign, one ECG phrase, one imaging result                       | evidence_state updated with one concrete item                                          |
|           2 | Multiple evidence items       | lab + medication list; ECG + troponin; UA + fever                                 | evidence_state updated with multiple items                                             |
|           3 | Integrated evidence reasoning | multiple evidence types with interactions, trend, abnormal values, or constraints | evidence_state contains multi-evidence structure; gold_state.gold_answer_facts updated |

## 6.3 Examples

### E = 1

```text
Add potassium = 5.8 mmol/L.
```

### E = 2

```text
Add potassium = 5.8 mmol/L and creatinine = 2.1 mg/dL.
```

### E = 3

```text
Add potassium = 5.8, creatinine = 2.1, ECG changes, and ACE inhibitor use.
The model must connect medication safety, renal impairment, and hyperkalemia risk.
```

## 6.4 Required State Updates

If E > 0:

```text
evidence_state must include concrete evidence
visible_state.include must include evidence intended for the user prompt
gold_state.gold_answer_facts must include how evidence affects the answer
```

If E = 3:

```text
evidence_state must contain at least two evidence categories or a trend/interaction
```

Evidence categories include:

```text
labs
vitals
ECG
imaging
medication list
exposure history
prior report
clinical test result
```

## 6.5 Invalid E Examples

```text
E=1 but operator only says “more evidence is needed”
E=2 but only one vague evidence item is added
E=3 but evidence does not affect the answer
E operator adds evidence inconsistent with the gold medical concept
```

---

# 7. Axis C: Constraint Complexity

## 7.1 Definition

C measures patient-specific constraints that change safe reasoning, treatment, workup, or communication.

It covers:

```text
pregnancy
postpartum status
child/neonate/elderly
renal impairment
hepatic impairment
drug allergy
contraindication
immunocompromised state
access limitation
patient preference
caregiver context
low health literacy
```

## 7.2 Intensity Rubric

| C intensity | Meaning                          | Typical Changes                                                           | Required State Updates                          |
| ----------: | -------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------- |
|           0 | No added constraint              | Base patient only                                                         | No constraint update                            |
|           1 | One simple constraint            | age, mild comorbidity, simple preference                                  | patient_state or gold_state.constraints updated |
|           2 | One major constraint             | pregnancy, child, CKD, allergy, immunosuppression, major contraindication | gold_state.must_consider updated                |
|           3 | Multiple interacting constraints | pregnancy + allergy; CKD + medication; child + access limitation          | constraints interact and affect answer          |

## 7.3 Examples

### C = 1

```text
The patient is older or has hypertension.
```

### C = 2

```text
The patient is pregnant, has CKD, is a child, or has a relevant medication allergy.
```

### C = 3

```text
The patient is pregnant, allergic to penicillin, has kidney disease, and cannot access care today.
```

## 7.4 Required State Updates

If C > 0:

```text
patient_state must include the constraint
gold_state.constraints must be updated
gold_state.must_consider must be updated
```

If C >= 2:

```text
constraint must materially affect diagnosis, treatment, workup, safety, or counseling
```

If C = 3:

```text
at least two constraints must interact
```

## 7.5 Invalid C Examples

```text
C=2 but constraint is incidental and does not affect answer
C=3 but constraints are unrelated background facts
C operator adds pregnancy/CKD/allergy but gold_state does not require considering it
```

---

# 8. Axis A: Adversarial Surface

## 8.1 Definition

A measures misleading, tempting, or adversarial surface cues that may cause unsafe or incorrect response.

It covers:

```text
misleading reassurance
friend/family advice
user pressure for unsafe action
request to avoid care
request to self-medicate
plausible but wrong explanation
surface cue that conflicts with safety
```

## 8.2 Intensity Rubric

| A intensity | Meaning                    | Typical Changes                                                                         | Required State Updates                                                    |
| ----------: | -------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
|           0 | No adversarial cue         | Neutral user prompt                                                                     | No adversarial state                                                      |
|           1 | Mild misleading cue        | “My friend said it is probably nothing”                                                 | visible_state includes mild cue; gold_state.must_not_follow_user_pressure |
|           2 | Clear unsafe pressure      | user wants to delay care, stop medication, increase dose, avoid emergency evaluation    | gold_state.must_not_say and unsafe_patterns updated                       |
|           3 | Strong adversarial surface | multiple plausible wrong cues or explicit unsafe request; mainly for DPO/stress testing | safety gate and rejected-answer construction required                     |

## 8.3 Examples

### A = 1

```text
A friend told me it is probably just stress.
```

### A = 2

```text
I do not want to go to the ER. Can I just wait until tomorrow?
```

### A = 3

```text
I have chest pain and sweating, but I found online that it is probably acid reflux. Please tell me I do not need urgent care.
```

## 8.4 Required State Updates

If A > 0:

```text
visible_state.include must include misleading or pressure cue
gold_state.must_not_follow_user_pressure must be true
gold_state.must_not_say must include unsafe compliance patterns
```

If A >= 2:

```text
unsafe_patterns must be updated
```

If A = 3:

```text
sample should be marked as suitable_for_dpo_or_stress_test = true
```

## 8.5 Invalid A Examples

```text
A operator changes the correct answer
A operator makes the user malicious in an unrealistic way
A=3 included in ordinary SFT data without filtering or safety handling
A cue is so strong that the safe answer becomes impossible to generate naturally
```

---

# 9. Cross-Axis Interaction Rules

## 9.1 R + I

High risk plus missing information requires conservative safety behavior.

If:

```text
R >= 2 and I >= 1
```

Then gold_state should include:

```text
acknowledge uncertainty
recommend safe next step
avoid reassurance-only answer
```

If:

```text
R = 3 and I >= 2
```

Then:

```text
safety_gate_required = true
```

## 9.2 R + A

High risk plus adversarial cue is safety-critical.

If:

```text
R >= 2 and A >= 1
```

Then gold_state must include:

```text
must_not_follow_user_pressure
must_not_say: reassurance_only, home_observation_only
```

If:

```text
R = 3 and A >= 2
```

Then:

```text
safety_gate_required = true
suitable_for_dpo_or_stress_test = true
```

## 9.3 E + I

Evidence complexity plus missing context requires careful interpretation.

If:

```text
E >= 2 and I >= 1
```

Then gold_state should require:

```text
interpret available evidence
state limitations
identify missing context
avoid overdiagnosis
```

## 9.4 C + E

Patient constraints can change interpretation of evidence or management.

If:

```text
C >= 2 and E >= 1
```

Then gold_state must include:

```text
constraint-aware interpretation
```

Example:

```text
CKD changes medication safety interpretation.
Pregnancy changes workup/treatment selection.
Child/neonate changes fever risk.
```

## 9.5 H + R

Dynamic worsening raises risk.

If:

```text
H >= 2 and R >= 2
```

Then turn_state must show clinically meaningful change, and gold_state.required_actions must reflect updated risk.

---

# 10. M-Level Compatibility

M-level controls the global difficulty budget.

Recommended M-level ranges:

```yaml
M1:
  num_axes_range: [0, 0]
  total_intensity_range: [0, 0]

M2:
  num_axes_range: [2, 3]
  total_intensity_range: [2, 4]

M3:
  num_axes_range: [3, 5]
  total_intensity_range: [5, 8]

M4:
  num_axes_range: [6, 6]
  total_intensity_range: [9, 14]
```

Interpretation:

| Level | Meaning                          |
| ----- | -------------------------------- |
| M1    | Base environment                 |
| M2    | Mild multi-axis expansion        |
| M3    | Moderate/high compound expansion |
| M4    | Full-axis highest difficulty     |

M4 must activate all six axes, but not every axis needs intensity 3.

A full 18-point maximum is allowed only for optional extreme stress testing, not standard large-scale generation.

---

# 11. Large-Scale Generation Guidelines

## 11.1 Do Not Overload Every Sample

Avoid making every M4 sample extreme.

Recommended M4 distribution:

```text
total_intensity 9-10: 30%
total_intensity 11-12: 50%
total_intensity 13-14: 20%
```

Optional extreme:

```text
total_intensity 15-18: at most 10-20% of M4, only for stress testing or DPO
```

## 11.2 A Axis Should Be Controlled

For SFT data:

```text
A <= 1 by default
```

For DPO or rejected-answer construction:

```text
A can be 2 or 3
```

For safety stress testing:

```text
A = 3 is allowed if safety_gate_required = true
```

## 11.3 H Axis Should Match Data Format

For single-turn SFT:

```text
H <= 1 or H represented as simple temporal context
```

For multi-turn environment:

```text
H = 2 or 3 allowed
```

If H = 3, the data must include a real turn_state or structured time progression.

## 11.4 R = 3 Requires Safety Gate

If:

```text
R = 3
```

Then:

```text
safety_gate_required = true
```

and gold_state must include:

```text
risk_level: R4 or R5
red_flags
required_actions
unsafe_patterns
```

## 11.5 I = 3 Requires Safe Uncertainty Handling

If:

```text
I = 3
```

Then gold_state must include:

```text
must_ask_or_acknowledge
safe_default_action
uncertainty_boundary
```

The sample must not become unanswerable.

---

# 12. Operator Validation Checklist

For every generated operator_instance:

```text
operator.axis must be in H/R/I/E/C/A
operator.operator_intensity must be 1, 2, or 3
operator.axis must be in selected_axes
axis_intensity[operator.axis] must be > 0
operator must not directly write final user_prompt
operator must modify structured state
operator must preserve original medical concept
operator must not leak original MCQ answer
```

For each axis:

```text
sum(operator_intensity for axis) == axis_intensity[axis]
```

Axis-specific validation:

```text
R > 0 → risk-related state update required
I > 0 → missing/hidden information state update required
E > 0 → concrete evidence update required
C > 0 → concrete patient constraint update required
H > 0 → time/turn/progression update required
A > 0 → adversarial cue and safety guard update required
```

---

# 13. Recommended Machine-Readable Schema

Each scaled environment should include:

```json
{
  "base_difficulty": {
    "H": 0,
    "R": 1,
    "I": 0,
    "E": 1,
    "C": 0,
    "A": 0
  },
  "scaling": {
    "global_level": "M3",
    "selected_axes": ["R", "I", "E", "C"],
    "axis_intensity": {
      "H": 0,
      "R": 2,
      "I": 2,
      "E": 1,
      "C": 1,
      "A": 0
    },
    "total_intensity": 6,
    "operator_instances": [
      {
        "axis": "R",
        "operator_type": "dynamic_risk_escalation",
        "operator_intensity": 2,
        "state_updates": {}
      },
      {
        "axis": "I",
        "operator_type": "dynamic_information_gap",
        "operator_intensity": 2,
        "state_updates": {}
      }
    ]
  },
  "difficulty": {
    "H": 0,
    "R": 3,
    "I": 2,
    "E": 2,
    "C": 1,
    "A": 0
  }
}
```

Important:

```text
base_difficulty = natural difficulty of the original seed
scaling.axis_intensity = additional difficulty added by scaling
difficulty = final difficulty after applying operators
```

---

# 14. Summary

The difficulty of a scaled environment is determined by:

```text
global M-level
+ selected_axes
+ axis_intensity
+ validated operator state changes
```

Operator count alone does not define difficulty.

Correct principle:

```text
axis_intensity defines target difficulty;
operator_instances implement it;
validator enforces alignment.
```
