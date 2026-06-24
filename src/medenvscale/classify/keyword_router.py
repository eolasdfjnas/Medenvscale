from __future__ import annotations

DOMAIN_KEYWORDS = {
    "cardiology": [
        "chest pain",
        "angina",
        "myocardial",
        "infarction",
        "heart failure",
        "hypertension",
        "ecg",
        "troponin",
        "arrhythmia",
        "murmur",
    ],
    "pulmonology": ["dyspnea", "asthma", "copd", "pneumonia", "pulmonary", "wheezing", "crackles", "cough"],
    "infectious_disease": ["fever", "infection", "antibiotic", "sepsis", "meningitis", "hiv", "tuberculosis", "culture"],
    "obstetrics_gynecology": ["pregnant", "pregnancy", "gestational", "postpartum", "vaginal bleeding", "contraception", "preeclampsia"],
    "pediatrics": ["infant", "child", "boy", "girl", "newborn", "vaccination", "developmental delay"],
    "pharmacology": ["drug", "medication", "adverse effect", "contraindication", "mechanism of action", "toxicity", "dose"],
    "hematology_oncology": ["anemia", "leukemia", "lymphoma", "platelet", "coagulation", "cancer", "tumor"],
    "neurology": ["seizure", "stroke", "headache", "weakness", "numbness", "cranial nerve", "multiple sclerosis"],
    "endocrinology": ["diabetes", "thyroid", "adrenal", "cortisol", "insulin", "hyperglycemia", "hypoglycemia"],
    "gastroenterology": ["abdominal pain", "diarrhea", "vomiting", "liver", "hepatitis", "gi bleeding", "pancreatitis"],
    "psychiatry": ["depression", "anxiety", "psychosis", "bipolar", "suicide", "substance use"],
}


def route_keywords(text: str) -> tuple[list[str], dict[str, list[str]]]:
    text = text.lower()
    hits: dict[str, list[str]] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword in text]
        if matched:
            hits[domain] = matched
    candidates = sorted(hits, key=lambda item: (-len(hits[item]), item))
    if not candidates:
        candidates = ["general_medicine"]
    return candidates, hits
