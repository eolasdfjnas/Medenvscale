from __future__ import annotations


def detect_red_flags(text: str, symptoms: list[str]) -> list[str]:
    lowered = text.lower()
    return [symptom for symptom in symptoms if symptom.lower() in lowered]
