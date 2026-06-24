from __future__ import annotations

from medenvscale.classify.taxonomy import normalize_domain_name

def normalize_domain(domain: str, allowed_domains: list[str]) -> str:
    normalized = normalize_domain_name(domain)
    if normalized in allowed_domains:
        return normalized
    return "general_medicine"
