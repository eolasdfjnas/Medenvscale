from __future__ import annotations


def medication_unsafe_patterns(context: dict) -> list[str]:
    task_type = context.get("primary_task_type") or context.get("task_type")
    if task_type == "medication_safety":
        return ["safe to take any medication", "increase the dose on your own"]
    return []
