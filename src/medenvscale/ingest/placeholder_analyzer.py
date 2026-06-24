from __future__ import annotations

import re

from medenvscale.classify.taxonomy import normalize_solution_form_name

PLACEHOLDER_TOKEN = "<<insert solution here>>"


def detect_solution_form(problem: str, context: str, signature: str | None = None, code: str | None = None) -> str:
    text = f"{problem}\n{context}\n{signature or ''}\n{code or ''}".lower()
    placeholder_line = _placeholder_line(context)
    stripped = placeholder_line.strip()

    if any(term in text for term in ["bugfix", "fix the bug", "patch", "repair", "broken implementation"]):
        return "patch_or_bugfix"

    if re.search(r"@\w[\w\.]*\s*\n\s*" + re.escape(PLACEHOLDER_TOKEN), context):
        return "decorated_function_definition"

    if signature and signature.strip() and re.search(r":\s*\n[ \t]*" + re.escape(PLACEHOLDER_TOKEN), context):
        return "function_body"

    if re.search(r"(return|=|\[|\(|,)\s*" + re.escape(PLACEHOLDER_TOKEN), context):
        return "expression_completion"

    if stripped == PLACEHOLDER_TOKEN and _looks_like_function_header_nearby(context):
        return "function_body" if signature else "statement_block_completion"

    if stripped == PLACEHOLDER_TOKEN and signature and signature.strip():
        return "function_definition"

    if stripped == PLACEHOLDER_TOKEN:
        return "statement_block_completion"

    return normalize_solution_form_name(None)


def summarize_context(context: str, max_lines: int = 8, max_chars: int = 480) -> str:
    lines = [line.rstrip() for line in context.splitlines() if line.strip()]
    summary = "\n".join(lines[:max_lines]).strip()
    if len(summary) > max_chars:
        return summary[: max_chars - 3].rstrip() + "..."
    return summary


def _placeholder_line(context: str) -> str:
    for line in context.splitlines():
        if PLACEHOLDER_TOKEN in line:
            return line
    return ""


def _looks_like_function_header_nearby(context: str) -> bool:
    lines = context.splitlines()
    for index, line in enumerate(lines):
        if PLACEHOLDER_TOKEN not in line:
            continue
        window = "\n".join(lines[max(0, index - 2) : index + 1])
        return "def " in window or "class " in window
    return False
