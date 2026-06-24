from __future__ import annotations

from typing import Any


DEFAULT_STAGE06_TOOL_NAMES = {
    "get_task_context",
    "read_resource_file",
    "validate_candidate_code",
    "run_custom_test",
    "submit_final_code",
}


def stage06_tool_schemas(tool_pool_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    tools = (tool_pool_cfg or {}).get("agent_tools") or []
    schemas = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("tool_name") or "")
        if name not in DEFAULT_STAGE06_TOOL_NAMES:
            continue
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("openai_parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    if schemas:
        return schemas
    return _fallback_stage06_tool_schemas()


def stage06_tool_names(tool_pool_cfg: dict[str, Any] | None = None) -> set[str]:
    return {item["function"]["name"] for item in stage06_tool_schemas(tool_pool_cfg)}


def _fallback_stage06_tool_schemas() -> list[dict[str, Any]]:
    return [
        _tool_schema(
            "get_task_context",
            "Read public task context, scaffold, signature, resource manifest, and public requirements.",
            {"type": "object", "properties": {"window": {"type": "integer", "default": 4000}}},
        ),
        _tool_schema(
            "read_resource_file",
            "Read a bounded slice of a public resource file listed in the resource manifest.",
            {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 0},
                    "max_bytes": {"type": "integer", "default": 4000},
                },
            },
        ),
        _tool_schema(
            "validate_candidate_code",
            "Check whether a complete Python program compiles and preserves the target signature.",
            {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
        ),
        _tool_schema(
            "run_custom_test",
            "Run a self-authored public test snippet against the complete candidate program.",
            {
                "type": "object",
                "required": ["code", "test_snippet"],
                "properties": {
                    "code": {"type": "string"},
                    "test_snippet": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 5},
                },
            },
        ),
        _tool_schema(
            "submit_final_code",
            "Submit the final complete Python program. This ends the episode and triggers hidden oracle evaluation.",
            {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
        ),
    ]


def _tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
