from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.hidden_test_runner import insert_solution
from medenvscale.utils import stable_hash


def materialize_semantic_test_specs(
    specs: list[dict[str, Any]],
    task_context: dict[str, Any],
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
    config: dict | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    materialized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        result = materialize_semantic_test_spec(spec, task_context, llm_client=llm_client, prompt_runner=prompt_runner, config=config or {})
        if result.get("materialization_status") == "success":
            materialized.append(result)
        else:
            failures.append(result)
    return materialized, failures


def materialize_semantic_test_spec(
    spec: dict[str, Any],
    task_context: dict[str, Any],
    llm_client: LLMClient | None = None,
    prompt_runner: PromptRunner | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    signature = str(task_context.get("signature") or "")
    context = str(task_context.get("context") or "")
    solution = str(task_context.get("scaled_gold_solution") or task_context.get("gold_solution") or "")
    placeholder = str(task_context.get("placeholder_token") or "<<insert solution here>>")
    function_name, param_names = _parse_signature(signature)
    if not function_name:
        return _failure(spec, "MATERIALIZATION_FAILED", "Unable to parse function signature.")
    oracle_bundle = _build_oracle_bundle(context=context, solution=solution, placeholder=placeholder, function_name=function_name)
    if not oracle_bundle["success"]:
        return _failure(spec, "MATERIALIZATION_FAILED", oracle_bundle["error"])

    invocation = _build_invocation_inputs(
        spec=spec,
        task_context=task_context,
        param_names=param_names,
        function_name=function_name,
    )
    if not invocation["success"] and llm_client is not None:
        invocation = _llm_fallback_invocation(
            spec=spec,
            task_context=task_context,
            param_names=param_names,
            function_name=function_name,
            llm_client=llm_client,
        )
    if not invocation["success"]:
        return _failure(spec, "MATERIALIZATION_FAILED", invocation["error"])

    try:
        expected_value = oracle_bundle["callable"](*invocation["args"], **invocation["kwargs"])
    except Exception as exc:
        return _failure(spec, "MATERIALIZATION_FAILED", f"oracle execution failed: {exc.__class__.__name__}: {exc}")

    normalized_expected = _normalize_runtime_value(expected_value)
    setup_code = invocation.get("setup_code", "")
    call_expr = invocation["call_expr"]
    code = _build_hidden_test_code(
        test_id=str(spec.get("spec_id") or f"test_{stable_hash(spec)[:8]}"),
        function_name=function_name,
        setup_code=setup_code,
        call_expr=call_expr,
        normalized_expected=normalized_expected,
    )
    try:
        compile(code, f"{spec.get('spec_id', 'semantic_test')}.py", "exec")
    except SyntaxError as exc:
        return _failure(spec, "MATERIALIZATION_FAILED", f"compile failed: {exc.msg}")

    result = dict(spec)
    result.update(
        {
            "test_id": str(spec.get("spec_id") or f"test_{stable_hash(spec)[:8]}"),
            "targets_operator_id": str(spec.get("targets_operator_id") or ""),
            "axis": str(spec.get("axis") or ""),
            "semantic_intent": str(spec.get("semantic_intent") or ""),
            "target_constraint": str(spec.get("target_constraint") or ""),
            "expected_failure_mode": str(spec.get("expected_failure_mode") or ""),
            "code": code,
            "assertion_code": code,
            "is_semantic": True,
            "is_placeholder": False,
            "test_tier": "semantic",
            "counts_as_hidden_test": True,
            "eligible_for_clean_export": True,
            "materialization_status": "success",
        }
    )
    return result


def _build_oracle_bundle(context: str, solution: str, placeholder: str, function_name: str) -> dict[str, Any]:
    materialized = insert_solution(context=context, solution=solution, placeholder=placeholder)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    try:
        exec(materialized, namespace, namespace)
    except Exception as exc:
        return {"success": False, "error": f"context execution failed: {exc.__class__.__name__}: {exc}"}
    callable_obj = namespace.get(function_name)
    if not callable(callable_obj):
        return {"success": False, "error": f"callable {function_name} not found after materialization"}
    return {"success": True, "callable": callable_obj, "namespace": namespace}


def _build_invocation_inputs(
    spec: dict[str, Any],
    task_context: dict[str, Any],
    param_names: list[str],
    function_name: str,
) -> dict[str, Any]:
    setup_lines: list[str] = []
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    arg_exprs: list[str] = []
    axis = str(spec.get("axis") or "")
    solution_form = str(task_context.get("solution_form") or "")
    if solution_form == "expression_completion":
        for name in param_names:
            value = _default_value_for_param(name, axis=axis)
            args.append(value)
            arg_exprs.append(repr(value))
        return {"success": True, "args": args, "kwargs": kwargs, "setup_code": "", "call_expr": f"{function_name}({', '.join(arg_exprs)})"}

    file_payload = None
    input_variant = spec.get("input_variant") if isinstance(spec.get("input_variant"), dict) else {}
    for name in param_names:
        value, expr, extra_setup = _value_and_expr_for_param(
            name=name,
            axis=axis,
            task_context=task_context,
            input_variant=input_variant,
        )
        if extra_setup:
            setup_lines.extend(extra_setup)
        if name in {"path", "file_path", "filepath"} and isinstance(value, dict) and value.get("kind") == "temp_csv":
            file_payload = value
            expr = value["var_name"]
            args.append(value["materialized_path"])
            arg_exprs.append(expr)
            continue
        args.append(value)
        arg_exprs.append(expr)
    call_expr = f"{function_name}({', '.join(arg_exprs)})"
    return {
        "success": True,
        "args": args,
        "kwargs": kwargs,
        "setup_code": "\n".join(setup_lines),
        "call_expr": call_expr,
        "file_payload": file_payload,
    }


def _value_and_expr_for_param(
    name: str,
    axis: str,
    task_context: dict[str, Any],
    input_variant: dict[str, Any] | None = None,
) -> tuple[Any, str, list[str]]:
    lowered = name.lower()
    if lowered in {"path", "file_path", "filepath"}:
        rows = _csv_rows_for_task(task_context, axis=axis)
        fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="medenvscale-semantic-")
        os.close(fd)
        Path(tmp_path).write_text(_csv_text(rows), encoding="utf-8")
        setup = [
            "import tempfile",
            "from pathlib import Path",
            f"_semantic_rows = {repr(rows)}",
            "with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8') as _semantic_tmp:",
            "    _semantic_tmp.write('\\\\n'.join(','.join(str(cell) for cell in row) for row in _semantic_rows))",
            "    _semantic_path = _semantic_tmp.name",
        ]
        return {"kind": "temp_csv", "materialized_path": tmp_path, "var_name": "_semantic_path"}, "_semantic_path", setup
    value = _variant_value_for_param(name=name, axis=axis, input_variant=input_variant) 
    if value is _MISSING:
        value = _default_value_for_param(name, axis=axis)
    return value, repr(value), []


def _default_value_for_param(name: str, axis: str) -> Any:
    lowered = name.lower()
    if lowered in {"records", "record_dict"}:
        return {"seq1": "ATTA", "seq2": "TA"}
    if lowered == "motif":
        return "TA"
    if lowered in {"value"}:
        return 0 if axis == "C" else 7
    if lowered in {"mean"}:
        return 5
    if lowered in {"std"}:
        return 0 if axis == "C" else 2
    if lowered.endswith("s") or lowered in {"items", "values", "xs", "nums", "numbers"}:
        return [] if axis == "C" else [1, 2, 3]
    if "record" in lowered:
        return {"id": 1, "value": 3}
    if "unit" in lowered:
        return "mg"
    if "dose" in lowered or "dosage" in lowered:
        return 5
    if "source" in lowered:
        return {"source_a": [{"id": 1, "value": 2}], "source_b": [{"id": 2, "value": 4}]}
    if lowered in {"x", "y", "z", "n", "count"}:
        return 1 if axis == "A" else 4
    return 1


_MISSING = object()


def _variant_value_for_param(name: str, axis: str, input_variant: dict[str, Any] | None) -> Any:
    if not input_variant:
        return _MISSING
    lowered = name.lower()
    kind = str(input_variant.get("kind") or "").lower()
    value = input_variant.get("value")

    if kind == "empty_input":
        if lowered.endswith("s") or lowered in {"items", "values", "xs", "nums", "numbers"}:
            return []
        if lowered in {"value", "count", "n"}:
            return 0
    if kind == "boundary_case":
        if lowered in {"x", "y", "z", "n", "count", "value"}:
            return -1 if axis in {"C", "V"} else 9
    if kind == "anti_shortcut_variant":
        if lowered in {"x", "y", "z", "n", "count", "value"}:
            return 17
        if lowered.endswith("s") or lowered in {"items", "values", "xs", "nums", "numbers"}:
            return [3, 7, 11]
    if kind == "implicit_default_case":
        if "unit" in lowered:
            return None
        if lowered in {"value", "dose", "dosage"}:
            return 5
    if kind == "risk_invalid_unit_case":
        if "unit" in lowered:
            return "kg"
        if "dose" in lowered or "dosage" in lowered:
            return 5000
    if kind == "resource_nested_case":
        if lowered in {"records", "sources"} or "source" in lowered:
            return {
                "source_a": [{"id": 1, "value": 2}, {"id": 2, "value": 5}],
                "source_b": [{"id": 3, "value": 7}],
            }
    if kind == "step_dependency_case":
        if lowered in {"records", "items", "values"}:
            return [{"raw": "A "}, {"raw": " a"}]
    if isinstance(value, dict) and lowered in value:
        return value[lowered]
    return _MISSING


def _csv_rows_for_task(task_context: dict[str, Any], axis: str) -> list[list[Any]]:
    problem = str(task_context.get("problem") or "").lower()
    if "urgent" in problem:
        return [
            ["patient_id", "urgent_review", "service"],
            ["1001", "True", "cardiology"],
            ["1002", "False", "oncology"],
            ["1003", "True", "cardiology"],
        ]
    return [
        ["value", "flag"],
        ["1", "True"],
        ["2", "False"],
        ["3", "True"],
    ]


def _csv_text(rows: list[list[Any]]) -> str:
    return "\n".join(",".join(str(cell) for cell in row) for row in rows)


def _build_hidden_test_code(
    test_id: str,
    function_name: str,
    setup_code: str,
    call_expr: str,
    normalized_expected: Any,
) -> str:
    setup_prefix = f"{setup_code}\n" if setup_code else ""
    return (
        "def _semantic_normalize(value):\n"
        "    if hasattr(value, 'to_dict'):\n"
        "        try:\n"
        "            return value.to_dict(orient='records')\n"
        "        except TypeError:\n"
        "            return value.to_dict()\n"
        "    if isinstance(value, tuple):\n"
        "        return [_semantic_normalize(item) for item in value]\n"
        "    if isinstance(value, list):\n"
        "        return [_semantic_normalize(item) for item in value]\n"
        "    if isinstance(value, dict):\n"
        "        return {key: _semantic_normalize(val) for key, val in value.items()}\n"
        "    return value\n\n"
        f"{setup_prefix}"
        f"_semantic_result = {call_expr}\n"
        f"_semantic_expected = {repr(normalized_expected)}\n"
        "assert _semantic_normalize(_semantic_result) == _semantic_expected\n"
    )


def _normalize_runtime_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    if isinstance(value, tuple):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_runtime_value(val) for key, val in value.items()}
    return value


def _parse_signature(signature: str) -> tuple[str | None, list[str]]:
    signature = signature.strip()
    if not signature:
        return None, []
    try:
        parsed = ast.parse(f"{signature}\n    pass\n")
    except SyntaxError:
        return _parse_signature_fallback(signature)
    fn = parsed.body[0]
    if not isinstance(fn, ast.FunctionDef):
        return _parse_signature_fallback(signature)
    return fn.name, [arg.arg for arg in fn.args.args]


def _parse_signature_fallback(signature: str) -> tuple[str | None, list[str]]:
    if "(" not in signature or ")" not in signature:
        return None, []
    header = signature.split("(", 1)[0].strip()
    name = header.replace("def", "").replace("async", "").strip().rstrip(":")
    params_blob = signature.split("(", 1)[1].split(")", 1)[0]
    params = [item.split("=")[0].strip() for item in params_blob.split(",") if item.strip()]
    return name or None, [param for param in params if param != "self"]


def _llm_fallback_invocation(
    spec: dict[str, Any],
    task_context: dict[str, Any],
    param_names: list[str],
    function_name: str,
    llm_client: LLMClient,
) -> dict[str, Any]:
    prompt = (
        "Build executable Python invocation inputs for a semantic hidden test.\n"
        "Return a JSON object with keys: setup_code, args, kwargs.\n"
        "Do not include markdown.\n"
        f"Function name: {function_name}\n"
        f"Parameter names: {json.dumps(param_names, ensure_ascii=False)}\n"
        f"Problem: {task_context.get('problem') or ''}\n"
        f"Spec: {json.dumps(spec, ensure_ascii=False)}\n"
    )
    response = llm_client.complete_json(
        task_name="semantic_test_materializer_invocation",
        prompt=prompt,
        context={
            "function_name": function_name,
            "param_names": param_names,
            "problem": task_context.get("problem"),
            "spec": spec,
        },
        mock_builder=lambda ctx: {
            "setup_code": "",
            "args": [_default_value_for_param(name, axis=str(spec.get("axis") or "")) for name in ctx["param_names"]],
            "kwargs": {},
        },
    )
    payload = response.payload if isinstance(response.payload, dict) else {}
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        return {"success": False, "error": "LLM fallback did not return executable args/kwargs."}
    setup_code = str(payload.get("setup_code") or "")
    call_parts = [repr(arg) for arg in args]
    call_parts.extend(f"{key}={repr(val)}" for key, val in kwargs.items())
    call_expr = f"{function_name}({', '.join(call_parts)})"
    return {
        "success": True,
        "args": args,
        "kwargs": kwargs,
        "setup_code": setup_code,
        "call_expr": call_expr,
    }


def _failure(spec: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    result = dict(spec)
    result.update({"materialization_status": "failed", "failure_code": code, "failure_message": message})
    return result
