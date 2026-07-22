from __future__ import annotations

import http.client
import json
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError

from medenvscale.llm.cache import DiskCache
from medenvscale.llm.json_repair import parse_json_payload
from medenvscale.utils import append_jsonl

_RETRYABLE_NETWORK_EXCEPTIONS = (
    URLError,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    http.client.BadStatusLine,
    http.client.CannotSendRequest,
    http.client.ResponseNotReady,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


@dataclass
class LLMResponse:
    payload: dict[str, Any]
    raw_text: str
    source: str
    cached: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | str


@dataclass
class ToolLLMResponse:
    content: str
    tool_calls: list[ToolCall]
    raw_message: dict[str, Any]
    source: str
    cached: bool = False


class LLMClient:
    def __init__(self, config: dict[str, Any], mode: str, cache_dir: str, trace_path: str | None = None) -> None:
        self.config = config
        self.mode = mode
        self.cache = DiskCache(cache_dir)
        self.trace_path = trace_path

    def complete_json(
        self,
        task_name: str,
        prompt: str,
        context: dict[str, Any],
        mock_builder,
    ) -> LLMResponse:
        key = {
            "task_name": task_name,
            "prompt": prompt,
            "mode": self.mode,
            "llm_identity": self._cache_identity(),
            "context": context,
        }
        cached = self.cache.get(key)
        if cached is not None:
            return LLMResponse(payload=cached, raw_text=json.dumps(cached, ensure_ascii=False), source=self.mode, cached=True)

        if self.mode == "mock":
            payload = mock_builder(context)
            response = LLMResponse(payload=payload, raw_text=json.dumps(payload, ensure_ascii=False), source="mock")
        elif self.mode == "api":
            payload = self._call_openai_compatible(prompt)
            response = LLMResponse(payload=payload, raw_text=json.dumps(payload, ensure_ascii=False), source="api")
        elif self.mode == "local":
            payload, raw_text = self._call_local_json(prompt)
            response = LLMResponse(payload=payload, raw_text=raw_text, source="local")
        else:
            raise ValueError(f"Unsupported llm mode: {self.mode}")

        self.cache.set(key, response.payload)
        if self.trace_path:
            append_jsonl(
                self.trace_path,
                {
                    "task_name": task_name,
                    "mode": self.mode,
                    "prompt": prompt,
                    "context": context,
                    "response": response.payload,
                    "cached": response.cached,
                },
            )
        return response

    def complete_with_tools(
        self,
        task_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        context: dict[str, Any],
        mock_builder=None,
    ) -> ToolLLMResponse:
        key = {
            "task_name": task_name,
            "messages": messages,
            "tools": tools,
            "mode": self.mode,
            "llm_identity": self._cache_identity(),
            "context": context,
        }
        cached = self.cache.get(key)
        if cached is not None:
            return _tool_response_from_message(cached, source=self.mode, cached=True, tools=tools)

        if self.mode == "mock":
            message = mock_builder(context) if mock_builder is not None else {"role": "assistant", "content": "{}"}
            source = "mock"
        elif self.mode == "local":
            message = self._call_local_with_tools(messages=messages, tools=tools)
            source = "local"
        elif self.mode == "api":
            message = self._call_openai_compatible_with_tools(messages=messages, tools=tools)
            source = "api"
        else:
            raise ValueError(f"Unsupported llm mode: {self.mode}")

        self.cache.set(key, message)
        if self.trace_path:
            append_jsonl(
                self.trace_path,
                {
                    "task_name": task_name,
                    "mode": self.mode,
                    "messages": messages,
                    "tools": tools,
                    "context": context,
                    "response": message,
                    "cached": False,
                },
            )
        return _tool_response_from_message(message, source=source, cached=False, tools=tools)

    def _cache_identity(self) -> dict[str, Any]:
        api_cfg = self.config.get("api", {}) or {}
        local_cfg = self.config.get("local", {}) or {}
        return {
            "base_url": api_cfg.get("base_url"),
            "model": api_cfg.get("model"),
            "model_path": local_cfg.get("model_path"),
            "adapter_path": local_cfg.get("adapter_path"),
            "api_key_env": api_cfg.get("api_key_env"),
            "temperature": api_cfg.get("temperature"),
            "local_temperature": local_cfg.get("temperature"),
            "top_p": local_cfg.get("top_p"),
            "max_new_tokens": local_cfg.get("max_new_tokens"),
            "response_format": api_cfg.get("response_format"),
        }

    def _call_local_json(self, prompt: str) -> tuple[dict[str, Any], str]:
        local_prompt = (
            "Return exactly one valid JSON object. Do not include Markdown fences or prose.\n\n"
            + str(prompt or "")
        )
        text = self._generate_local_text(local_prompt)
        return parse_json_payload(text), text

    def _call_local_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = _local_tool_prompt(messages=messages, tools=tools)
        text = self._generate_local_text(prompt)
        try:
            payload = parse_json_payload(text)
        except json.JSONDecodeError as exc:
            payload = _repair_local_tool_payload(text, tools)
            if payload is None:
                raise RuntimeError(
                    "Local model returned non-JSON content for a tool-agent turn. "
                    f"model_path={self.config.get('local', {}).get('model_path')}. Raw content: {text[:500]}"
                ) from exc
        if not isinstance(payload, dict):
            payload = {"content": json.dumps(payload, ensure_ascii=False)}
        tool_calls = _local_payload_to_tool_calls(payload, tools)
        if tool_calls:
            return {"role": "assistant", "content": str(payload.get("content") or ""), "tool_calls": tool_calls}
        content = payload.get("content")
        if isinstance(content, str) and content.strip().startswith(("{", "[")):
            try:
                nested_payload = parse_json_payload(content)
            except json.JSONDecodeError:
                nested_payload = None
            if isinstance(nested_payload, dict):
                nested_tool_calls = _local_payload_to_tool_calls(nested_payload, tools)
                if nested_tool_calls:
                    return {
                        "role": "assistant",
                        "content": str(nested_payload.get("content") or ""),
                        "tool_calls": nested_tool_calls,
                    }
        if not isinstance(content, str):
            content = json.dumps(payload, ensure_ascii=False)
        return {"role": "assistant", "content": content}

    def _generate_local_text(self, prompt: str) -> str:
        tokenizer, model = self._load_local_model()
        local_cfg = self.config.get("local", {}) or {}
        max_new_tokens = int(local_cfg.get("max_new_tokens", 2048))
        temperature = float(local_cfg.get("temperature", self.config.get("api", {}).get("temperature", 0.2)))
        top_p = float(local_cfg.get("top_p", 1.0))
        do_sample = bool(local_cfg.get("do_sample", temperature > 0))
        inputs = tokenizer(str(prompt or ""), return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample and top_p < 1.0 else None,
            "pad_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}
        output = model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated = output[0][input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _load_local_model(self):
        if hasattr(self, "_local_model_pair"):
            return self._local_model_pair
        model_path = self._local_model_path()
        if not model_path:
            raise RuntimeError("Local LLM mode requires --model_path or local.model_path in configs/agent_llm.yaml.")
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local LLM mode requires transformers and torch. Install them in the active environment "
                "or run Stage06 with --llm_mode api."
            ) from exc
        local_cfg = self.config.get("local", {}) or {}
        trust_remote_code = bool(local_cfg.get("trust_remote_code", True))
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        dtype_cfg = str(local_cfg.get("torch_dtype", "auto"))
        torch_dtype = "auto"
        if dtype_cfg not in {"", "auto"}:
            torch_dtype = getattr(torch, dtype_cfg)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map=local_cfg.get("device_map", "auto"),
            torch_dtype=torch_dtype,
        )
        adapter_path = str(local_cfg.get("adapter_path") or "").strip()
        if adapter_path:
            try:
                from peft import PeftModel  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError("Local adapter inference requires peft. Install peft or remove local.adapter_path.") from exc
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        self._local_model_pair = (tokenizer, model)
        return self._local_model_pair

    def _local_model_path(self) -> str:
        return str(((self.config.get("local", {}) or {}).get("model_path") or "")).strip()

    def _call_openai_compatible(self, prompt: str) -> dict[str, Any]:
        import urllib.parse
        import urllib.request

        api_cfg = self.config["api"]
        response_format_cfg = str(api_cfg.get("response_format", "json")).strip().lower()
        base_prompt = prompt
        response_format = None
        if response_format_cfg == "json":
            response_format = {"type": "json_object"}
            if "json" not in prompt.lower():
                base_prompt = "Return a valid JSON object.\n\n" + prompt
        api_key = os.getenv(api_cfg["api_key_env"], "")
        if not api_key.strip():
            raise RuntimeError(
                f"Missing API key. Set the environment variable {api_cfg['api_key_env']} before running "
                f"the pipeline against {api_cfg['base_url']}."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        total_attempts = max(1, int(api_cfg.get("max_retries", 3)) + 1)
        json_retry_attempts = max(0, int(api_cfg.get("json_retry_attempts", 1)))
        timeout_seconds = api_cfg.get("timeout_seconds", 60)
        last_error: Exception | None = None
        parse_error: json.JSONDecodeError | None = None
        raw_content = ""

        for json_attempt in range(json_retry_attempts + 1):
            request_prompt = base_prompt
            if json_attempt > 0:
                request_prompt = (
                    "Your previous response was not valid JSON. "
                    "Return only one valid JSON object that exactly matches the requested schema.\n\n"
                    + base_prompt
                )

            body = {
                "model": api_cfg["model"],
                "messages": [{"role": "user", "content": request_prompt}],
                "temperature": api_cfg.get("temperature", 0.2),
            }
            if response_format is not None:
                body["response_format"] = response_format
            request = urllib.request.Request(
                f"{api_cfg['base_url'].rstrip('/')}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            data: dict[str, Any] | None = None
            for attempt in range(1, total_attempts + 1):
                try:
                    with _open_with_post_redirects(
                        request,
                        timeout_seconds=timeout_seconds,
                        max_redirects=int(api_cfg.get("max_redirects", 3)),
                        urljoin=urllib.parse.urljoin,
                        urlopen=urllib.request.urlopen,
                        request_cls=urllib.request.Request,
                    ) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    break
                except HTTPError as exc:
                    error_body = ""
                    if exc.fp is not None:
                        try:
                            error_body = exc.fp.read().decode("utf-8", errors="replace")
                        except Exception:
                            error_body = ""
                    if exc.code == 401:
                        raise RuntimeError(
                            f"DeepSeek authentication failed with HTTP 401. Check {api_cfg['api_key_env']} and make sure "
                            f"it is a valid key for {api_cfg['base_url']}."
                        ) from exc
                    if exc.code == 400:
                        message = (
                            f"LLM request failed with HTTP 400 from {api_cfg['base_url']}. "
                            f"model={api_cfg['model']}. Response body: {error_body or '<empty>'}"
                        )
                        raise RuntimeError(message) from exc
                    if exc.code == 429 or exc.code >= 500:
                        last_error = RuntimeError(
                            f"LLM transient HTTP error {exc.code} from {api_cfg['base_url']}. "
                            f"model={api_cfg['model']}. Response body: {error_body or '<empty>'}"
                        )
                        if attempt < total_attempts:
                            time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
                            continue
                        raise last_error from exc
                    raise
                except _RETRYABLE_NETWORK_EXCEPTIONS as exc:
                    last_error = RuntimeError(
                        f"LLM network error from {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. attempt={attempt}/{total_attempts}. "
                        f"Underlying error: {_network_error_detail(exc)}"
                    )
                    if attempt < total_attempts:
                        time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
                        continue
                    raise last_error from exc

            if data is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("LLM request failed before any response payload was received.")

            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            raw_content = content
            try:
                return parse_json_payload(content)
            except json.JSONDecodeError as exc:
                parse_error = exc
                if json_attempt < json_retry_attempts:
                    time.sleep(min(2.0, 0.25 * (json_attempt + 1)))
                    continue
                break

        snippet = raw_content[:1000]
        raise RuntimeError(
            "LLM returned non-JSON content that could not be repaired. "
            f"model={api_cfg['model']}. Raw content: {snippet}"
        ) from parse_error

    def _call_openai_compatible_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import urllib.parse
        import urllib.request

        api_cfg = self.config["api"]
        api_key = os.getenv(api_cfg["api_key_env"], "")
        if not api_key.strip():
            raise RuntimeError(
                f"Missing API key. Set the environment variable {api_cfg['api_key_env']} before running "
                f"the tool-agent against {api_cfg['base_url']}."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": api_cfg["model"],
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": api_cfg.get("temperature", 0.2),
        }
        request = urllib.request.Request(
            f"{api_cfg['base_url'].rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        total_attempts = max(1, int(api_cfg.get("max_retries", 3)) + 1)
        timeout_seconds = api_cfg.get("timeout_seconds", 60)
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                with _open_with_post_redirects(
                    request,
                    timeout_seconds=timeout_seconds,
                    max_redirects=int(api_cfg.get("max_redirects", 3)),
                    urljoin=urllib.parse.urljoin,
                    urlopen=urllib.request.urlopen,
                    request_cls=urllib.request.Request,
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]
            except HTTPError as exc:
                error_body = ""
                if exc.fp is not None:
                    try:
                        error_body = exc.fp.read().decode("utf-8", errors="replace")
                    except Exception:
                        error_body = ""
                if _is_model_not_found_error(error_body):
                    raise RuntimeError(
                        f"Agent LLM model was not found by {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. Check configs/agent_llm.yaml and use the exact model id "
                        "configured by the provider or gateway. Response body: "
                        f"{error_body or '<empty>'}"
                    ) from exc
                if exc.code in {401, 403}:
                    raise RuntimeError(
                        f"LLM tool request was rejected with HTTP {exc.code} from {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. Check {api_cfg['api_key_env']}, account/model access, "
                        f"provider credits, and tool-calling support. Response body: {error_body or '<empty>'}"
                    ) from exc
                if exc.code == 400:
                    raise RuntimeError(
                        f"LLM tool request failed with HTTP 400 from {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. Response body: {error_body or '<empty>'}"
                    ) from exc
                if exc.code == 429 or exc.code >= 500:
                    last_error = RuntimeError(
                        f"LLM transient HTTP error {exc.code} from {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. Response body: {error_body or '<empty>'}"
                    )
                    if attempt < total_attempts:
                        time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
                        continue
                    raise last_error from exc
                raise
            except _RETRYABLE_NETWORK_EXCEPTIONS as exc:
                last_error = RuntimeError(
                    f"LLM network error from {api_cfg['base_url']}. "
                    f"model={api_cfg['model']}. attempt={attempt}/{total_attempts}. "
                    f"Underlying error: {_network_error_detail(exc)}"
                )
                if attempt < total_attempts:
                    time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
                    continue
                raise last_error from exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM tool request failed before any response payload was received.")


def _tool_response_from_message(
    message: dict[str, Any],
    source: str,
    cached: bool,
    tools: list[dict[str, Any]] | None = None,
) -> ToolLLMResponse:
    calls = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{len(calls) + 1}"),
                name=str(function.get("name") or ""),
                arguments=function.get("arguments") or {},
            )
        )
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    if not calls and source == "local" and content.strip().startswith(("{", "[")):
        try:
            payload = parse_json_payload(content)
        except json.JSONDecodeError:
            payload = _repair_local_tool_payload(content, tools or [])
        if isinstance(payload, dict):
            for item in _local_payload_to_tool_calls(payload, tools or []):
                function = item.get("function") or {}
                calls.append(
                    ToolCall(
                        id=str(item.get("id") or f"call_{len(calls) + 1}"),
                        name=str(function.get("name") or ""),
                        arguments=function.get("arguments") or {},
                    )
                )
    return ToolLLMResponse(content=content, tool_calls=calls, raw_message=message, source=source, cached=cached)


def _local_tool_prompt(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    compact_tools = []
    for item in tools:
        function = item.get("function") or {}
        compact_tools.append(
            {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            }
        )
    rendered_messages = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        name = message.get("name")
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        prefix = f"{role} {name}:" if name else f"{role}:"
        rendered_messages.append(f"{prefix}\n{content}")
    return (
        "You are a tool-using coding agent. You must respond with exactly one valid JSON object and no Markdown.\n"
        "Available tools are described below. To call tools, return:\n"
        '{"tool_calls":[{"name":"tool_name","arguments":{...}}],"content":""}\n'
        "To finish without a tool call, return:\n"
        '{"final_code":"<complete executable Python code>","notes":["..."]}\n'
        "Do not invent tool names. Do not ask for hidden oracle cases.\n\n"
        f"AVAILABLE_TOOLS:\n{json.dumps(compact_tools, ensure_ascii=False)}\n\n"
        "CONVERSATION:\n"
        + "\n\n".join(rendered_messages)
        + "\n\nReturn exactly one JSON object now."
    )


def _local_payload_to_tool_calls(payload: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    raw_calls_value = payload.get("tool_calls")
    raw_calls = raw_calls_value if isinstance(raw_calls_value, list) else []
    if raw_calls_value is None and payload.get("name"):
        raw_calls = [payload]
    elif raw_calls_value is not None and not isinstance(raw_calls_value, list) and payload.get("name"):
        raw_calls = [payload]
    if not raw_calls:
        inferred = _infer_local_tool_call(payload, tools or [])
        if inferred is not None:
            raw_calls = [inferred]
    calls = []
    for index, item in enumerate(raw_calls, start=1):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = str(item.get("name") or function.get("name") or "")
        if not name:
            continue
        arguments = item.get("arguments", function.get("arguments", {}))
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            arguments_text = json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False)
        calls.append(
            {
                "id": str(item.get("id") or f"local_call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_text,
                },
            }
        )
    return calls


def _infer_local_tool_call(payload: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not tools:
        return None
    payload_keys = {str(key) for key in payload}
    if not payload_keys:
        return None
    if "final_code" in payload or "content" in payload or "tool_calls" in payload:
        return None
    if payload_keys == {"window"} and _has_tool(tools, "get_task_context"):
        return {"name": "get_task_context", "arguments": {"window": payload.get("window")}}
    if payload_keys == {"code"} and _has_tool(tools, "submit_final_code"):
        return {"name": "submit_final_code", "arguments": {"code": payload.get("code")}}
    return None


def _repair_local_tool_payload(text: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if "tool_calls" not in raw:
        return None
    name = _extract_repair_tool_name(raw)
    if name not in {"submit_final_code", "validate_candidate_code"}:
        return None
    if tools and not _has_tool(tools, name):
        return None
    code = _extract_repair_code_argument(raw)
    if code is None:
        return None
    return {
        "tool_calls": [
            {
                "name": name,
                "arguments": {"code": code},
            }
        ],
        "content": "",
    }


def _extract_repair_tool_name(text: str) -> str | None:
    match = re.search(r'"name"\s*:\s*"(submit_final_code|validate_candidate_code)"', text)
    return match.group(1) if match else None


def _extract_repair_code_argument(text: str) -> str | None:
    return _extract_repair_string_argument(text, "code")


def _extract_repair_string_argument(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not match:
        return None
    start = match.end()
    end = _find_relaxed_json_string_end(text, start)
    fragment = text[start:end] if end is not None else text[start:]
    return _decode_relaxed_json_fragment(fragment).strip()


def _find_relaxed_json_string_end(text: str, start: int) -> int | None:
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char != '"':
            continue
        suffix = text[index + 1 :]
        if not suffix.strip() or _looks_like_repaired_code_suffix(suffix):
            return index
    return None


def _looks_like_repaired_code_suffix(suffix: str) -> bool:
    return bool(
        re.match(r"\s*}\s*}\s*]\s*(,\s*\"content\"\s*:|})", suffix)
        or re.match(r"\s*}\s*]\s*(,\s*\"content\"\s*:|})", suffix)
    )


def _decode_relaxed_json_fragment(fragment: str) -> str:
    return (
        fragment.replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
    )


def _has_tool(tools: list[dict[str, Any]], name: str) -> bool:
    for item in tools:
        function = item.get("function") if isinstance(item, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False


def _is_model_not_found_error(error_body: str) -> bool:
    lowered = str(error_body or "").lower()
    return "model not found" in lowered or '"code":"10404"' in lowered or '"code":10404' in lowered


def _network_error_detail(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    if reason:
        return f"{type(exc).__name__}: {reason}"
    return f"{type(exc).__name__}: {exc}"


def _open_with_post_redirects(
    request: Any,
    *,
    timeout_seconds: int | float,
    max_redirects: int,
    urljoin: Any,
    urlopen: Any,
    request_cls: Any,
) -> Any:
    current = request
    for _ in range(max_redirects + 1):
        try:
            return urlopen(current, timeout=timeout_seconds)
        except HTTPError as exc:
            if exc.code not in {307, 308}:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            current = request_cls(
                urljoin(current.full_url, location),
                data=current.data,
                headers=dict(current.header_items()),
                method=current.get_method(),
            )
    raise RuntimeError(f"LLM request exceeded {max_redirects} HTTP redirects.")
