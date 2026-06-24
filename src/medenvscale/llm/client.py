from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError

from medenvscale.llm.cache import DiskCache
from medenvscale.llm.json_repair import parse_json_payload
from medenvscale.utils import append_jsonl


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
            payload = mock_builder(context)
            response = LLMResponse(payload=payload, raw_text=json.dumps(payload, ensure_ascii=False), source="local_fallback")
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
            return _tool_response_from_message(cached, source=self.mode, cached=True)

        if self.mode in {"mock", "local"}:
            message = mock_builder(context) if mock_builder is not None else {"role": "assistant", "content": "{}"}
            source = "mock" if self.mode == "mock" else "local_fallback"
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
        return _tool_response_from_message(message, source=source, cached=False)

    def _cache_identity(self) -> dict[str, Any]:
        api_cfg = self.config.get("api", {}) or {}
        return {
            "base_url": api_cfg.get("base_url"),
            "model": api_cfg.get("model"),
            "api_key_env": api_cfg.get("api_key_env"),
            "temperature": api_cfg.get("temperature"),
            "response_format": api_cfg.get("response_format"),
        }

    def _call_openai_compatible(self, prompt: str) -> dict[str, Any]:
        import urllib.parse
        import urllib.request
        from urllib.error import HTTPError, URLError

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
                except URLError as exc:
                    last_error = RuntimeError(
                        f"LLM network error from {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. attempt={attempt}/{total_attempts}. "
                        f"Underlying error: {exc.reason}"
                    )
                    if attempt < total_attempts:
                        time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
                        continue
                    raise last_error from exc
                except TimeoutError as exc:
                    last_error = RuntimeError(
                        f"LLM request timed out against {api_cfg['base_url']}. "
                        f"model={api_cfg['model']}. attempt={attempt}/{total_attempts}."
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
        from urllib.error import HTTPError, URLError

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
            except (URLError, TimeoutError) as exc:
                last_error = RuntimeError(
                    f"LLM network error from {api_cfg['base_url']}. "
                    f"model={api_cfg['model']}. attempt={attempt}/{total_attempts}. "
                    f"Underlying error: {exc}"
                )
                if attempt < total_attempts:
                    time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
                    continue
                raise last_error from exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM tool request failed before any response payload was received.")


def _tool_response_from_message(message: dict[str, Any], source: str, cached: bool) -> ToolLLMResponse:
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
    return ToolLLMResponse(content=content, tool_calls=calls, raw_message=message, source=source, cached=cached)


def _is_model_not_found_error(error_body: str) -> bool:
    lowered = str(error_body or "").lower()
    return "model not found" in lowered or '"code":"10404"' in lowered or '"code":10404' in lowered


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
