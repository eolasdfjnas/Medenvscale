from __future__ import annotations

import http.client
import json
import os
import tempfile
import unittest
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from medenvscale.llm.client import LLMClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LLMClientTests(unittest.TestCase):
    def test_complete_json_retries_once_after_bad_json_response(self) -> None:
        client = LLMClient(
            config={
                "api": {
                    "base_url": "https://example.invalid",
                    "api_key_env": "MEDENVSCALE_TEST_API_KEY",
                    "model": "test-model",
                    "temperature": 0.0,
                    "response_format": "json",
                    "max_retries": 0,
                    "json_retry_attempts": 1,
                    "timeout_seconds": 1,
                }
            },
            mode="api",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        responses = [
            _FakeHTTPResponse({"choices": [{"message": {"content": '{"broken": "bad "quote" here"}'}}]}),
            _FakeHTTPResponse({"choices": [{"message": {"content": '{"fixed": true}'}}]}),
        ]

        with patch.dict(os.environ, {"MEDENVSCALE_TEST_API_KEY": "dummy-key"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=responses) as mocked_urlopen:
                result = client.complete_json(
                    task_name="retry_json",
                    prompt="Return JSON only.",
                    context={},
                    mock_builder=lambda _: {},
                )

        self.assertEqual(result.payload, {"fixed": True})
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_complete_json_preserves_post_on_307_redirect(self) -> None:
        client = LLMClient(
            config={
                "api": {
                    "base_url": "https://example.invalid",
                    "api_key_env": "MEDENVSCALE_TEST_API_KEY",
                    "model": "test-model",
                    "temperature": 0.0,
                    "response_format": "json",
                    "max_retries": 0,
                    "json_retry_attempts": 0,
                    "timeout_seconds": 1,
                }
            },
            mode="api",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )
        redirect = HTTPError(
            "https://example.invalid/chat/completions",
            307,
            "Temporary Redirect",
            {"Location": "https://redirected.invalid/chat/completions"},
            None,
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            if len(requests) == 1:
                raise redirect
            return _FakeHTTPResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})

        with patch.dict(os.environ, {"MEDENVSCALE_TEST_API_KEY": "dummy-key"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                result = client.complete_json(
                    task_name="redirect_json",
                    prompt="Return JSON only.",
                    context={},
                    mock_builder=lambda _: {},
                )

        self.assertEqual(result.payload, {"ok": True})
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1].full_url, "https://redirected.invalid/chat/completions")
        self.assertEqual(requests[1].get_method(), "POST")
        self.assertEqual(requests[1].data, requests[0].data)

    def test_complete_json_retries_after_remote_disconnect(self) -> None:
        client = LLMClient(
            config={
                "api": {
                    "base_url": "https://example.invalid",
                    "api_key_env": "MEDENVSCALE_TEST_API_KEY",
                    "model": "test-model",
                    "temperature": 0.0,
                    "response_format": "json",
                    "max_retries": 1,
                    "json_retry_attempts": 0,
                    "timeout_seconds": 1,
                }
            },
            mode="api",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )
        responses = [
            http.client.RemoteDisconnected("Remote end closed connection without response"),
            _FakeHTTPResponse({"choices": [{"message": {"content": '{"ok": true}'}}]}),
        ]

        with patch.dict(os.environ, {"MEDENVSCALE_TEST_API_KEY": "dummy-key"}, clear=False):
            with patch("time.sleep", return_value=None):
                with patch("urllib.request.urlopen", side_effect=responses) as mocked_urlopen:
                    result = client.complete_json(
                        task_name="remote_disconnect_json",
                        prompt="Return JSON only.",
                        context={},
                        mock_builder=lambda _: {},
                    )

        self.assertEqual(result.payload, {"ok": True})
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_complete_with_tools_reports_http_400_body(self) -> None:
        client = LLMClient(
            config={
                "api": {
                    "base_url": "https://example.invalid",
                    "api_key_env": "MEDENVSCALE_TEST_API_KEY",
                    "model": "bad_model",
                    "temperature": 0.0,
                    "max_retries": 0,
                    "timeout_seconds": 1,
                }
            },
            mode="api",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )
        error = HTTPError(
            "https://example.invalid/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":{"message":"tools unsupported by this endpoint"}}'),
        )

        with patch.dict(os.environ, {"MEDENVSCALE_TEST_API_KEY": "dummy-key"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(RuntimeError) as ctx:
                    client.complete_with_tools(
                        task_name="tool_400",
                        messages=[{"role": "user", "content": "hi"}],
                        tools=[],
                        context={},
                    )

        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("bad_model", str(ctx.exception))
        self.assertIn("tools unsupported by this endpoint", str(ctx.exception))

    def test_complete_with_tools_reports_http_403_body(self) -> None:
        client = LLMClient(
            config={
                "api": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key_env": "MEDENVSCALE_TEST_API_KEY",
                    "model": "openai/gpt-4o-mini",
                    "temperature": 0.0,
                    "max_retries": 0,
                    "timeout_seconds": 1,
                }
            },
            mode="api",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )
        error = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            403,
            "Forbidden",
            {},
            BytesIO(b'{"error":{"message":"No auth credentials found or provider access denied"}}'),
        )

        with patch.dict(os.environ, {"MEDENVSCALE_TEST_API_KEY": "dummy-key"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(RuntimeError) as ctx:
                    client.complete_with_tools(
                        task_name="tool_403",
                        messages=[{"role": "user", "content": "hi"}],
                        tools=[],
                        context={},
                    )

        self.assertIn("HTTP 403", str(ctx.exception))
        self.assertIn("openai/gpt-4o-mini", str(ctx.exception))
        self.assertIn("provider access denied", str(ctx.exception))

    def test_complete_with_tools_retries_after_remote_disconnect(self) -> None:
        client = LLMClient(
            config={
                "api": {
                    "base_url": "https://example.invalid",
                    "api_key_env": "MEDENVSCALE_TEST_API_KEY",
                    "model": "tool-model",
                    "temperature": 0.0,
                    "max_retries": 1,
                    "timeout_seconds": 1,
                }
            },
            mode="api",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )
        responses = [
            http.client.RemoteDisconnected("Remote end closed connection without response"),
            _FakeHTTPResponse({"choices": [{"message": {"role": "assistant", "content": "done"}}]}),
        ]

        with patch.dict(os.environ, {"MEDENVSCALE_TEST_API_KEY": "dummy-key"}, clear=False):
            with patch("time.sleep", return_value=None):
                with patch("urllib.request.urlopen", side_effect=responses) as mocked_urlopen:
                    result = client.complete_with_tools(
                        task_name="remote_disconnect_tools",
                        messages=[{"role": "user", "content": "hi"}],
                        tools=[],
                        context={},
                    )

        self.assertEqual(result.content, "done")
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_local_complete_with_tools_parses_tool_call_json(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                self.last_prompt = prompt
                return '{"tool_calls":[{"name":"get_task_context","arguments":{"window":1200}}],"content":""}'

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_tools",
            messages=[{"role": "user", "content": "inspect task"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_task_context",
                        "description": "Read context",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(result.source, "local")
        self.assertEqual(result.tool_calls[0].name, "get_task_context")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"window": 1200})
        self.assertIn("AVAILABLE_TOOLS", client.last_prompt)

    def test_local_complete_with_tools_preserves_final_code_json(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return '{"final_code":"result = 1","notes":["ok"]}'

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_final",
            messages=[{"role": "user", "content": "solve"}],
            tools=[],
            context={},
        )

        self.assertFalse(result.tool_calls)
        self.assertEqual(json.loads(result.content)["final_code"], "result = 1")

    def test_local_complete_with_tools_infers_context_call_from_window_args(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return '{"window":4000}'

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_shorthand_context",
            messages=[{"role": "user", "content": "inspect task"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_task_context",
                        "description": "Read context",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(result.tool_calls[0].name, "get_task_context")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"window": 4000})

    def test_local_complete_with_tools_infers_context_call_from_cached_window_args(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                raise AssertionError("cached response should skip local generation")

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )
        messages = [{"role": "user", "content": "inspect task"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_task_context",
                    "description": "Read context",
                    "parameters": {"type": "object"},
                },
            }
        ]
        client.cache.set(
            {
                "task_name": "local_cached_shorthand_context",
                "messages": messages,
                "tools": tools,
                "mode": "local",
                "llm_identity": client._cache_identity(),
                "context": {},
            },
            {"role": "assistant", "content": '{"window":4000}'},
        )

        result = client.complete_with_tools(
            task_name="local_cached_shorthand_context",
            messages=messages,
            tools=tools,
            context={},
        )

        self.assertTrue(result.cached)
        self.assertEqual(result.tool_calls[0].name, "get_task_context")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"window": 4000})

    def test_local_complete_with_tools_infers_submit_from_code_args(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return '{"code":"print(1)"}'

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_shorthand_submit",
            messages=[{"role": "user", "content": "solve"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_final_code",
                        "description": "Submit code",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(result.tool_calls[0].name, "submit_final_code")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"code": "print(1)"})

    def test_local_complete_with_tools_repairs_multiline_submit_code_argument(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return (
                    '{"tool_calls": [{"name": "submit_final_code", "arguments": {"code": "import os\n'
                    'from pathlib import Path\n'
                    'print(\\"done\\")"}}], "content": ""}'
                )

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_repair_submit",
            messages=[{"role": "user", "content": "solve"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_final_code",
                        "description": "Submit code",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(result.tool_calls[0].name, "submit_final_code")
        self.assertEqual(
            json.loads(result.tool_calls[0].arguments),
            {"code": 'import os\nfrom pathlib import Path\nprint("done")'},
        )

    def test_local_complete_with_tools_repairs_multiline_validate_code_argument(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return (
                    '{"tool_calls": [{"name": "validate_candidate_code", "arguments": {"code": "def solve():\n'
                    '    return 1'
                )

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_repair_validate",
            messages=[{"role": "user", "content": "validate"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "validate_candidate_code",
                        "description": "Validate code",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(result.tool_calls[0].name, "validate_candidate_code")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"code": "def solve():\n    return 1"})

    def test_local_repair_does_not_truncate_python_dict_literals(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return (
                    '{"tool_calls": [{"name": "submit_final_code", "arguments": {"code": "def solve():\n'
                    '    data = {\\"a\\": \\"b\\"}\n'
                    '    return data"}}], "content": ""}'
                )

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_repair_dict_literal",
            messages=[{"role": "user", "content": "solve"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submit_final_code",
                        "description": "Submit code",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(
            json.loads(result.tool_calls[0].arguments),
            {"code": 'def solve():\n    data = {"a": "b"}\n    return data'},
        )

    def test_local_complete_with_tools_accepts_top_level_named_call(self) -> None:
        class FakeLocalClient(LLMClient):
            def _generate_local_text(self, prompt: str) -> str:
                return '{"name":"get_task_context","arguments":{"window":2000}}'

        client = FakeLocalClient(
            config={"local": {"model_path": "/models/TinyLocal"}},
            mode="local",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-llm-client-"),
            trace_path=None,
        )

        result = client.complete_with_tools(
            task_name="local_top_level_named_call",
            messages=[{"role": "user", "content": "inspect"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_task_context",
                        "description": "Read context",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            context={},
        )

        self.assertEqual(result.tool_calls[0].name, "get_task_context")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"window": 2000})


if __name__ == "__main__":
    unittest.main()
