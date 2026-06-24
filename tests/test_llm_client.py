from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
