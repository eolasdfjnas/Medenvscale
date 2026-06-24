from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medenvscale.agent.candidate_execution import run_custom_test
from medenvscale.agent.runner import _system_prompt, run_stage06_tool_agent, run_tool_agent_for_env
from medenvscale.agent.tool_runtime import ToolRuntime
from medenvscale.config import AppConfig
from medenvscale.llm.client import ToolLLMResponse
from medenvscale.schemas import ExecutableEnvSpec


class Stage06ToolRuntimeTests(unittest.TestCase):
    def _env_without_cases(self) -> ExecutableEnvSpec:
        return ExecutableEnvSpec(
            env_id="env_no_cases_M1",
            original_task_id="task_no_cases",
            split="train",
            problem="Write any code.",
            context="",
            signature="",
            solution_form="complete_program",
            primary_domain="scientific_software_engineering",
            primary_task_type="code_generation",
            gold_solution="result = 1",
            seed_execution_case={},
            validated_oracle_cases=[],
        )

    def test_submit_final_code_without_cases_does_not_pass_execution(self) -> None:
        env = self._env_without_cases()
        runtime = ToolRuntime(env, cfg=None)  # type: ignore[arg-type]

        result = runtime.submit_final_code("result = 1")

        self.assertFalse(result["ok"])
        self.assertTrue(result["compile_passed"])
        self.assertFalse(result["execution_passed"])
        self.assertEqual(result["evaluation_case_source"], "none")
        self.assertEqual(result["total_cases"], 0)
        self.assertIn("NO_EVALUATION_CASES", runtime.final_eval["failure_reasons"])

    def test_submit_final_code_rejects_empty_without_terminating(self) -> None:
        runtime = ToolRuntime(self._env_without_cases(), cfg=None)  # type: ignore[arg-type]

        result = runtime.submit_final_code("")

        self.assertFalse(result["ok"])
        self.assertFalse(result["terminated"])
        self.assertFalse(result["preflight_passed"])
        self.assertIn("EMPTY_FINAL_CODE", result["errors"])
        self.assertIsNone(runtime.final_eval)
        self.assertFalse(runtime.terminated)

    def test_submit_final_code_rejects_markdown_without_terminating(self) -> None:
        runtime = ToolRuntime(self._env_without_cases(), cfg=None)  # type: ignore[arg-type]

        result = runtime.submit_final_code("```python\nresult = 1\n```")

        self.assertFalse(result["ok"])
        self.assertFalse(result["terminated"])
        self.assertIn("MARKDOWN_OR_NATURAL_LANGUAGE_OUTPUT", result["errors"])
        self.assertFalse(runtime.terminated)

    def test_submit_final_code_rejects_unavailable_import_without_terminating(self) -> None:
        runtime = ToolRuntime(self._env_without_cases(), cfg=None)  # type: ignore[arg-type]

        result = runtime.submit_final_code("from my_module import helper\nresult = helper()")

        self.assertFalse(result["ok"])
        self.assertFalse(result["terminated"])
        self.assertIn("unavailable_import:my_module", result["errors"])
        self.assertIn("inline the required helper behavior", " ".join(result["repair_hints"]))
        self.assertFalse(runtime.terminated)

    def test_failed_submit_preflight_does_not_exhaust_final_submit_budget(self) -> None:
        runtime = ToolRuntime(self._env_without_cases(), cfg=None)  # type: ignore[arg-type]

        first = runtime.execute("submit_final_code", {"code": "from my_module import helper\nresult = helper()"})
        second = runtime.execute("submit_final_code", {"code": "result = 1"})

        self.assertFalse(first["terminated"])
        self.assertTrue(second["terminated"])
        self.assertTrue(runtime.terminated)
        self.assertEqual(runtime.call_counts.get("submit_final_code"), 1)

    def test_runner_allows_repair_after_final_code_preflight_failure(self) -> None:
        class FakeClient:
            mode = "unit"
            config = {"api": {"model": "fake"}}

            def __init__(self) -> None:
                self.calls = 0

            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls += 1
                code = "from my_module import helper\nresult = helper()" if self.calls == 1 else "result = 1"
                return ToolLLMResponse(
                    content=json.dumps({"final_code": code, "notes": []}),
                    tool_calls=[],
                    raw_message={"role": "assistant", "content": code},
                    source="unit",
                )

        client = FakeClient()

        row = run_tool_agent_for_env(
            env=self._env_without_cases(),
            cfg=None,  # type: ignore[arg-type]
            llm_client=client,  # type: ignore[arg-type]
            agent_cfg={"max_turns": 2},
            tool_pool_cfg=None,
            budget_cfg={"stage06_tool_agent": {"max_turns_by_level": {"M2": 2}}},
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(row["run"]["final_code"], "result = 1")
        self.assertEqual(row["trace"]["tool_trace"][0]["result"]["errors"], ["unavailable_import:my_module"])
        self.assertEqual(row["eval"]["failure_reasons"], ["NO_EVALUATION_CASES"])

    def test_stage06_records_llm_error_and_continues_by_default(self) -> None:
        class FailingClient:
            mode = "api"
            config = {"api": {"model": "xopqwen35v35b"}}

            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("LLM transient HTTP error 500")

        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="stage06-llm-error-"))
        cfg = AppConfig(
            root=root,
            values={"stage06": {"tool_agent": {"abort_on_llm_error": False}}, "output": {}},
            llm_values={},
            dataset_name="biocoder",
        )

        result = run_stage06_tool_agent(
            cfg=cfg,
            environments=[self._env_without_cases()],
            llm_client=FailingClient(),  # type: ignore[arg-type]
            output_dir=temp_dir,
            limit=1,
        )

        self.assertEqual(result["eval_report"][0]["evaluation_case_source"], "llm_error")
        self.assertEqual(result["eval_report"][0]["failure_reasons"], ["LLM_API_ERROR:RuntimeError"])
        self.assertFalse(result["runs"][0]["passed"])
        self.assertTrue((temp_dir / "xopqwen35v35b" / "agent_eval_report.jsonl").exists())

    def test_system_prompt_warns_against_unavailable_helper_imports(self) -> None:
        prompt = _system_prompt()

        self.assertIn("self-contained", prompt)
        self.assertIn("my_module", prompt)
        self.assertIn("implement the needed helper behavior inline", prompt)
        self.assertIn("requests_toolbelt", prompt)

    def test_run_custom_test_diagnoses_missing_module_import(self) -> None:
        result = run_custom_test(
            "from my_module import helper\n\nresult = helper()",
            "assert result is not None",
        )

        self.assertFalse(result["ok"])
        self.assertIn("ModuleNotFoundError", result["traceback_tail"])
        self.assertIn("module 'my_module' is unavailable", result["diagnosis"])
        self.assertIn("inline the required helper behavior", result["diagnosis"])


if __name__ == "__main__":
    unittest.main()
