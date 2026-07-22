from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medenvscale.agent.candidate_execution import run_custom_test
from medenvscale.agent.runner import _system_prompt, build_stage06_summary, run_stage06_tool_agent, run_tool_agent_for_env
from medenvscale.agent.tool_schemas import stage06_tool_names
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

    def test_create_test_file_is_available_to_run_custom_test(self) -> None:
        runtime = ToolRuntime(self._env_without_cases(), cfg=None)  # type: ignore[arg-type]

        created = runtime.execute("create_test_file", {"path": "data/input.tsv", "content": "gene\tvalue\nA\t3\n"})
        result = runtime.execute(
            "run_custom_test",
            {
                "code": "from pathlib import Path\n\ndef load_value(path):\n    return Path(path).read_text().splitlines()[1].split('\\t')[1]",
                "test_snippet": "assert load_value('data/input.tsv') == '3'",
            },
        )

        self.assertTrue(created["ok"])
        self.assertTrue(result["ok"])

    def test_tool_budget_overage_is_marked_but_tool_still_runs(self) -> None:
        runtime = ToolRuntime(
            self._env_without_cases(),
            cfg=None,  # type: ignore[arg-type]
            budget={
                "max_total_tool_calls": 1,
                "max_calls_per_tool": {
                    "get_task_context": 1,
                    "create_test_file": 1,
                    "validate_candidate_code": 1,
                    "run_custom_test": 1,
                    "submit_final_code": 1,
                },
            },
        )

        first = runtime.execute("get_task_context", {"window": 4000})
        second = runtime.execute("get_task_context", {"window": 4000})

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(second["budget_violation"])
        self.assertEqual(second["budget_error"], "tool_budget_exceeded:get_task_context")
        self.assertTrue(runtime.trace[1]["budget_violation"])
        self.assertEqual(runtime.trace[1]["result"]["env_id"], "env_no_cases_M1")

    def test_create_test_file_rejects_unsafe_path(self) -> None:
        runtime = ToolRuntime(self._env_without_cases(), cfg=None)  # type: ignore[arg-type]

        result = runtime.execute("create_test_file", {"path": "../escape.txt", "content": "bad"})

        self.assertFalse(result["ok"])
        self.assertIn("unsafe_path", result["error"])

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
                self.seen_messages = []

            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls += 1
                self.seen_messages.append(list(kwargs["messages"]))
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
        second_turn_messages = client.seen_messages[1]
        self.assertTrue(
            any(message.get("role") == "tool" and message.get("name") == "submit_final_code" for message in second_turn_messages)
        )
        self.assertFalse(
            any(
                message.get("role") == "user" and "failed public preflight" in str(message.get("content") or "")
                for message in second_turn_messages
            )
        )

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
        retry_rows = [
            json.loads(line)
            for line in (temp_dir / "xopqwen35v35b" / "retry_failed_envs.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual([row["env_id"] for row in retry_rows], ["env_no_cases_M1"])

    def test_stage06_retry_failed_merges_results_and_recomputes_summary(self) -> None:
        class FailingClient:
            mode = "api"
            config = {"api": {"model": "xopqwen35v35b"}}

            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("LLM network error from https://api.example.test. RemoteDisconnected")

        class RepairClient:
            mode = "api"
            config = {"api": {"model": "xopqwen35v35b"}}

            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                return ToolLLMResponse(
                    content=json.dumps({"final_code": "result = 1", "notes": []}),
                    tool_calls=[],
                    raw_message={"role": "assistant", "content": "result = 1"},
                    source="unit",
                )

        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="stage06-retry-"))
        cfg = AppConfig(
            root=root,
            values={"stage06": {"tool_agent": {"abort_on_llm_error": False, "max_turns": 1}}, "output": {}},
            llm_values={},
            dataset_name="biocoder",
        )
        env = self._env_without_cases()

        first = run_stage06_tool_agent(
            cfg=cfg,
            environments=[env],
            llm_client=FailingClient(),  # type: ignore[arg-type]
            output_dir=temp_dir,
        )
        self.assertEqual(first["eval_report"][0]["evaluation_case_source"], "llm_error")

        retry = run_stage06_tool_agent(
            cfg=cfg,
            environments=[env],
            llm_client=RepairClient(),  # type: ignore[arg-type]
            output_dir=temp_dir,
            retry_failed=True,
        )

        self.assertEqual(len(retry["eval_report"]), 1)
        self.assertEqual(retry["eval_report"][0]["evaluation_case_source"], "none")
        self.assertEqual(retry["summary"]["evaluation_case_sources"], {"none": 1})
        self.assertEqual(
            (temp_dir / "xopqwen35v35b" / "retry_failed_envs.jsonl").read_text(encoding="utf-8"),
            "",
        )

    def test_stage06_resume_rebuilds_outputs_from_checkpoint(self) -> None:
        class RepairClient:
            mode = "api"
            config = {"api": {"model": "xopqwen35v35b"}}

            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                return ToolLLMResponse(
                    content=json.dumps({"final_code": "result = 1", "notes": []}),
                    tool_calls=[],
                    raw_message={"role": "assistant", "content": "result = 1"},
                    source="unit",
                )

        class BombClient(RepairClient):
            def complete_with_tools(self, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("resume should not call the model")

        root = Path(__file__).resolve().parent.parent
        temp_dir = Path(tempfile.mkdtemp(prefix="stage06-resume-"))
        cfg = AppConfig(
            root=root,
            values={"stage06": {"tool_agent": {"abort_on_llm_error": False, "max_turns": 1}}, "output": {}},
            llm_values={},
            dataset_name="biocoder",
        )
        env = self._env_without_cases()
        run_stage06_tool_agent(
            cfg=cfg,
            environments=[env],
            llm_client=RepairClient(),  # type: ignore[arg-type]
            output_dir=temp_dir,
            resume=True,
        )
        leaf = temp_dir / "xopqwen35v35b"
        for name in ("agent_runs.jsonl", "agent_traces.jsonl", "agent_eval_report.jsonl", "summary.json"):
            (leaf / name).unlink()

        resumed = run_stage06_tool_agent(
            cfg=cfg,
            environments=[env],
            llm_client=BombClient(),  # type: ignore[arg-type]
            output_dir=temp_dir,
            resume=True,
        )

        self.assertEqual(len(resumed["runs"]), 1)
        self.assertTrue((leaf / "agent_runs.jsonl").exists())
        self.assertEqual(resumed["runs"][0]["env_id"], env.env_id)

    def test_summary_reports_zero_case_penalized_case_rate(self) -> None:
        class FakeClient:
            mode = "api"
            config = {"api": {"model": "fake-model"}}

        runs = [
            {"env_id": "env_a_M1", "difficulty": {"global_level": "M1"}},
            {"env_id": "env_b_M1", "difficulty": {"global_level": "M1"}},
            {"env_id": "env_c_M1", "difficulty": {"global_level": "M1"}},
        ]
        eval_rows = [
            {"env_id": "env_a_M1", "level": "M1", "execution_passed": True, "compile_passed": True, "passed_cases": 2, "total_cases": 2},
            {"env_id": "env_b_M1", "level": "M1", "execution_passed": False, "compile_passed": True, "passed_cases": 1, "total_cases": 2},
            {
                "env_id": "env_c_M1",
                "level": "M1",
                "execution_passed": False,
                "compile_passed": False,
                "passed_cases": 0,
                "total_cases": 0,
                "failure_reasons": ["MAX_TURNS_WITHOUT_VALID_FINAL_CODE"],
            },
        ]

        summary = build_stage06_summary(
            runs=runs,
            traces=[{"tool_trace": []}, {"tool_trace": []}, {"tool_trace": []}],
            eval_rows=eval_rows,
            llm_client=FakeClient(),  # type: ignore[arg-type]
        )

        self.assertEqual(summary["sample_pass_rate"], 0.3333)
        self.assertEqual(summary["case_pass_rate"], 0.6)
        self.assertEqual(summary["case_pass_rate_nonzero_only"], 0.75)
        self.assertEqual(summary["zero_case_samples"], 1)
        self.assertEqual(summary["unevaluated_samples"], 1)
        self.assertEqual(summary["no_final_code_samples"], 1)
        self.assertEqual(summary["case_pass_rate_with_zero_case_penalty"], 0.6)
        self.assertEqual(summary["levels"]["M1"]["case_pass_rate"], 0.6)
        self.assertEqual(summary["levels"]["M1"]["unevaluated_samples"], 1)
        self.assertEqual(summary["levels"]["M1"]["no_final_code_samples"], 1)
        self.assertEqual(summary["levels"]["M1"]["case_pass_rate_with_zero_case_penalty"], 0.6)

    def test_system_prompt_warns_against_unavailable_helper_imports(self) -> None:
        prompt = _system_prompt()

        self.assertIn("self-contained", prompt)
        self.assertIn("create_test_file", prompt)
        self.assertIn("my_module", prompt)
        self.assertIn("implement the needed helper behavior inline", prompt)
        self.assertIn("requests_toolbelt", prompt)

    def test_stage06_tool_names_include_create_test_file(self) -> None:
        self.assertIn("create_test_file", stage06_tool_names(None))

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
