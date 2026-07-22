from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medenvscale.config import resolve_dataset_config_path, resolve_dataset_config_path_with_fallback
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.scaling.tool_config_validator import build_tool_config
from medenvscale.schemas import DomainHint
from medenvscale.utils import load_yaml


class ToolConfigSecondaryDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.root = root
        self.budgets_cfg = load_yaml(
            resolve_dataset_config_path_with_fallback(
                root,
                "m_level_budgets_4axis.yaml",
                "m_level_budgets_7axis.yaml",
                dataset="biocoder",
            )
        )
        self.tool_pool_cfg = load_yaml(resolve_dataset_config_path(root, "tool_pool.yaml", dataset="biocoder"))
        self.client = LLMClient(
            config={"api": {"model": "unused", "api_key_env": "UNUSED", "base_url": "https://example.com"}},
            mode="mock",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-tool-"),
        )
        self.prompt_runner = PromptRunner(root / "prompts")

    def test_secondary_domains_only_add_limited_tools(self) -> None:
        config = build_tool_config(
            env_id="env_demo_M3",
            global_level="M3",
            task_type="validation_and_code_utility",
            primary_domain="systems_molecular_modeling",
            secondary_domains=[
                DomainHint(domain="bioinformatics_sequence_structure", relevance=0.7),
                DomainHint(domain="omics_measurement_analysis", relevance=0.8),
            ],
            solution_form="function_body",
            resource_manifest=["input.fasta"],
            required_capabilities=["file_handling"],
            scaling_plan={"selected_axes": ["D", "V"], "axis_intensity": {"D": 2, "C": 1, "A": 0, "V": 2}},
            budgets_cfg=self.budgets_cfg,
            tool_pool_cfg=self.tool_pool_cfg,
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo", "problem": "Inspect structure and validate numeric outputs."},
        )
        tool_names = [tool.tool_name for tool in config.allowed_tools]
        self.assertIn("get_task_context", tool_names)
        self.assertIn("create_test_file", tool_names)
        self.assertIn("run_custom_test", tool_names)
        self.assertIn("submit_final_code", tool_names)
        self.assertNotIn("run_validated_oracle_cases", tool_names)
        self.assertIn("secondary_domains=bioinformatics_sequence_structure, omics_measurement_analysis", config.tool_choice_reason)
        self.assertIn(config.planning_source, {"llm", "repaired"})

    def test_primary_domain_required_tool_survives_small_tool_cap(self) -> None:
        config = build_tool_config(
            env_id="env_demo_M1",
            global_level="M1",
            task_type="numerical_and_statistical_computation",
            primary_domain="systems_molecular_modeling",
            secondary_domains=[DomainHint(domain="biomedical_data_analysis", relevance=0.7)],
            solution_form="statement_block_completion",
            resource_manifest=["input.csv"],
            required_capabilities=[],
            scaling_plan={
                "global_level": "M1",
                "selected_axes": [],
                "axis_intensity": {"D": 0, "C": 0, "A": 0, "V": 0},
            },
            budgets_cfg=self.budgets_cfg,
            tool_pool_cfg=self.tool_pool_cfg,
            llm_client=self.client,
            prompt_runner=self.prompt_runner,
            seed_task={"task_id": "demo_m1", "problem": "Compute BAR free energy from work arrays."},
        )
        tool_names = [tool.tool_name for tool in config.allowed_tools]
        self.assertIn("get_task_context", tool_names)
        self.assertIn("submit_final_code", tool_names)
