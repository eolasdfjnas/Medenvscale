from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medenvscale.classify.llm_full_taxonomy_router import route_with_llm_full_taxonomy
from medenvscale.classify.routing_validator import validate_routing, validate_secondary_domains
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import MedAgentGymTask


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed_domains = [
            "scientific_software_engineering",
            "bioinformatics_sequence_structure",
            "biomedical_data_analysis",
            "systems_molecular_modeling",
            "omics_measurement_analysis",
        ]
        self.allowed_task_types = [
            "file_io_and_formatting",
            "sequence_and_structure_processing",
            "numerical_and_statistical_computation",
            "tabular_data_transformation",
            "domain_model_or_image_analysis",
            "validation_and_code_utility",
        ]
        self.allowed_solution_forms = [
            "function_definition",
            "function_body",
            "expression_completion",
            "statement_block_completion",
            "decorated_function_definition",
            "patch_or_bugfix",
        ]
        self.client = LLMClient(
            config={"api": {"model": "unused", "api_key_env": "UNUSED", "base_url": "https://example.com"}},
            mode="mock",
            cache_dir=tempfile.mkdtemp(prefix="medenvscale-routing-"),
        )
        self.prompt_runner = PromptRunner(Path(__file__).resolve().parent.parent / "prompts")

    def test_task_type_decides_axis_priority_not_domain(self) -> None:
        item = MedAgentGymTask(
            task_id="demo",
            source_split="train",
            problem="Load a CSV into a pandas DataFrame and filter rows.",
            solution="return df[df['keep']]",
            context="def run(path):\n    <<insert solution here>>\n",
            signature="def run(path):",
            resource_files=["data/input.csv"],
        )
        payload = route_with_llm_full_taxonomy(item, self.client, self.prompt_runner, self.allowed_domains, self.allowed_task_types)
        routed = validate_routing(
            item=item,
            routing=payload,
            allowed_domains=self.allowed_domains,
            allowed_task_types=self.allowed_task_types,
            allowed_solution_forms=self.allowed_solution_forms,
            min_confidence=0.60,
            review_confidence=0.75,
        )
        self.assertEqual(routed.domain, "biomedical_data_analysis")
        self.assertEqual(routed.task_type, "tabular_data_transformation")
        self.assertEqual(routed.solution_form, "function_body")
        self.assertEqual([item.domain for item in routed.secondary_domains], [])

    def test_sequence_task_routing(self) -> None:
        item = MedAgentGymTask(
            task_id="demo2",
            source_split="train",
            problem="Count motifs in FASTA records.",
            solution="return 0",
            context="def count(records, motif):\n    <<insert solution here>>\n",
            signature="def count(records, motif):",
            resource_files=["seq/example.fasta"],
        )
        payload = route_with_llm_full_taxonomy(item, self.client, self.prompt_runner, self.allowed_domains, self.allowed_task_types)
        self.assertEqual(payload["primary_domain"], "bioinformatics_sequence_structure")
        self.assertEqual(payload["primary_task_type"], "sequence_and_structure_processing")

    def test_secondary_domains_are_validated_and_trimmed(self) -> None:
        item = MedAgentGymTask(
            task_id="demo3",
            source_split="train",
            problem="Analyze a proteomics dataframe and compute grouped intensity summaries.",
            solution="return df",
            context="def summarize(df):\n    <<insert solution here>>\n",
            signature="def summarize(df):",
            resource_files=["proteomics/input.csv"],
        )
        routed = validate_routing(
            item=item,
            routing={
                "primary_domain": "omics_measurement_analysis",
                "secondary_domains": [
                    {"domain": "biomedical_data_analysis", "relevance": 0.85, "reason": "Dataframe-heavy processing"},
                    {"domain": "scientific_software_engineering", "relevance": 0.40, "reason": "Formatting logic"},
                    {"domain": "omics_measurement_analysis", "relevance": 0.9, "reason": "duplicate of primary"},
                    {"domain": "bioinformatics_sequence_structure", "relevance": 0.2, "reason": "too weak"},
                ],
                "primary_task_type": "tabular_data_transformation",
                "solution_form": "function_body",
                "routing_reason": "test",
                "confidence": 0.9,
            },
            allowed_domains=self.allowed_domains,
            allowed_task_types=self.allowed_task_types,
            allowed_solution_forms=self.allowed_solution_forms,
            min_confidence=0.60,
            review_confidence=0.75,
        )
        self.assertEqual([item.domain for item in routed.secondary_domains], ["biomedical_data_analysis", "scientific_software_engineering"])
        self.assertEqual(validate_secondary_domains(routed.primary_domain, routed.secondary_domains), [])
