from __future__ import annotations

import unittest

from medenvscale.schemas import DynamicOperatorInstance, DomainHint, ExecutableEnvSpec, OperatorConstraints, RoutingResult, StateUpdates, VerificationContract, VerifierDelta


class SchemaTests(unittest.TestCase):
    def test_routing_result(self) -> None:
        route = RoutingResult(
            task_id="demo",
            source_split="train",
            primary_domain="biomedical_data_analysis",
            secondary_domains=[DomainHint(domain="omics_measurement_analysis", relevance=0.8, reason="omics table")],
            primary_task_type="tabular_data_transformation",
            solution_form="function_body",
            routing_reason="test",
            confidence=0.9,
        )
        self.assertEqual(route.primary_domain, "biomedical_data_analysis")
        self.assertEqual(route.primary_task_type, "tabular_data_transformation")
        self.assertEqual(route.secondary_domains[0].domain, "omics_measurement_analysis")

    def test_nested_environment_and_operator(self) -> None:
        env = ExecutableEnvSpec(
            env_id="env_demo_M2",
            original_task_id="demo",
            split="train",
            problem="Patch the validator",
            context="def validate(x):\n    <<insert solution here>>\n",
            solution_form="patch_or_bugfix",
            primary_domain="scientific_software_engineering",
            secondary_domains=[DomainHint(domain="bioinformatics_sequence_structure", relevance=0.5)],
            primary_task_type="validation_and_code_utility",
            gold_solution="return True",
        )
        op = DynamicOperatorInstance(
            operator_id="env_demo_M2_v_01",
            axis="V",
            operator_type="validator_v_scientific_software_engineering",
            operator_intensity=1,
            transformation_goal="Add hidden checks",
            rationale="test",
            state_updates=StateUpdates(task_state_patch={"axis_constraints": ["V"]}),
            verifier_delta=VerifierDelta(new_hidden_tests=[{"name": "x", "assertion_code": "assert True"}]),
            verification_contract=VerificationContract(),
            constraints=OperatorConstraints(),
        )
        self.assertEqual(env.original_task_id, "demo")
        self.assertEqual(env.secondary_domains[0].domain, "bioinformatics_sequence_structure")
        self.assertEqual(op.axis, "V")
