from __future__ import annotations

from medenvscale.classify.taxonomy import normalize_domain_name, normalize_task_type_name
from medenvscale.ingest.placeholder_analyzer import detect_solution_form
from medenvscale.llm import LLMClient
from medenvscale.llm.prompt_runner import PromptRunner
from medenvscale.schemas import DomainHint, MedAgentGymTask


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _infer_domain(text: str, resources: list[str], task_family: str | None) -> tuple[str, list[dict], list[str], list[str]]:
    resource_text = " ".join(resources).lower()
    family = (task_family or "").lower()

    if _contains_any(
        text + " " + resource_text + " " + family,
        ["fasta", "fastq", "bam", "vcf", "pdb", "residue", "chain", "genome", "sequence", "motif"],
    ):
        secondary = []
        if any(token in resource_text for token in [".csv", ".tsv", "table", "dataframe"]):
            secondary.append({"domain": "biomedical_data_analysis", "relevance": 0.55, "reason": "The task also relies on tabular resource handling."})
        if any(token in text + " " + resource_text for token in ["file", "parse", "yaml", "json", "cli", "subprocess"]):
            secondary.append({"domain": "scientific_software_engineering", "relevance": 0.65, "reason": "The task includes file parsing or software-oriented execution details."})
        return "bioinformatics_sequence_structure", secondary[:3], ["sequence", "structure"], ["sequence parsing", "structured biology objects"]

    if _contains_any(
        text + " " + resource_text + " " + family,
        ["flux", "stoichiometry", "reaction", "metabolic model", "mbar", "bar", "free energy", "thermodynamic"],
    ):
        secondary = [{"domain": "biomedical_data_analysis", "relevance": 0.45, "reason": "The task often includes matrix or table style analysis."}]
        return "systems_molecular_modeling", secondary[:3], ["systems biology", "molecular modeling"], ["scientific modeling", "numeric validation"]

    if _contains_any(
        text + " " + resource_text + " " + family,
        ["fdr", "psm", "lc-ms", "retention time", "mz", "metabolomics", "proteomics", "lipidomics"],
    ):
        secondary = [{"domain": "biomedical_data_analysis", "relevance": 0.82, "reason": "The task requires dataframe-level processing of omics measurements."}]
        return "omics_measurement_analysis", secondary[:3], ["omics measurements"], ["measurement analysis", "tolerance-aware scoring"]

    if _contains_any(
        text + " " + resource_text + " " + family,
        ["dataframe", "pandas", "csv", "tsv", "numpy", "matrix", "table", "segmentation", "mask", "image"],
    ):
        secondary = []
        if any(token in text + " " + resource_text for token in ["mz", "retention time", "proteomics", "metabolomics", "lipidomics"]):
            secondary.append({"domain": "omics_measurement_analysis", "relevance": 0.7, "reason": "The task processes omics-style measurements in a tabular representation."})
        if any(token in text + " " + resource_text for token in ["sequence", "fasta", "pdb"]):
            secondary.append({"domain": "bioinformatics_sequence_structure", "relevance": 0.45, "reason": "The task also touches sequence or structure-derived objects."})
        return "biomedical_data_analysis", secondary[:3], ["tabular analysis"], ["table inspection", "array reasoning"]

    secondary = []
    if any(token in text + " " + resource_text for token in ["reaction", "metabolic model", "free energy", "mbar", "bar"]):
        secondary.append({"domain": "systems_molecular_modeling", "relevance": 0.55, "reason": "The code utility is embedded in a modeling workflow."})
    if any(token in text + " " + resource_text for token in ["sequence", "fasta", "pdb", "genome"]):
        secondary.append({"domain": "bioinformatics_sequence_structure", "relevance": 0.5, "reason": "The software task manipulates bioinformatics resources."})
    return "scientific_software_engineering", secondary[:3], ["general scientific software"], ["code synthesis", "runtime validation"]


def _infer_task_types(text: str, resources: list[str], task_family: str | None) -> tuple[str, list[str]]:
    resource_text = " ".join(resources).lower()
    family = (task_family or "").lower()
    joint = f"{text} {resource_text} {family}"

    if _contains_any(joint, ["dataframe", "pandas", "groupby", "merge", "column", ".csv", ".tsv", "table"]):
        return "structured_data_processing", ["io_format_and_cli"]
    if _contains_any(joint, ["fasta", "fastq", "vcf", "bam", "pdb", "sequence", "residue", "motif"]):
        return "structured_data_processing", ["io_format_and_cli"]
    if _contains_any(joint, ["array", "matrix", "numeric", "statistic", "mean", "median", "variance", "mbar", "bar", "fdr"]):
        return "numerical_computation", ["code_validation_and_utility"]
    if _contains_any(joint, ["mask", "image", "segmentation", "reaction", "metabolic model", "object state"]):
        return "structured_data_processing", ["code_validation_and_utility"]
    if _contains_any(joint, ["validate", "exception", "raise", "decorator", "property", "patch", "bug", "utility"]):
        return "code_validation_and_utility", ["io_format_and_cli"]
    return "io_format_and_cli", ["code_validation_and_utility"]


def _infer_verifier_type(task_type: str, solution_form: str) -> str:
    if task_type == "structured_data_processing":
        return "dataframe_equal"
    if task_type == "numerical_computation":
        return "numeric_tolerance"
    if task_type == "code_validation_and_utility":
        return "exception_unit_test" if solution_form == "patch_or_bugfix" else "static_check"
    return "runtime_output_match"


def _mock_route_builder(context: dict) -> dict:
    text = f"{context['problem']} {context.get('context') or ''} {context.get('signature') or ''}".lower()
    resources = [str(path) for path in context.get("resource_files", []) or []]
    domain, secondary_domains, concepts, capabilities = _infer_domain(text, resources, context.get("task_family"))
    task_type, secondary = _infer_task_types(text, resources, context.get("task_family"))
    solution_form = detect_solution_form(
        problem=context["problem"],
        context=context.get("context") or "",
        signature=context.get("signature"),
        code=context.get("code"),
    )
    confidence = 0.83
    if task_type == "io_format_and_cli" and domain == "scientific_software_engineering":
        confidence = 0.72
    return {
        "primary_domain": domain,
        "secondary_domains": secondary_domains,
        "primary_task_type": task_type,
        "solution_form": solution_form,
        "secondary_task_types": secondary,
        "domain_concepts": concepts,
        "required_capabilities": capabilities,
        "verifier_type_hint": _infer_verifier_type(task_type, solution_form),
        "routing_reason": f"Mock router matched {task_type} in domain {domain}.",
        "confidence": confidence,
    }


def route_with_llm_full_taxonomy(
    medagent_item: MedAgentGymTask,
    llm_client: LLMClient,
    prompt_runner: PromptRunner,
    allowed_domains: list[str],
    allowed_task_types: list[str],
) -> dict:
    prompt = prompt_runner.render(
        "task_router.jinja",
        problem=medagent_item.problem,
        context=medagent_item.context_summary or medagent_item.context,
        signature=medagent_item.signature or "",
        code=medagent_item.code or "",
        resource_files=medagent_item.resource_files,
        task_family=medagent_item.task_family or "",
        domains=allowed_domains,
        task_types=allowed_task_types,
    )
    response = llm_client.complete_json(
        task_name="route_medagentgym_task",
        prompt=prompt,
        context={
            "task_id": medagent_item.task_id,
            "problem": medagent_item.problem,
            "context": medagent_item.context,
            "signature": medagent_item.signature,
            "code": medagent_item.code,
            "resource_files": medagent_item.resource_files,
            "task_family": medagent_item.task_family,
        },
        mock_builder=_mock_route_builder,
    )
    payload = dict(response.payload)
    payload["primary_domain"] = normalize_domain_name(payload.get("primary_domain") or payload.get("domain"))
    payload["primary_task_type"] = normalize_task_type_name(payload.get("primary_task_type") or payload.get("task_type"))
    secondary_domains = payload.get("secondary_domains", []) or []
    if isinstance(secondary_domains, dict):
        secondary_domains = [secondary_domains]
    payload["secondary_domains"] = [
        (item if isinstance(item, DomainHint) else DomainHint.model_validate(item)).model_dump() for item in secondary_domains
    ]
    payload.setdefault("routing_trace", {})
    payload["routing_trace"].update(
        {
            "router": "llm_full_taxonomy",
            "source": response.source,
            "model_name": llm_client.config.get("api", {}).get("model", f"{llm_client.mode}_router"),
        }
    )
    return payload
