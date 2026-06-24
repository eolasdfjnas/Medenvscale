from __future__ import annotations

CANONICAL_DOMAINS = [
    "scientific_software_engineering",
    "bioinformatics_sequence_structure",
    "biomedical_data_analysis",
    "systems_molecular_modeling",
    "omics_measurement_analysis",
]

CANONICAL_TASK_TYPES = [
    "io_format_and_cli",
    "structured_data_processing",
    "numerical_computation",
    "code_validation_and_utility",
]

CANONICAL_SOLUTION_FORMS = [
    "function_definition",
    "function_body",
    "expression_completion",
    "statement_block_completion",
    "decorated_function_definition",
    "patch_or_bugfix",
]

DOMAIN_ALIASES = {
    "scientific_software_engineering": "scientific_software_engineering",
    "scientific_software": "scientific_software_engineering",
    "software_engineering": "scientific_software_engineering",
    "cli": "scientific_software_engineering",
    "report_generation": "scientific_software_engineering",
    "bioinformatics_sequence_structure": "bioinformatics_sequence_structure",
    "bioinformatics": "bioinformatics_sequence_structure",
    "sequence": "bioinformatics_sequence_structure",
    "sequence_structure": "bioinformatics_sequence_structure",
    "genomics": "bioinformatics_sequence_structure",
    "protein_structure": "bioinformatics_sequence_structure",
    "biomedical_data_analysis": "biomedical_data_analysis",
    "data_analysis": "biomedical_data_analysis",
    "pandas": "biomedical_data_analysis",
    "numpy": "biomedical_data_analysis",
    "tabular": "biomedical_data_analysis",
    "systems_molecular_modeling": "systems_molecular_modeling",
    "systems_biology": "systems_molecular_modeling",
    "molecular_modeling": "systems_molecular_modeling",
    "metabolic_model": "systems_molecular_modeling",
    "omics_measurement_analysis": "omics_measurement_analysis",
    "omics": "omics_measurement_analysis",
    "proteomics": "omics_measurement_analysis",
    "metabolomics": "omics_measurement_analysis",
}

TASK_TYPE_ALIASES = {
    "io_format_and_cli": "io_format_and_cli",
    "file_io": "io_format_and_cli",
    "formatting": "io_format_and_cli",
    "cli": "io_format_and_cli",
    "report_generation": "io_format_and_cli",
    "structured_data_processing": "structured_data_processing",
    "sequence_and_structure_processing": "structured_data_processing",
    "sequence_processing": "structured_data_processing",
    "structure_processing": "structured_data_processing",
    "tabular_data_transformation": "structured_data_processing",
    "tabular": "structured_data_processing",
    "dataframe": "structured_data_processing",
    "domain_model_or_image_analysis": "structured_data_processing",
    "domain_model": "structured_data_processing",
    "image_analysis": "structured_data_processing",
    "numerical_computation": "numerical_computation",
    "numerical_and_statistical_computation": "numerical_computation",
    "numerical": "numerical_computation",
    "statistical": "numerical_computation",
    "code_validation_and_utility": "code_validation_and_utility",
    "validation_and_code_utility": "code_validation_and_utility",
    "validation": "code_validation_and_utility",
    "code_utility": "code_validation_and_utility",
    "utility": "code_validation_and_utility",
}

SOLUTION_FORM_ALIASES = {
    "function_definition": "function_definition",
    "function_body": "function_body",
    "expression_completion": "expression_completion",
    "statement_block_completion": "statement_block_completion",
    "decorated_function_definition": "decorated_function_definition",
    "patch_or_bugfix": "patch_or_bugfix",
    "bugfix": "patch_or_bugfix",
    "patch": "patch_or_bugfix",
}


def _compact(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_domain_name(domain: str | None) -> str:
    if not domain:
        return "scientific_software_engineering"
    compact = _compact(domain)
    return DOMAIN_ALIASES.get(compact, "scientific_software_engineering")


def normalize_task_type_name(task_type: str | None) -> str:
    if not task_type:
        return "code_validation_and_utility"
    compact = _compact(task_type)
    return TASK_TYPE_ALIASES.get(compact, "code_validation_and_utility")


def normalize_solution_form_name(solution_form: str | None) -> str:
    if not solution_form:
        return "statement_block_completion"
    compact = _compact(solution_form)
    return SOLUTION_FORM_ALIASES.get(compact, "statement_block_completion")


def is_known_domain_name(domain: str | None) -> bool:
    if not domain:
        return False
    return _compact(domain) in DOMAIN_ALIASES


def is_known_task_type_name(task_type: str | None) -> bool:
    if not task_type:
        return False
    return _compact(task_type) in TASK_TYPE_ALIASES


def is_known_solution_form_name(solution_form: str | None) -> bool:
    if not solution_form:
        return False
    return _compact(solution_form) in SOLUTION_FORM_ALIASES
