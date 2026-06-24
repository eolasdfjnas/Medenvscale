from __future__ import annotations


DEMO_MEDAGENTGYM_ROWS = [
    {
        "idx": "demo_001",
        "problem": "Complete the function so it loads a CSV and returns only urgent cardiology rows.",
        "context": "import pandas as pd\n\ndef filter_urgent_rows(path):\n    <<insert solution here>>\n",
        "signature": "def filter_urgent_rows(path):",
        "solution": "df = pd.read_csv(path)\n    return df[df['urgent_review'] == True]",
        "code": "import pandas as pd",
        "task_family": "tabular_data_transformation",
        "resources": ["ehr/admissions.csv"],
    },
    {
        "idx": "demo_002",
        "problem": "Fill in the function body to count motifs in a FASTA sequence dictionary.",
        "context": "def count_motif(records, motif):\n    <<insert solution here>>\n",
        "signature": "def count_motif(records, motif):",
        "solution": "return sum(seq.count(motif) for seq in records.values())",
        "task_family": "sequence_and_structure_processing",
        "resources": ["seq/example.fasta"],
    },
    {
        "idx": "demo_003",
        "problem": "Write the expression that computes a stable z-score from mean and standard deviation.",
        "context": "def stable_zscore(value, mean, std):\n    return <<insert solution here>>\n",
        "signature": "def stable_zscore(value, mean, std):",
        "solution": "(value - mean) / std if std else 0.0",
        "task_family": "numerical_and_statistical_computation",
    },
    {
        "idx": "demo_004",
        "problem": "Patch the broken validator so it raises ValueError for empty sample names.",
        "context": "def validate_sample_name(name):\n    if name is None:\n        return False\n    <<insert solution here>>\n",
        "signature": "def validate_sample_name(name):",
        "solution": "if not name:\n        raise ValueError('name must not be empty')\n    return True",
        "task_family": "validation_and_code_utility",
    },
    {
        "idx": "demo_005",
        "problem": "Complete the function that converts a segmentation mask into the set of non-background labels.",
        "context": "def extract_labels(mask):\n    <<insert solution here>>\n",
        "signature": "def extract_labels(mask):",
        "solution": "return {int(value) for row in mask for value in row if int(value) != 0}",
        "task_family": "domain_model_or_image_analysis",
        "resources": ["mask/sample.npy"],
    },
]
