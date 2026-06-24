from __future__ import annotations

from collections import Counter
from pathlib import Path

from _common import base_parser
from medenvscale.pipeline_ops import load_config, normalize_output_path, routing_output_path, stage02_route


def _print_summary(rows) -> None:
    domain_counts = Counter(row.primary_domain for row in rows)
    task_counts = Counter(row.primary_task_type for row in rows)
    usable_count = sum(1 for row in rows if row.usable_for_generation)
    review_count = sum(1 for row in rows if row.needs_review)

    print(f"Total items: {len(rows)}")
    print(f"Usable for generation: {usable_count}")
    print(f"Filtered: {len(rows) - usable_count}")
    print(f"Needs review: {review_count}")
    print("Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    print("Task type distribution:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count}")


if __name__ == "__main__":
    parser = base_parser("Route MedQA domain/task type")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)
    input_path = Path(args.input) if args.input else normalize_output_path(cfg)
    output_path = Path(args.output) if args.output else routing_output_path(cfg)
    rows = stage02_route(cfg, limit=args.limit, llm_mode=args.llm_mode, input_path=input_path, output_path=output_path)
    _print_summary(rows)
    print(f"Routing results written to: {output_path}")
