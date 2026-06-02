from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from bioevidence.evaluation.runner import format_report, run_evaluation, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local BioEvidence evaluation harness.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/demo/demo_eval_dataset.jsonl"),
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON evaluation report.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "agent"),
        default="baseline",
        help="Workflow mode to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/corpora/demo"),
        help="Corpus data directory. The runner reads processed/*.documents.jsonl under this path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of dataset items to evaluate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_evaluation(args.dataset, mode=args.mode, data_dir=args.data_dir, limit=args.limit)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    if args.output is not None:
        write_report(report, args.output)
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
