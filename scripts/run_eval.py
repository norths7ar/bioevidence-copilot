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
        default=Path("data/eval/dataset.jsonl"),
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON evaluation report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_evaluation(args.dataset)
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
