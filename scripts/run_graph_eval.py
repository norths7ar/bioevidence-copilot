from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from bioevidence.evaluation.graph_gain import run_graph_gain_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure whether Hetionet expansion improves PMID retrieval.")
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluations/demo/demo_eval_dataset.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_graph_gain_evaluation(
            args.dataset,
            data_dir=args.data_dir,
            limit=args.limit,
        )
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Graph evaluation failed: {exc}", file=sys.stderr)
        return 1
    payload = report.to_dict()
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
