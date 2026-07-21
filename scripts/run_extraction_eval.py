from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.evaluation.extraction_runner import (
    format_extraction_report,
    run_extraction_evaluation,
    write_extraction_report,
)
from bioevidence.extraction.model_backend import ExtractionBackend, PromptedExtractionBackend, RuleBasedExtractionBackend
from bioevidence.retrieval.corpus import load_local_documents


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    documents = load_local_documents(args.data_dir)
    annotations = load_extraction_annotations(args.dataset, documents)
    backend: ExtractionBackend
    if args.backend == "rules":
        backend = RuleBasedExtractionBackend()
    else:
        backend = PromptedExtractionBackend(
            api_key=args.api_key or os.getenv("EXTRACTION_API_KEY", ""),
            base_url=args.base_url or os.getenv("EXTRACTION_BASE_URL", ""),
            model=args.model or os.getenv("EXTRACTION_MODEL", ""),
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
    report = run_extraction_evaluation(annotations, backend, limit=args.limit)
    write_extraction_report(report, args.output)
    print(format_extraction_report(report))
    print(f"Report: {args.output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an evidence extraction backend.")
    parser.add_argument("--backend", choices=("rules", "prompted"), default="rules")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/extraction_prompted.json"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
