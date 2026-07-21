from __future__ import annotations

import argparse
from pathlib import Path

from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.evaluation.extraction_review import render_extraction_review
from bioevidence.retrieval.corpus import load_local_documents


def main() -> None:
    args = _parse_args()
    documents = load_local_documents(args.data_dir)
    annotations = load_extraction_annotations(args.dataset, documents)
    report = render_extraction_review(annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"Wrote review report: {args.output}")
    print(f"Review items: {len(annotations)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Markdown review packet for extraction annotations.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/annotation_reviews/extraction_pilot_review.md"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
