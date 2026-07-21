from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.retrieval.corpus import load_local_documents


def main() -> None:
    args = _parse_args()
    documents = load_local_documents(args.data_dir)
    annotations = load_extraction_annotations(args.dataset, documents)

    evidence_counts = Counter(annotation.extraction.evidence_status.value for annotation in annotations)
    review_counts = Counter(annotation.annotation_status.value for annotation in annotations)
    print(f"Validated annotations: {len(annotations)}")
    print(f"Corpus documents: {len(documents)}")
    print("Evidence status: " + _format_counts(evidence_counts))
    print("Annotation status: " + _format_counts(review_counts))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate query-focused evidence extraction annotations.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
        help="Annotation JSONL path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/corpora/demo"),
        help="Corpus directory containing processed/*.documents.jsonl files.",
    )
    return parser.parse_args()


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


if __name__ == "__main__":
    main()
