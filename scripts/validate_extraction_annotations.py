from __future__ import annotations

import argparse
import json
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
    if args.candidates is not None:
        candidate_keys = _load_candidate_keys(args.candidates)
        annotation_keys = {(annotation.id, annotation.query, annotation.document.pmid) for annotation in annotations}
        missing = candidate_keys - annotation_keys
        extra = annotation_keys - candidate_keys
        if missing or extra:
            raise ValueError(f"candidate coverage mismatch: missing={len(missing)}, extra={len(extra)}")
        print(f"Candidate coverage: {len(annotation_keys)}/{len(candidate_keys)}")


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
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional candidate JSONL whose id/query/PMID coverage must match exactly.",
    )
    return parser.parse_args()


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _load_candidate_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        payload = json.loads(line)
        try:
            key = (str(payload["id"]), str(payload["query"]), str(payload["pmid"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: candidate requires id, query, and pmid") from exc
        if key in keys:
            raise ValueError(f"{path}:{line_number}: duplicate candidate key {key}")
        keys.add(key)
    return keys


if __name__ == "__main__":
    main()
