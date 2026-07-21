from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioevidence.evaluation.extraction_dataset import AnnotationStatus, ExtractionAnnotation, load_extraction_annotations
from bioevidence.evaluation.extraction_sft import SplitRatios, write_sft_dataset
from bioevidence.retrieval.corpus import load_local_documents


def main() -> int:
    args = _parse_args()
    dataset_paths = args.datasets or [Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl")]
    documents = load_local_documents(args.data_dir)
    annotations = [
        annotation
        for dataset_path in dataset_paths
        for annotation in load_extraction_annotations(dataset_path, documents)
    ]
    _validate_unique_annotations(annotations)
    included_statuses = {AnnotationStatus(value) for value in args.annotation_status}
    selected = [annotation for annotation in annotations if annotation.annotation_status in included_statuses]
    source_metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    manifest = write_sft_dataset(
        selected,
        args.output_dir,
        source_dataset=" + ".join(path.as_posix() for path in dataset_paths),
        ratios=SplitRatios(train=args.train_ratio, dev=args.dev_ratio, test=args.test_ratio),
        seed=args.seed,
        source_metadata=source_metadata,
    )
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_output.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Dataset: {args.output_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PMID-safe Qwen chat-format extraction data.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        type=Path,
        help="Annotation JSONL. Repeat to combine versioned sources; defaults to the pilot.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_dataset_metadata.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/pilot_sft"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_split_manifest.json"),
    )
    parser.add_argument(
        "--annotation-status",
        nargs="+",
        choices=[status.value for status in AnnotationStatus],
        default=[status.value for status in AnnotationStatus],
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_unique_annotations(annotations: list[ExtractionAnnotation]) -> None:
    ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for annotation in annotations:
        annotation_id = annotation.id
        pair = (annotation.query, annotation.document.pmid)
        if annotation_id in ids:
            raise ValueError(f"duplicate annotation id across datasets: {annotation_id}")
        if pair in pairs:
            raise ValueError(f"duplicate query-PMID pair across datasets: {pair}")
        ids.add(annotation_id)
        pairs.add(pair)


if __name__ == "__main__":
    raise SystemExit(main())
