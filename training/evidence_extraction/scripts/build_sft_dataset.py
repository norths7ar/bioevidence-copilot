from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioevidence.evaluation.extraction_dataset import AnnotationStatus, load_extraction_annotations
from bioevidence.evaluation.extraction_sft import SplitRatios, write_sft_dataset
from bioevidence.retrieval.corpus import load_local_documents


def main() -> int:
    args = _parse_args()
    annotations = load_extraction_annotations(args.dataset, load_local_documents(args.data_dir))
    included_statuses = {AnnotationStatus(value) for value in args.annotation_status}
    selected = [annotation for annotation in annotations if annotation.annotation_status in included_statuses]
    source_metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    manifest = write_sft_dataset(
        selected,
        args.output_dir,
        source_dataset=args.dataset.as_posix(),
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
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
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


if __name__ == "__main__":
    raise SystemExit(main())
