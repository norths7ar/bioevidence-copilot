from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioevidence.evaluation.extraction_candidates import (
    build_annotation_prompt_records,
    build_candidate_manifest,
    load_candidate_topics,
    select_expansion_candidates,
)
from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.retrieval.corpus import load_local_documents


def main() -> int:
    args = _parse_args()
    documents = load_local_documents(args.data_dir)
    annotations = load_extraction_annotations(args.annotations, documents)
    candidates = select_expansion_candidates(
        load_candidate_topics(args.corpus_manifest),
        documents,
        annotations,
        high_per_topic=args.high_per_topic,
        broad_per_topic=args.broad_per_topic,
        hard_negative_per_topic=args.hard_negative_per_topic,
    )
    manifest = build_candidate_manifest(
        candidates,
        source_corpus=args.corpus,
        existing_annotations=args.annotations,
        high_per_topic=args.high_per_topic,
        broad_per_topic=args.broad_per_topic,
        hard_negative_per_topic=args.hard_negative_per_topic,
    )
    _write_jsonl(args.output, [candidate.to_record() for candidate in candidates])
    _write_jsonl(args.prompt_output, build_annotation_prompt_records(candidates))
    _write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"Candidates: {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stratified extraction annotation queue.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpora/demo/processed/demo.documents.jsonl"),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/corpora/demo/processed/demo.manifest.json"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/expansion_candidates.v1.jsonl"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/expansion_candidates.v1.manifest.json"),
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/expansion_prompt_queue.v1.jsonl"),
    )
    parser.add_argument("--high-per-topic", type=int, default=4)
    parser.add_argument("--broad-per-topic", type=int, default=2)
    parser.add_argument("--hard-negative-per-topic", type=int, default=2)
    return parser.parse_args()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
