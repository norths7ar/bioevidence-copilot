from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioevidence.evaluation.extraction_dataset import ExtractionAnnotation
from bioevidence.extraction.model_backend import build_extraction_messages


SPLIT_NAMES = ("train", "dev", "test")
TARGET_SCHEMA_ID = "https://bioevidence.local/schemas/model-evidence-extraction-v1.json"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be non-negative")
        if not abs(sum(values) - 1.0) < 1e-9:
            raise ValueError("split ratios must sum to 1.0")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(SPLIT_NAMES, (self.train, self.dev, self.test), strict=True))


def assign_pmid_splits(
    annotations: Sequence[ExtractionAnnotation],
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, str]:
    """Assign every unique PMID to one deterministic split."""

    pmids = sorted({annotation.document.pmid for annotation in annotations})
    ranked_pmids = sorted(pmids, key=lambda pmid: _stable_rank(pmid, seed))
    counts = _allocate_counts(len(ranked_pmids), ratios.as_dict())

    assignments: dict[str, str] = {}
    offset = 0
    for split_name in SPLIT_NAMES:
        next_offset = offset + counts[split_name]
        assignments.update({pmid: split_name for pmid in ranked_pmids[offset:next_offset]})
        offset = next_offset
    return assignments


def build_chat_example(
    annotation: ExtractionAnnotation,
    *,
    split: str,
    source_dataset: str,
) -> dict[str, Any]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}")
    messages = build_extraction_messages(annotation.query, annotation.document)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                annotation.extraction.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )
    return {
        "messages": messages,
        "metadata": {
            "annotation_id": annotation.id,
            "pmid": annotation.document.pmid,
            "annotation_status": annotation.annotation_status.value,
            "evidence_status": annotation.extraction.evidence_status.value,
            "split": split,
            "source_dataset": source_dataset,
            "target_schema": TARGET_SCHEMA_ID,
        },
    }


def write_sft_dataset(
    annotations: Sequence[ExtractionAnnotation],
    output_dir: Path,
    *,
    source_dataset: str,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not annotations:
        raise ValueError("at least one annotation is required")

    assignments = assign_pmid_splits(annotations, ratios=ratios, seed=seed)
    grouped: dict[str, list[ExtractionAnnotation]] = {name: [] for name in SPLIT_NAMES}
    for annotation in sorted(annotations, key=lambda item: (item.document.pmid, item.id)):
        grouped[assignments[annotation.document.pmid]].append(annotation)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_annotations = grouped[split_name]
        records = [
            build_chat_example(annotation, split=split_name, source_dataset=source_dataset)
            for annotation in split_annotations
        ]
        _write_jsonl(output_dir / f"{split_name}.jsonl", records)
        split_summaries[split_name] = _summarize_split(split_annotations)

    manifest = {
        "format": "qwen_chat_messages_v1",
        "source_dataset": source_dataset,
        "target_schema": TARGET_SCHEMA_ID,
        "seed": seed,
        "ratios": ratios.as_dict(),
        "rows": len(annotations),
        "unique_pmids": len(assignments),
        "splits": split_summaries,
        "source_metadata": dict(source_metadata or {}),
    }
    _write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _stable_rank(pmid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{pmid}".encode()).hexdigest()


def _allocate_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(exact[name]) for name in SPLIT_NAMES}
    remaining = total - sum(counts.values())
    priority = sorted(SPLIT_NAMES, key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)))
    for name in priority[:remaining]:
        counts[name] += 1
    return counts


def _summarize_split(annotations: Sequence[ExtractionAnnotation]) -> dict[str, Any]:
    return {
        "rows": len(annotations),
        "unique_pmids": len({annotation.document.pmid for annotation in annotations}),
        "pmids": sorted({annotation.document.pmid for annotation in annotations}),
        "annotation_ids": sorted(annotation.id for annotation in annotations),
        "annotation_status": dict(sorted(Counter(item.annotation_status.value for item in annotations).items())),
        "evidence_status": dict(sorted(Counter(item.extraction.evidence_status.value for item in annotations).items())),
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records)
    _write_text(path, content + ("\n" if content else ""))


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
