import json
from pathlib import Path

import pytest

from bioevidence.evaluation.extraction_dataset import AnnotationStatus, ExtractionAnnotation, load_extraction_annotations
from bioevidence.evaluation.extraction_sft import (
    SplitRatios,
    assign_pmid_splits,
    build_chat_example,
    write_sft_dataset,
)
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import EvidenceStatus, ModelEvidenceExtraction, StudyDesign
from bioevidence.retrieval.corpus import load_local_documents


def test_assign_pmid_splits_is_deterministic_and_keeps_documents_together() -> None:
    annotations = [_annotation("a", "1"), _annotation("b", "1"), _annotation("c", "2"), _annotation("d", "3")]

    assignments = assign_pmid_splits(annotations, ratios=SplitRatios(0.5, 0.25, 0.25), seed=7)
    reversed_assignments = assign_pmid_splits(list(reversed(annotations)), ratios=SplitRatios(0.5, 0.25, 0.25), seed=7)

    assert assignments == reversed_assignments
    assert set(assignments) == {"1", "2", "3"}
    assert sum(split == "train" for split in assignments.values()) == 1
    assert sum(split == "dev" for split in assignments.values()) == 1
    assert sum(split == "test" for split in assignments.values()) == 1


def test_build_chat_example_uses_runtime_prompt_and_compact_json_target() -> None:
    annotation = _annotation("example", "1")

    example = build_chat_example(annotation, split="train", source_dataset="pilot.jsonl")

    assert [message["role"] for message in example["messages"]] == ["system", "user", "assistant"]
    assert "JSON_SCHEMA:" in example["messages"][1]["content"]
    assert json.loads(example["messages"][2]["content"]) == annotation.extraction.model_dump(mode="json")
    assert example["metadata"]["pmid"] == "1"
    assert example["metadata"]["split"] == "train"


def test_write_sft_dataset_emits_manifest_without_pmid_leakage(tmp_path: Path) -> None:
    annotations = [_annotation("a", "1"), _annotation("b", "1"), _annotation("c", "2"), _annotation("d", "3")]

    manifest = write_sft_dataset(
        annotations,
        tmp_path,
        source_dataset="pilot.jsonl",
        ratios=SplitRatios(0.5, 0.25, 0.25),
        seed=7,
        source_metadata={"label_source": "model-assisted annotation"},
    )

    split_pmids = [set(manifest["splits"][name]["pmids"]) for name in ("train", "dev", "test")]
    assert not (split_pmids[0] & split_pmids[1] or split_pmids[0] & split_pmids[2] or split_pmids[1] & split_pmids[2])
    assert sum(manifest["splits"][name]["rows"] for name in ("train", "dev", "test")) == 4
    assert manifest["source_metadata"]["label_source"] == "model-assisted annotation"
    assert all((tmp_path / f"{name}.jsonl").exists() for name in ("train", "dev", "test"))
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitRatios(0.8, 0.2, 0.2)


def test_tracked_pilot_manifest_matches_current_annotations(tmp_path: Path) -> None:
    dataset = Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl")
    metadata = json.loads(
        Path("data/evaluations/evidence_extraction/pilot_dataset_metadata.json").read_text(encoding="utf-8")
    )
    annotations = load_extraction_annotations(dataset, load_local_documents(Path("data/corpora/demo")))

    manifest = write_sft_dataset(
        annotations,
        tmp_path,
        source_dataset=dataset.as_posix(),
        source_metadata=metadata,
    )

    tracked = json.loads(
        Path("data/evaluations/evidence_extraction/pilot_split_manifest.json").read_text(encoding="utf-8")
    )
    assert tracked == manifest


def _annotation(annotation_id: str, pmid: str) -> ExtractionAnnotation:
    return ExtractionAnnotation(
        id=annotation_id,
        query="example query",
        document=Document(pmid=pmid, title="Example", abstract="Example abstract."),
        extraction=ModelEvidenceExtraction(
            evidence_status=EvidenceStatus.NONE,
            study_design=StudyDesign.NOT_REPORTED,
            population_or_system=None,
            intervention_or_exposure=None,
            comparator=None,
            outcomes=(),
            evidence_summary=None,
        ),
        annotation_status=AnnotationStatus.DRAFT,
    )
