import json
from pathlib import Path

import pytest

from bioevidence.evaluation.extraction_dataset import AnnotationStatus, load_extraction_annotations
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.schemas.document import Document


def test_tracked_extraction_pilot_is_valid_and_remains_draft() -> None:
    documents = load_local_documents(Path("data/corpora/demo"))

    annotations = load_extraction_annotations(
        Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
        documents,
    )

    assert len(annotations) == 20
    assert {annotation.annotation_status for annotation in annotations} == {AnnotationStatus.DRAFT}
    assert sum(annotation.extraction.evidence_status == "direct" for annotation in annotations) == 3
    assert sum(annotation.extraction.evidence_status == "indirect" for annotation in annotations) == 10
    assert sum(annotation.extraction.evidence_status == "none" for annotation in annotations) == 7


def test_tracked_extraction_expansion_is_valid_and_covers_all_candidates() -> None:
    documents = load_local_documents(Path("data/corpora/demo"))
    annotations = load_extraction_annotations(
        Path("data/evaluations/evidence_extraction/expansion_annotations.v1.jsonl"),
        documents,
    )
    candidates = [
        json.loads(line)
        for line in Path("data/evaluations/evidence_extraction/expansion_candidates.v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(annotations) == 40
    assert {(annotation.query, annotation.document.pmid) for annotation in annotations} == {
        (candidate["query"], candidate["pmid"]) for candidate in candidates
    }
    assert sum(annotation.extraction.evidence_status == "direct" for annotation in annotations) == 5
    assert sum(annotation.extraction.evidence_status == "indirect" for annotation in annotations) == 19
    assert sum(annotation.extraction.evidence_status == "none" for annotation in annotations) == 16


def test_extraction_dataset_rejects_non_verbatim_evidence_span(tmp_path: Path) -> None:
    dataset_path = tmp_path / "annotations.jsonl"
    payload = {
        "id": "example-1",
        "query": "example query",
        "pmid": "1",
        "annotation_status": "draft",
        "extraction": {
            "evidence_status": "direct",
            "study_design": "cohort",
            "population_or_system": "adults",
            "intervention_or_exposure": "an intervention",
            "comparator": None,
            "outcomes": [
                {
                    "name": "an outcome",
                    "direction": "decreased",
                    "result_text": "The outcome decreased.",
                    "evidence_span": "This sentence is not in the abstract.",
                }
            ],
            "evidence_summary": "The intervention was associated with the outcome.",
        },
    }
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_span must be copied verbatim"):
        load_extraction_annotations(
            dataset_path,
            [Document(pmid="1", abstract="The outcome decreased.")],
        )


def test_extraction_dataset_rejects_unknown_pmid(tmp_path: Path) -> None:
    dataset_path = tmp_path / "annotations.jsonl"
    payload = {
        "id": "example-1",
        "query": "example query",
        "pmid": "missing",
        "annotation_status": "draft",
        "extraction": {
            "evidence_status": "none",
            "study_design": "not_reported",
            "population_or_system": None,
            "intervention_or_exposure": None,
            "comparator": None,
            "outcomes": [],
            "evidence_summary": None,
        },
    }
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="is not present in the supplied corpus"):
        load_extraction_annotations(dataset_path, [])
