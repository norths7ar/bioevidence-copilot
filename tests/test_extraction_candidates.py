import json
from pathlib import Path

import pytest

from bioevidence.evaluation.extraction_candidates import (
    CandidateTopic,
    ExtractionCandidate,
    build_annotation_prompt_records,
    build_candidate_manifest,
    load_candidate_topics,
    select_expansion_candidates,
)
from bioevidence.evaluation.extraction_dataset import AnnotationStatus, ExtractionAnnotation
from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import EvidenceStatus, ModelEvidenceExtraction, StudyDesign


def test_select_expansion_candidates_stratifies_and_excludes_existing_pairs() -> None:
    topics = [
        CandidateTopic(query="asthma trial", pmids=("1", "2", "3", "4")),
        CandidateTopic(query="diabetes trial", pmids=("5", "6", "7", "8")),
    ]
    documents = [Document(pmid=str(index), title=title, abstract="trial") for index, title in enumerate(
        ["asthma", "asthma steroid", "asthma biologic", "airway", "diabetes", "metformin", "glucose", "insulin"],
        start=1,
    )]
    existing = [_annotation("existing", "asthma trial", documents[0])]

    candidates = select_expansion_candidates(
        topics,
        documents,
        existing,
        high_per_topic=1,
        broad_per_topic=1,
        hard_negative_per_topic=1,
    )

    assert len(candidates) == 6
    assert ("asthma trial", "1") not in {(candidate.query, candidate.document.pmid) for candidate in candidates}
    assert {candidate.selection_band for candidate in candidates} == {
        "topic_high",
        "topic_broad",
        "cross_topic_hard_negative",
    }
    assert len({(candidate.query, candidate.document.pmid) for candidate in candidates}) == 6


def test_select_expansion_candidates_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        select_expansion_candidates([], [], [], high_per_topic=-1)


def test_build_candidate_manifest_records_source_hash(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")

    manifest = build_candidate_manifest([], source_corpus=corpus, existing_annotations=Path("pilot.jsonl"))

    assert manifest["candidate_pairs"] == 0
    assert len(manifest["source_corpus_sha256"]) == 64
    assert manifest["selection"]["high_per_topic"] == 4


def test_build_candidate_manifest_records_additional_annotation_sources(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")

    manifest = build_candidate_manifest(
        [],
        source_corpus=corpus,
        existing_annotations=Path("pilot.jsonl"),
        additional_annotations=(Path("expansion.jsonl"),),
    )

    assert manifest["existing_annotations"] == "pilot.jsonl"
    assert manifest["additional_annotations"] == ["expansion.jsonl"]


def test_candidate_manifest_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    lf_corpus = tmp_path / "lf.jsonl"
    crlf_corpus = tmp_path / "crlf.jsonl"
    lf_corpus.write_bytes(b'{"pmid":"1"}\n{"pmid":"2"}\n')
    crlf_corpus.write_bytes(b'{"pmid":"1"}\r\n{"pmid":"2"}\r\n')

    lf_manifest = build_candidate_manifest([], source_corpus=lf_corpus, existing_annotations=Path("pilot.jsonl"))
    crlf_manifest = build_candidate_manifest(
        [],
        source_corpus=crlf_corpus,
        existing_annotations=Path("pilot.jsonl"),
    )

    assert lf_manifest["source_corpus_sha256"] == crlf_manifest["source_corpus_sha256"]


def test_build_annotation_prompt_records_uses_runtime_prompt() -> None:
    candidate = ExtractionCandidate(
        id="candidate-1",
        query="asthma trial",
        document=Document(pmid="1", title="Title", abstract="Abstract"),
        source_topic="asthma trial",
        selection_band="topic_high",
        bm25_score=1.0,
        bm25_rank=1,
    )

    record = build_annotation_prompt_records([candidate])[0]

    assert [message["role"] for message in record["messages"]] == ["system", "user"]
    assert "JSON_SCHEMA:" in record["messages"][1]["content"]
    assert "ABSTRACT:\nAbstract" in record["messages"][1]["content"]
    assert record["metadata"]["label_status"] == "unlabeled"


def test_tracked_expansion_candidates_match_current_corpus_and_pilot() -> None:
    corpus = Path("data/corpora/demo/processed/demo.documents.jsonl")
    annotations_path = Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl")
    documents = load_local_documents(Path("data/corpora/demo"))
    annotations = load_extraction_annotations(annotations_path, documents)

    candidates = select_expansion_candidates(
        load_candidate_topics(Path("data/corpora/demo/processed/demo.manifest.json")),
        documents,
        annotations,
    )
    tracked_records = [
        json.loads(line)
        for line in Path("data/evaluations/evidence_extraction/expansion_candidates.v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tracked_manifest = json.loads(
        Path("data/evaluations/evidence_extraction/expansion_candidates.v1.manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert tracked_records == [candidate.to_record() for candidate in candidates]
    assert tracked_manifest == build_candidate_manifest(
        candidates,
        source_corpus=corpus,
        existing_annotations=annotations_path,
    )


def test_tracked_v2_candidates_exclude_all_v1_annotations() -> None:
    corpus = Path("data/corpora/demo/processed/demo.documents.jsonl")
    pilot_path = Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl")
    expansion_path = Path("data/evaluations/evidence_extraction/expansion_annotations.v1.jsonl")
    documents = load_local_documents(Path("data/corpora/demo"))
    annotations = load_extraction_annotations(pilot_path, documents)
    annotations.extend(load_extraction_annotations(expansion_path, documents))

    candidates = select_expansion_candidates(
        load_candidate_topics(Path("data/corpora/demo/processed/demo.manifest.json")),
        documents,
        annotations,
        high_per_topic=8,
        broad_per_topic=4,
        hard_negative_per_topic=0,
    )
    tracked_records = [
        json.loads(line)
        for line in Path("data/evaluations/evidence_extraction/expansion_candidates.v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tracked_manifest = json.loads(
        Path("data/evaluations/evidence_extraction/expansion_candidates.v2.manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert tracked_records == [candidate.to_record() for candidate in candidates]
    assert tracked_manifest == build_candidate_manifest(
        candidates,
        source_corpus=corpus,
        existing_annotations=pilot_path,
        additional_annotations=(expansion_path,),
        high_per_topic=8,
        broad_per_topic=4,
        hard_negative_per_topic=0,
    )


def _annotation(annotation_id: str, query: str, document: Document) -> ExtractionAnnotation:
    return ExtractionAnnotation(
        id=annotation_id,
        query=query,
        document=document,
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
