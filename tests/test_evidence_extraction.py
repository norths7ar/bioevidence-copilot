from __future__ import annotations

from bioevidence.extraction.evidence_extractor import extract_evidence
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.query import Query


def _build_documents() -> list[Document]:
    return [
        Document(
            pmid="1",
            title="Corticosteroids for asthma control",
            abstract="Corticosteroids reduce asthma exacerbations and improve control.",
            journal="Journal A",
            year=2024,
        ),
        Document(
            pmid="2",
            title="Asthma management in children",
            abstract="This study discusses pediatric asthma care.",
            journal="Journal B",
            year=2023,
        ),
    ]


def test_extract_evidence_from_documents_builds_structured_rows():
    query = Query(text="asthma corticosteroids")

    records = extract_evidence(query, _build_documents())

    assert len(records) == 2
    assert records[0].pmid == "1"
    assert records[0].title == "Corticosteroids for asthma control"
    assert records[0].entities == ("asthma", "corticosteroids")
    assert records[0].summary == "Corticosteroids reduce asthma exacerbations and improve control"
    assert records[0].relevance_score > records[1].relevance_score


def test_extract_evidence_from_candidates_uses_rank_and_score():
    query = Query(text="asthma corticosteroids")
    documents = _build_documents()
    candidates = [
        RetrievedCandidate(document=documents[1], score=0.2, rank=2),
        RetrievedCandidate(document=documents[0], score=0.9, rank=1),
    ]

    records = extract_evidence(query, candidates)

    assert [record.pmid for record in records] == ["2", "1"]
    assert records[1].relevance_score > records[0].relevance_score
    assert records[1].entities == ("asthma", "corticosteroids")

