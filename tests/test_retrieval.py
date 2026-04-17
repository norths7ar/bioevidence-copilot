from __future__ import annotations

import json
from pathlib import Path

from bioevidence.retrieval.bm25 import bm25_retrieve
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.retrieval.hybrid import hybrid_retrieve
from bioevidence.retrieval.embeddings import DenseRetrievalError
from bioevidence.retrieval.rerank import rerank_candidates
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
        Document(
            pmid="3",
            title="Unrelated biomedical topic",
            abstract="This abstract does not discuss the query terms.",
            journal="Journal C",
            year=2022,
        ),
    ]


def test_load_local_documents_reads_processed_jsonl(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "alpha.documents.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pmid": "1",
                        "title": "Local title",
                        "abstract": "Local abstract",
                        "journal": "Local Journal",
                        "year": 2024,
                    }
                ),
                json.dumps(
                    {
                        "pmid": "1",
                        "title": "Duplicate title",
                        "abstract": "Duplicate abstract",
                        "journal": "Duplicate Journal",
                        "year": 2023,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    documents = load_local_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].pmid == "1"
    assert documents[0].title == "Local title"


def test_bm25_retrieve_orders_relevant_documents_first():
    query = Query(text="asthma corticosteroids")
    documents = _build_documents()

    candidates = bm25_retrieve(query, documents=documents)

    assert [candidate.document.pmid for candidate in candidates] == ["1", "2"]
    assert candidates[0].score >= candidates[1].score
    assert [candidate.rank for candidate in candidates] == [1, 2]


def test_hybrid_retrieve_merges_sources_and_rerank_is_stable(monkeypatch):
    query = Query(text="asthma corticosteroids")
    documents = _build_documents()

    bm25_candidates = [
        RetrievedCandidate(document=documents[0], score=3.0, rank=1),
        RetrievedCandidate(document=documents[1], score=1.5, rank=2),
    ]
    dense_candidates = [
        RetrievedCandidate(document=documents[1], score=0.9, rank=1),
        RetrievedCandidate(document=documents[2], score=0.8, rank=2),
    ]

    monkeypatch.setattr("bioevidence.retrieval.hybrid.bm25_retrieve", lambda *args, **kwargs: bm25_candidates)
    monkeypatch.setattr("bioevidence.retrieval.hybrid.dense_retrieve", lambda *args, **kwargs: dense_candidates)

    merged_candidates = hybrid_retrieve(query, documents=documents)
    reranked_candidates = rerank_candidates(
        [
            RetrievedCandidate(document=documents[1], score=0.5, rank=1),
            RetrievedCandidate(document=documents[0], score=0.5, rank=2),
        ]
    )

    assert [candidate.document.pmid for candidate in merged_candidates] == ["1", "2"]
    assert merged_candidates[0].score > merged_candidates[1].score
    assert [candidate.rank for candidate in merged_candidates] == [1, 2]
    assert [candidate.document.pmid for candidate in reranked_candidates] == ["1", "2"]
    assert [candidate.rank for candidate in reranked_candidates] == [1, 2]


def test_hybrid_retrieve_falls_back_when_dense_is_unavailable(monkeypatch):
    query = Query(text="asthma corticosteroids")
    documents = _build_documents()

    bm25_candidates = [
        RetrievedCandidate(document=documents[0], score=3.0, rank=1),
    ]

    monkeypatch.setattr("bioevidence.retrieval.hybrid.bm25_retrieve", lambda *args, **kwargs: bm25_candidates)
    monkeypatch.setattr(
        "bioevidence.retrieval.hybrid.dense_retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(DenseRetrievalError("dense backend unavailable")),
    )

    merged_candidates = hybrid_retrieve(query, documents=documents)

    assert [candidate.document.pmid for candidate in merged_candidates] == ["1"]
    assert merged_candidates[0].score == 0.7
