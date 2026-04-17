from __future__ import annotations

import json
from pathlib import Path

from bioevidence.retrieval.dense import dense_retrieve
from bioevidence.schemas.document import Document
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


def _fake_embed_texts(texts, *, client=None, settings=None):
    del client, settings
    vector_map = {
        "asthma corticosteroids": [1.0, 0.0],
    }
    return [vector_map[text] for text in texts]


def _fake_embed_documents(documents, *, client=None, settings=None):
    del client, settings
    vector_map = {
        "Corticosteroids for asthma control Corticosteroids reduce asthma exacerbations and improve control.": [1.0, 0.0],
        "Asthma management in children This study discusses pediatric asthma care.": [0.7, 0.7],
        "Unrelated biomedical topic This abstract does not discuss the query terms.": [0.0, 1.0],
    }
    return [vector_map[f"{document.title} {document.abstract}"] for document in documents]


def test_dense_retrieve_ranks_by_cosine_similarity(monkeypatch, tmp_path: Path):
    documents = _build_documents()
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_documents", _fake_embed_documents)
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_texts", _fake_embed_texts)

    candidates = dense_retrieve(Query(text="asthma corticosteroids"), documents=documents, data_dir=tmp_path, client=object())

    assert [candidate.document.pmid for candidate in candidates] == ["1", "2", "3"]
    assert candidates[0].score > candidates[1].score > candidates[2].score
    assert [candidate.rank for candidate in candidates] == [1, 2, 3]


def test_dense_retrieve_reuses_cached_embeddings(monkeypatch, tmp_path: Path):
    documents = _build_documents()
    calls: list[tuple[str, ...]] = []

    def fake_embed_texts(texts, *, client=None, settings=None):
        del client, settings
        calls.append(tuple(texts))
        return _fake_embed_texts(texts)

    def fake_embed_documents(documents, *, client=None, settings=None):
        del client, settings
        calls.append(tuple(f"{document.title} {document.abstract}" for document in documents))
        return _fake_embed_documents(documents)

    monkeypatch.setattr("bioevidence.retrieval.dense.embed_documents", fake_embed_documents)
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_texts", fake_embed_texts)

    dense_retrieve(Query(text="asthma corticosteroids"), documents=documents, data_dir=tmp_path, client=object())
    first_run_calls = list(calls)
    calls.clear()
    dense_retrieve(Query(text="asthma corticosteroids"), documents=documents, data_dir=tmp_path, client=object())

    cache_file = tmp_path / "cache" / "dense_embeddings.cache.json"
    assert cache_file.exists()
    assert first_run_calls == [
        tuple(
            [
                "Corticosteroids for asthma control Corticosteroids reduce asthma exacerbations and improve control.",
                "Asthma management in children This study discusses pediatric asthma care.",
                "Unrelated biomedical topic This abstract does not discuss the query terms.",
            ]
        ),
        ("asthma corticosteroids",),
    ]
    assert calls == [("asthma corticosteroids",)]
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cache["model"] == "text-embedding-v4"
    assert cache["dimensions"] == 1024
