from __future__ import annotations

import json
from hashlib import sha256
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bioevidence.config import Settings, load_settings
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.retrieval.embeddings import DenseRetrievalError, create_embedding_client, embed_documents, embed_texts
from bioevidence.utils.io import load_json, save_json
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.query import Query


DEFAULT_CACHE_FILENAME = "dense_embeddings.cache.json"


def dense_retrieve(
    query: Query,
    *,
    documents: Sequence[Document] | None = None,
    data_dir: Path | None = None,
    settings: Settings | None = None,
    client=None,
) -> list[RetrievedCandidate]:
    settings = settings or load_settings()
    documents = list(documents) if documents is not None else load_local_documents(data_dir, settings=settings)
    if not documents:
        return []

    if not query.text.strip():
        return []

    cache_path = _cache_path(data_dir, settings)
    corpus_signature = _corpus_signature(documents)
    cache = _load_cache(cache_path)
    if not _cache_matches(cache, corpus_signature, settings):
        client = client or create_embedding_client(settings)
        document_embeddings = embed_documents(documents, client=client, settings=settings)
        cache = {
            "corpus_signature": corpus_signature,
            "model": settings.qwen_embedding_model,
            "dimensions": settings.qwen_embedding_dimensions,
            "documents": [
                {
                    "pmid": document.pmid,
                    "embedding": embedding,
                }
                for document, embedding in zip(documents, document_embeddings, strict=False)
            ],
        }
        save_json(cache, cache_path)

    if not cache:
        return []

    document_embeddings_by_pmid = {
        str(entry["pmid"]): entry["embedding"]
        for entry in cache["documents"]
        if isinstance(entry, dict) and "pmid" in entry and "embedding" in entry
    }
    document_embeddings = [document_embeddings_by_pmid.get(document.pmid) for document in documents]
    if any(embedding is None for embedding in document_embeddings):
        client = client or create_embedding_client(settings)
        document_embeddings = embed_documents(documents, client=client, settings=settings)
        cache = {
            "corpus_signature": corpus_signature,
            "model": settings.qwen_embedding_model,
            "dimensions": settings.qwen_embedding_dimensions,
            "documents": [
                {
                    "pmid": document.pmid,
                    "embedding": embedding,
                }
                for document, embedding in zip(documents, document_embeddings, strict=False)
            ],
        }
        save_json(cache, cache_path)

    if not document_embeddings:
        return []

    client = client or create_embedding_client(settings)
    query_embedding = embed_texts([query.text], client=client, settings=settings)[0]
    scores = _cosine_similarity_batch(query_embedding, document_embeddings)
    ranked_documents = sorted(
        zip(documents, scores, strict=False),
        key=lambda item: (-item[1], item[0].pmid),
    )
    return [
        RetrievedCandidate(document=document, score=float(score), rank=index + 1)
        for index, (document, score) in enumerate(ranked_documents)
    ]


def _cache_path(base_dir: Path | None, settings: Settings) -> Path:
    if base_dir is not None:
        return Path(base_dir) / "cache" / DEFAULT_CACHE_FILENAME
    return Path(settings.embedding_cache_dir) / DEFAULT_CACHE_FILENAME


def _corpus_signature(documents: Sequence[Document]) -> str:
    payload = [
        {
            "pmid": document.pmid,
            "title": document.title,
            "abstract": document.abstract,
            "journal": document.journal,
            "year": document.year,
            "authors": list(document.authors),
            "source": document.source,
        }
        for document in sorted(documents, key=lambda item: item.pmid)
    ]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        cache = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return cache if isinstance(cache, dict) else None


def _cache_matches(cache: dict[str, object] | None, corpus_signature: str, settings: Settings) -> bool:
    if not cache:
        return False
    return (
        cache.get("corpus_signature") == corpus_signature
        and cache.get("model") == settings.qwen_embedding_model
        and cache.get("dimensions") == settings.qwen_embedding_dimensions
        and isinstance(cache.get("documents"), list)
    )


def _cosine_similarity_batch(query_embedding: Sequence[float], document_embeddings: Sequence[Sequence[float]]) -> list[float]:
    query_vector = np.asarray(query_embedding, dtype=np.float32)
    document_matrix = np.asarray(document_embeddings, dtype=np.float32)
    if query_vector.ndim != 1 or document_matrix.ndim != 2:
        raise DenseRetrievalError("Invalid embedding shapes received from the embedding backend")
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return [0.0] * len(document_embeddings)
    document_norms = np.linalg.norm(document_matrix, axis=1)
    denominator = document_norms * query_norm
    numerator = document_matrix @ query_vector
    similarities = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )
    return [float(score) for score in similarities]
