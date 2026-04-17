from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bioevidence.config import Settings
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.retrieval.scoring import bm25_score, document_tokens, tokenize_text
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.query import Query


def bm25_retrieve(
    query: Query,
    *,
    documents: Sequence[Document] | None = None,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> list[RetrievedCandidate]:
    documents = list(documents) if documents is not None else load_local_documents(data_dir, settings=settings)
    if not documents:
        return []

    query_tokens = tokenize_text(query.text)
    if not query_tokens:
        return []

    document_token_lists = [document_tokens(document) for document in documents]
    scores = bm25_score(query_tokens, document_token_lists)
    ranked_documents = [
        (document, score)
        for document, score in zip(documents, scores, strict=False)
        if score > 0
    ]
    ranked_documents.sort(key=lambda item: (-item[1], item[0].pmid))
    return [
        RetrievedCandidate(document=document, score=score, rank=index + 1)
        for index, (document, score) in enumerate(ranked_documents)
    ]
