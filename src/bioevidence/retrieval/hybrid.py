from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from bioevidence.config import Settings
from bioevidence.retrieval.bm25 import bm25_retrieve
from bioevidence.retrieval.dense import dense_retrieve
from bioevidence.retrieval.embeddings import DenseRetrievalError
from bioevidence.retrieval.scoring import normalize_scores
from bioevidence.schemas.document import RetrievedCandidate
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query


LOGGER = logging.getLogger(__name__)


def hybrid_retrieve(
    query: Query,
    *,
    documents: Sequence[Document] | None = None,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> list[RetrievedCandidate]:
    lexical_candidates = bm25_retrieve(query, documents=documents, data_dir=data_dir, settings=settings)
    try:
        dense_candidates = dense_retrieve(query, documents=documents, data_dir=data_dir, settings=settings)
    except DenseRetrievalError as exc:
        LOGGER.warning("Dense retrieval unavailable; falling back to lexical-only ranking: %s", exc)
        dense_candidates = []
    if not lexical_candidates and not dense_candidates:
        return []

    lexical_scores = normalize_scores([candidate.score for candidate in lexical_candidates])
    dense_scores = normalize_scores([candidate.score for candidate in dense_candidates])

    lexical_by_pmid = {
        candidate.document.pmid: (candidate.document, normalized_score)
        for candidate, normalized_score in zip(lexical_candidates, lexical_scores, strict=False)
    }
    dense_by_pmid = {
        candidate.document.pmid: (candidate.document, normalized_score)
        for candidate, normalized_score in zip(dense_candidates, dense_scores, strict=False)
    }

    pmids = sorted(set(lexical_by_pmid) | set(dense_by_pmid))
    merged_candidates: list[RetrievedCandidate] = []
    for pmid in pmids:
        lexical_document, lexical_score = lexical_by_pmid.get(pmid, (None, 0.0))
        dense_document, dense_score = dense_by_pmid.get(pmid, (None, 0.0))
        document = lexical_document or dense_document
        if document is None:
            continue
        combined_score = (0.7 * lexical_score) + (0.3 * dense_score)
        if combined_score <= 0:
            continue
        merged_candidates.append(RetrievedCandidate(document=document, score=combined_score))

    merged_candidates.sort(key=lambda candidate: (-candidate.score, candidate.document.pmid))
    return [
        RetrievedCandidate(document=candidate.document, score=candidate.score, rank=index + 1)
        for index, candidate in enumerate(merged_candidates)
    ]
