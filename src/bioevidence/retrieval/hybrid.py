from __future__ import annotations

from bioevidence.retrieval.bm25 import bm25_retrieve
from bioevidence.retrieval.dense import dense_retrieve
from bioevidence.schemas.document import RetrievedCandidate
from bioevidence.schemas.query import Query


def hybrid_retrieve(query: Query) -> list[RetrievedCandidate]:
    candidates = bm25_retrieve(query) + dense_retrieve(query)
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
