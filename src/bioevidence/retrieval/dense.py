from __future__ import annotations

from bioevidence.schemas.document import RetrievedCandidate
from bioevidence.schemas.query import Query


def dense_retrieve(query: Query) -> list[RetrievedCandidate]:
    _ = query
    return []
