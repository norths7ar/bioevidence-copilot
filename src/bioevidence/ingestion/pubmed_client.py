from __future__ import annotations

from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query


def search_pubmed(query: Query) -> list[Document]:
    """Placeholder PubMed search hook."""
    _ = query
    return []
