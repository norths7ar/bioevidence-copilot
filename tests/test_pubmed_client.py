from bioevidence.ingestion.pubmed_client import search_pubmed
from bioevidence.schemas.query import Query


def test_search_pubmed_returns_documents_list():
    results = search_pubmed(Query(text="alpha"))

    assert isinstance(results, list)
    assert results == []
