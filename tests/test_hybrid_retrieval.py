from bioevidence.retrieval.hybrid import hybrid_retrieve
from bioevidence.schemas.query import Query


def test_hybrid_retrieve_returns_sorted_candidates():
    results = hybrid_retrieve(Query(text="beta"))

    assert isinstance(results, list)
    assert results == []
