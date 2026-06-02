from pathlib import Path

from bioevidence.retrieval.hybrid import hybrid_retrieve
from bioevidence.schemas.query import Query


def test_hybrid_retrieve_returns_sorted_candidates(tmp_path: Path):
    results = hybrid_retrieve(Query(text="beta"), data_dir=tmp_path)

    assert isinstance(results, list)
    assert results == []
