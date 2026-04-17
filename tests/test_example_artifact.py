from __future__ import annotations

import json
from pathlib import Path


def test_milestone3_example_artifact_has_expected_shape():
    path = Path("examples/milestone3_evidence_example.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["retrieval_source"] == "local_corpus"
    assert data["query"]
    assert data["answer"]
    assert data["citations"]
    assert isinstance(data["evidence_table"], list)
    assert data["evidence_table"][0]["pmid"] == "12345678"
    assert data["evidence_table"][0]["entities"] == ["asthma", "corticosteroids"]
    assert "relevance_score" in data["evidence_table"][0]

