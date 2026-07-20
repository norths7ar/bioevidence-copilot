from __future__ import annotations

import json

from bioevidence.agent.state import AgentState
from bioevidence.workflows import AgentWorkflowResult, WorkflowResult
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _workflow_result(query_text: str = "asthma corticosteroids") -> WorkflowResult:
    document = Document(
        pmid="111",
        title="Corticosteroids for asthma",
        abstract="Corticosteroids reduced exacerbations.",
        journal="Journal",
        year=2024,
    )
    candidate = RetrievedCandidate(document=document, score=0.91, rank=1)
    evidence = EvidenceRecord(
        pmid="111",
        title=document.title,
        year=document.year,
        journal=document.journal,
        entities=("asthma",),
        summary=document.abstract,
        relevance_score=0.91,
    )
    return WorkflowResult(
        query=Query(text=query_text, top_k=5),
        documents=(document,),
        retrieved_candidates=(candidate,),
        evidence_records=(evidence,),
        answer=AnswerBundle(
            answer_text="Corticosteroids reduced exacerbations [111].",
            citations=("111",),
            evidence_records=(evidence,),
            rewritten_query=query_text,
        ),
        source="local_corpus",
    )


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from interfaces.api.main import app

    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_agent_stream_endpoint_emits_ndjson(monkeypatch):
    from fastapi.testclient import TestClient

    import interfaces.api.main as api_main

    monkeypatch.setattr(
        api_main,
        "stream_agent_workflow",
        lambda query, data_dir=None: iter(
            [
                {"node": "retrieve_baseline"},
                {"node": "discover_graph", "graph_discovery": {"status": "disabled"}},
            ]
        ),
    )
    client = TestClient(api_main.app)

    response = client.post("/api/v1/query/agent/stream", json={"query": "asthma evidence"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = response.text.strip().splitlines()
    assert '"node": "retrieve_baseline"' in lines[0]
    assert '"status": "disabled"' in lines[1]


def test_agent_stream_endpoint_returns_error_status_before_stream_starts(monkeypatch):
    from fastapi.testclient import TestClient

    import interfaces.api.main as api_main

    def failed_stream(query, data_dir=None):
        del query, data_dir
        raise FileNotFoundError("missing corpus")
        yield

    monkeypatch.setattr(api_main, "stream_agent_workflow", failed_stream)
    client = TestClient(api_main.app)

    response = client.post("/api/v1/query/agent/stream", json={"query": "asthma evidence"})

    assert response.status_code == 400
    assert response.json() == {"detail": "missing corpus"}


def test_agent_stream_endpoint_emits_terminal_error_after_stream_starts(monkeypatch):
    from fastapi.testclient import TestClient

    import interfaces.api.main as api_main

    def failed_stream(query, data_dir=None):
        del query, data_dir
        yield {"node": "retrieve_baseline"}
        raise RuntimeError("bug")

    monkeypatch.setattr(api_main, "stream_agent_workflow", failed_stream)
    client = TestClient(api_main.app)

    response = client.post("/api/v1/query/agent/stream", json={"query": "asthma evidence"})

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.strip().splitlines()]
    assert lines[0] == {"node": "retrieve_baseline"}
    assert lines[1] == {
        "node": "error",
        "error": {"status_code": 500, "detail": "agent workflow failed"},
    }


def test_baseline_endpoint_returns_workflow_shape(monkeypatch):
    from fastapi.testclient import TestClient

    import interfaces.api.main as api_main

    monkeypatch.setattr(api_main, "run_rag_pipeline", lambda query, *, data_dir=None: _workflow_result(query.text))
    client = TestClient(api_main.app)

    response = client.post("/api/v1/query/baseline", json={"query": "asthma corticosteroids", "top_k": 5})

    payload = response.json()
    assert response.status_code == 200
    assert payload["query"] == "asthma corticosteroids"
    assert payload["source"] == "local_corpus"
    assert payload["citations"] == ["111"]
    assert payload["retrieved_papers"][0]["pmid"] == "111"
    assert payload["evidence_table"][0]["pmid"] == "111"


def test_agent_endpoint_returns_trace_shape(monkeypatch):
    from fastapi.testclient import TestClient

    import interfaces.api.main as api_main

    baseline = _workflow_result()
    agent_result = AgentWorkflowResult(
        query=baseline.query,
        baseline=baseline,
        branch_results=tuple(),
        documents=baseline.documents,
        retrieved_candidates=baseline.retrieved_candidates,
        evidence_records=baseline.evidence_records,
        answer=baseline.answer,
        source="agent:local_corpus",
        state=AgentState(query=baseline.query, sufficient=True, stop_reason="sufficient_evidence"),
        comparison={"branch_count": 0, "stop_reason": "sufficient_evidence"},
    )
    monkeypatch.setattr(api_main, "run_agent_workflow", lambda query, *, data_dir=None: agent_result)
    client = TestClient(api_main.app)

    response = client.post("/api/v1/query/agent", json={"query": "asthma corticosteroids"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["source"] == "agent:local_corpus"
    assert payload["baseline"]["source"] == "local_corpus"
    assert payload["state"]["stop_reason"] == "sufficient_evidence"
    assert payload["comparison"]["branch_count"] == 0
    assert payload["trace"]["original_query"] == "asthma corticosteroids"
    assert payload["trace"]["stop"]["reason"] == "sufficient_evidence"


def test_baseline_endpoint_rejects_empty_query():
    from fastapi.testclient import TestClient

    from interfaces.api.main import app

    client = TestClient(app)
    response = client.post("/api/v1/query/baseline", json={"query": ""})

    assert response.status_code == 422
