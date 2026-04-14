import bioevidence.agent.workflow as workflow_module
from bioevidence.agent.workflow import run_workflow
from bioevidence.schemas.query import Query


def test_run_workflow_returns_answer_bundle(monkeypatch):
    monkeypatch.setattr(workflow_module, "search_pubmed", lambda query: [])
    monkeypatch.setattr(workflow_module, "hybrid_retrieve", lambda query: [])

    answer = run_workflow(Query(text="delta"))

    assert answer.answer_text
    assert answer.citations == ()
