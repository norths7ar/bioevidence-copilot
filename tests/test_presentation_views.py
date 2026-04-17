from bioevidence.agent.state import AgentState
from bioevidence.agent.workflow import AgentWorkflowResult, WorkflowResult
from bioevidence.presentation import build_agent_comparison_payload, build_result_view
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _workflow_result() -> WorkflowResult:
    document = Document(
        pmid="111",
        title="Corticosteroids for asthma control",
        abstract="Corticosteroids reduce asthma exacerbations.",
        journal="Journal A",
        year=2024,
    )
    candidate = RetrievedCandidate(document=document, score=0.92, rank=1)
    evidence_record = EvidenceRecord(
        pmid="111",
        title=document.title,
        year=document.year,
        journal=document.journal,
        entities=("asthma",),
        summary="Corticosteroids reduce asthma exacerbations.",
        relevance_score=0.92,
    )
    answer = AnswerBundle(
        answer_text="Baseline answer",
        citations=("111",),
        evidence_records=(evidence_record,),
        rewritten_query="asthma corticosteroids",
    )
    return WorkflowResult(
        query=Query(text="asthma corticosteroids"),
        documents=(document,),
        retrieved_candidates=(candidate,),
        evidence_records=(evidence_record,),
        answer=answer,
        source="local_corpus",
    )


def _agent_result() -> AgentWorkflowResult:
    baseline = _workflow_result()
    return AgentWorkflowResult(
        query=baseline.query,
        baseline=baseline,
        branch_results=tuple(),
        documents=baseline.documents,
        retrieved_candidates=baseline.retrieved_candidates,
        evidence_records=baseline.evidence_records,
        answer=baseline.answer,
        source="agent:local_corpus",
        state=AgentState(query=baseline.query, sufficient=True, stop_reason="sufficient_evidence"),
        comparison={
            "branch_count": 0,
            "iterations": 0,
            "unique_pmid_coverage": 1,
            "baseline_unique_pmids": 1,
            "agent_unique_pmids": 1,
            "citation_overlap": ["111"],
            "baseline_citations": ["111"],
            "agent_citations": ["111"],
            "stop_reason": "sufficient_evidence",
            "agent_backend_ready": True,
        },
    )


def test_build_result_view_normalizes_workflow_result():
    view = build_result_view(_workflow_result())

    payload = view.to_dict()
    assert payload["query"] == "asthma corticosteroids"
    assert payload["retrieved_papers"][0]["pmid"] == "111"
    assert payload["evidence_table"][0]["summary"] == "Corticosteroids reduce asthma exacerbations."
    assert payload["citations"] == ["111"]


def test_build_agent_comparison_payload_includes_baseline_and_agent():
    payload = build_agent_comparison_payload(_agent_result())

    assert payload["baseline"]["query"] == "asthma corticosteroids"
    assert payload["agent"]["retrieval_source"] == "agent:local_corpus"
    assert payload["comparison"]["agent_backend_ready"] is True
    assert payload["state"]["stop_reason"] == "sufficient_evidence"
