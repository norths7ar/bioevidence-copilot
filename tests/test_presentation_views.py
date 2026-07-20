from bioevidence.agent.state import AgentState
from bioevidence.workflows import AgentPlanningStep, AgentWorkflowResult, WorkflowResult
from bioevidence.presentation import (
    build_agent_comparison_payload,
    build_agent_report_payload,
    build_evidence_csv,
    build_markdown_report,
    build_result_view,
)
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
            "retrieval_coverage": {
                "baseline_unique_pmids": ["111"],
                "agent_unique_pmids": ["111"],
                "new_pmids_over_baseline": [],
                "overlap_pmids": ["111"],
            },
        },
        planning_steps=(
            AgentPlanningStep(
                iteration=0,
                existing_queries=("asthma corticosteroids",),
                proposed_queries=tuple(),
                accepted_queries=tuple(),
                rationale="Planning skipped because baseline evidence was sufficient.",
                source="skipped",
            ),
        ),
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
    assert payload["trace"]["original_query"] == "asthma corticosteroids"
    assert payload["trace"]["planning_steps"][0]["source"] == "skipped"
    assert payload["trace"]["retrieval_coverage"]["overlap_pmids"] == ["111"]


def test_build_agent_report_payload_stores_evidence_once() -> None:
    payload = build_agent_report_payload(_agent_result())

    assert payload["schema_version"] == 1
    assert payload["baseline"]["evidence_pmids"] == ["111"]
    assert payload["agent"]["evidence_pmids"] == ["111"]
    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["cited_by"] == ["baseline", "agent"]
    assert payload["comparison"]["new_evidence_pmids"] == []


def test_build_markdown_report_includes_answers_and_trace():
    payload = build_agent_comparison_payload(_agent_result())

    report = build_markdown_report(payload)

    assert "# BioEvidence Copilot Report" in report
    assert "## Baseline Answer" in report
    assert "## Agent Answer" in report
    assert "Planning skipped because baseline evidence was sufficient." in report
    assert "| 111 | 2024 | Journal A | 0.92 | Corticosteroids for asthma control |" in report


def test_build_evidence_csv_flattens_entities():
    payload = build_agent_comparison_payload(_agent_result())

    csv_text = build_evidence_csv(payload["agent"]["evidence_table"])

    assert csv_text.splitlines()[0] == "pmid,year,journal,relevance_score,entities,title,summary"
    assert "111,2024,Journal A,0.92,asthma,Corticosteroids for asthma control" in csv_text
