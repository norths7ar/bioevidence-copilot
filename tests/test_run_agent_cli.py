from pathlib import Path

import scripts.run_agent as run_agent_script
from bioevidence.agent.state import AgentState
from bioevidence.workflows import AgentWorkflowResult, WorkflowResult
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _agent_result() -> AgentWorkflowResult:
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
        summary=document.abstract,
        relevance_score=0.92,
    )
    answer = AnswerBundle(
        answer_text="Agent answer",
        citations=("111",),
        evidence_records=(evidence_record,),
        rewritten_query="asthma corticosteroids",
    )
    baseline = WorkflowResult(
        query=Query(text="asthma corticosteroids"),
        documents=(document,),
        retrieved_candidates=(candidate,),
        evidence_records=(evidence_record,),
        answer=answer,
        source="local_corpus",
    )
    return AgentWorkflowResult(
        query=baseline.query,
        baseline=baseline,
        branch_results=tuple(),
        documents=baseline.documents,
        retrieved_candidates=baseline.retrieved_candidates,
        evidence_records=baseline.evidence_records,
        answer=answer,
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


def test_run_agent_cli_prints_json_and_writes_output(tmp_path: Path, monkeypatch, capsys):
    output_path = tmp_path / "agent-report.json"
    monkeypatch.setattr(run_agent_script, "run_agent_workflow", lambda query, data_dir=None, settings=None: _agent_result())

    exit_code = run_agent_script.main([
        "--query",
        "asthma corticosteroids",
        "--output",
        str(output_path),
    ])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query": "asthma corticosteroids"' in captured.out
    assert output_path.exists()
    assert '"branch_count": 0' in output_path.read_text(encoding="utf-8")
