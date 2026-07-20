from dataclasses import replace
import json
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


def test_run_agent_cli_prints_summary_and_writes_compact_output(tmp_path: Path, monkeypatch, capsys):
    output_path = tmp_path / "agent-report.json"
    monkeypatch.setattr(
        run_agent_script,
        "run_agent_workflow",
        lambda query, data_dir=None, settings=None, trace_recorder=None: _agent_result(),
    )

    exit_code = run_agent_script.main([
        "--query",
        "asthma corticosteroids",
        "--output",
        str(output_path),
    ])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run ID:" in captured.out
    assert "Evidence: baseline 1 -> agent 1" in captured.out
    assert output_path.exists()
    assert '"schema_version": 1' in output_path.read_text(encoding="utf-8")
    assert '"branch_count": 0' in output_path.read_text(encoding="utf-8")


def test_run_agent_cli_writes_run_artifacts(tmp_path: Path, monkeypatch) -> None:
    def fake_workflow(query, data_dir=None, settings=None, trace_recorder=None):
        del query, data_dir, settings
        trace_recorder.emit("run_started", top_k=10)
        trace_recorder.emit("run_completed", stop_reason="sufficient_evidence")
        return replace(
            _agent_result(),
            run_id=trace_recorder.run_id,
            trace_events=trace_recorder.events(),
        )

    monkeypatch.setattr(run_agent_script, "run_agent_workflow", fake_workflow)

    exit_code = run_agent_script.main(
        [
            "--query",
            "asthma corticosteroids",
            "--artifacts-dir",
            str(tmp_path),
            "--debug",
        ]
    )

    run_directories = list(tmp_path.iterdir())
    assert exit_code == 0
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert (run_directory / "run.log").exists()
    assert (run_directory / "report.json").exists()
    assert (run_directory / "trace.jsonl").exists()
    assert (run_directory / "debug.json").exists()
    report = json.loads((run_directory / "report.json").read_text(encoding="utf-8"))
    trace_lines = (run_directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert report["run"]["run_id"]
    assert [json.loads(line)["event"] for line in trace_lines] == ["run_started", "run_completed"]
