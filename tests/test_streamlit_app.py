from pathlib import Path

import app.streamlit_app as streamlit_app
from bioevidence.agent.state import AgentState
from bioevidence.agent.workflow import AgentWorkflowResult, WorkflowResult
from bioevidence.config import Settings
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        embedding_cache_dir=tmp_path / "cache",
        agent_api_key="test-agent-key",
        agent_base_url="https://example.invalid/v1",
        agent_max_iterations=3,
        agent_max_output_tokens=256,
        agent_min_relevance_score=0.6,
        agent_min_unique_pmids=3,
        agent_model="test-model",
        agent_temperature=0.0,
        log_level="INFO",
        pubmed_email="",
        pubmed_tool_name="BioEvidence Copilot",
        embedding_api_key="test-embedding-key",
        embedding_base_url="https://example.invalid/v1",
        embedding_model="text-embedding-v4",
        embedding_dimensions=1024,
    )


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


def test_streamlit_entrypoint_imports_without_side_effects():
    assert hasattr(streamlit_app, "main")


def test_load_demo_payload_builds_comparison_view(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(streamlit_app, "load_settings", lambda: settings)
    monkeypatch.setattr(streamlit_app, "run_agent_workflow", lambda query, data_dir=None, settings=None: _agent_result())

    payload = streamlit_app.load_demo_payload("asthma corticosteroids", data_dir=str(tmp_path))

    assert payload["baseline"]["query"] == "asthma corticosteroids"
    assert payload["agent"]["retrieval_source"] == "agent:local_corpus"
    assert payload["agent_notice"] is None
