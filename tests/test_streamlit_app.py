from pathlib import Path

import interfaces.web.streamlit_app as streamlit_app
from bioevidence.agent.state import AgentState
from bioevidence.workflows import AgentWorkflowResult, WorkflowResult
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
        embedding_batch_size=10,
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
    monkeypatch.setattr(
        streamlit_app, "run_agent_workflow", lambda query, data_dir=None, settings=None: _agent_result()
    )

    payload = streamlit_app.load_demo_payload("asthma corticosteroids", data_dir=str(tmp_path))

    assert payload["baseline"]["query"] == "asthma corticosteroids"
    assert payload["agent"]["retrieval_source"] == "agent:local_corpus"
    assert payload["agent_notice"] is None


def test_build_run_summary_marks_agent_expansion():
    payload = {
        "baseline": {"retrieval_source": "local_corpus"},
        "agent": {"retrieval_source": "agent:local_corpus"},
        "branches": [{"query": "dietary sodium hypertension trial"}],
        "state": {"iterations": 1, "stop_reason": "sufficient_evidence"},
        "comparison": {
            "branch_count": 2,
            "iterations": 1,
            "stop_reason": "sufficient_evidence",
            "agent_backend_ready": True,
        },
    }

    summary = streamlit_app._build_run_summary(payload)

    assert summary == {
        "baseline_source": "local_corpus",
        "agent_source": "agent:local_corpus",
        "agent_status": "expanded",
        "agent_backend": "configured",
        "branch_count": 2,
        "iterations": 1,
        "stop_reason": "sufficient_evidence",
    }


def test_filter_sort_evidence_rows_applies_entity_journal_and_relevance():
    rows = [
        {
            "pmid": "111",
            "year": 2022,
            "journal": "Journal A",
            "entities": ["asthma"],
            "relevance_score": 0.7,
        },
        {
            "pmid": "222",
            "year": 2024,
            "journal": "Journal B",
            "entities": ["diabetes"],
            "relevance_score": 0.95,
        },
        {
            "pmid": "333",
            "year": 2023,
            "journal": "Journal A",
            "entities": ["asthma", "corticosteroids"],
            "relevance_score": 0.91,
        },
    ]

    filtered = streamlit_app._filter_sort_evidence_rows(
        rows,
        selected_entities=["asthma"],
        selected_journal="Journal A",
        min_relevance=0.8,
        sort_by="Year newest",
    )

    assert [row["pmid"] for row in filtered] == ["333"]


def test_build_trace_summary_counts_coverage():
    payload = {
        "trace": {
            "retrieval_coverage": {
                "baseline_unique_pmids": ["111"],
                "agent_unique_pmids": ["111", "222"],
                "new_pmids_over_baseline": ["222"],
            },
            "stop": {"reason": "sufficient_evidence"},
        }
    }

    summary = streamlit_app._build_trace_summary(payload)

    assert summary == {
        "baseline_unique_pmids": 1,
        "agent_unique_pmids": 2,
        "new_pmids": 1,
        "stop_reason": "sufficient_evidence",
    }


def test_build_branch_rows_flattens_diagnostics():
    rows = streamlit_app._build_branch_rows(
        [
            {
                "query": "asthma trial",
                "diagnostics": {
                    "new_pmids": ["222"],
                    "overlap_pmids": ["111"],
                    "retrieved_count": 2,
                    "evidence_count": 1,
                    "top_relevance_score": 0.92,
                    "stop_reason_after_branch": "sufficient_evidence",
                },
            }
        ]
    )

    assert rows == [
        {
            "query": "asthma trial",
            "new_pmids": "222",
            "overlap_pmids": "111",
            "retrieved_count": 2,
            "evidence_count": 1,
            "top_relevance_score": 0.92,
            "stop_reason_after_branch": "sufficient_evidence",
        }
    ]
