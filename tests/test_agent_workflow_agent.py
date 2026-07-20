from pathlib import Path
from dataclasses import replace

import bioevidence.workflows.agent as agent_workflow
import bioevidence.agent.planner as planner_module
import pytest
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings
from bioevidence.generation.agent_answerer import AgentSynthesisResult
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query
from bioevidence.graph.provider import GraphDiscoveryError


def _settings() -> Settings:
    return Settings(
        data_dir=Path("data"),
        embedding_cache_dir=Path("data/cache"),
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


def _candidate(pmid: str, score: float, rank: int) -> RetrievedCandidate:
    return RetrievedCandidate(
        document=Document(
            pmid=pmid,
            title=f"Title {pmid}",
            abstract=f"Abstract {pmid}",
            journal="Journal",
            year=2024,
        ),
        score=score,
        rank=rank,
    )


def _evidence(pmid: str, score: float) -> EvidenceRecord:
    return EvidenceRecord(
        pmid=pmid,
        title=f"Title {pmid}",
        year=2024,
        journal="Journal",
        entities=("asthma",),
        summary=f"Summary {pmid}",
        relevance_score=score,
    )


def test_run_agent_workflow_accumulates_branches_and_stops(monkeypatch):
    def fake_retrieval_stack(query: Query, *, data_dir=None, documents=None, settings=None):
        del data_dir, documents, settings
        if query.text == "asthma corticosteroids":
            documents = [
                Document(pmid="111", title="Title 111", abstract="Abstract 111", journal="Journal", year=2024),
                Document(pmid="222", title="Title 222", abstract="Abstract 222", journal="Journal", year=2024),
            ]
            candidates = [_candidate("111", 0.92, 1), _candidate("222", 0.81, 2)]
            evidence = [_evidence("111", 0.92), _evidence("222", 0.81)]
            return documents, candidates, evidence, "baseline"
        documents = [
            Document(pmid="222", title="Title 222", abstract="Abstract 222", journal="Journal", year=2024),
            Document(pmid="333", title="Title 333", abstract="Abstract 333", journal="Journal", year=2024),
        ]
        candidates = [_candidate("333", 0.95, 1), _candidate("222", 0.73, 2)]
        evidence = [_evidence("333", 0.95), _evidence("222", 0.73)]
        return documents, candidates, evidence, "branch"

    def fake_plan_next_steps_with_trace(state: AgentState, *, settings=None, client=None, branch_count=2):
        del settings, client, branch_count
        return planner_module.PlanningResult(
            proposed_queries=("asthma biomarkers",),
            accepted_queries=("asthma biomarkers",),
            rationale="Search for biomarker evidence that may add coverage.",
            source="model",
        )

    def fake_synthesize(state: AgentState, baseline_answer: str, *, settings=None, client=None):
        del state, baseline_answer, settings, client
        return AgentSynthesisResult(
            answer=AnswerBundle(
                answer_text="Agent synthesis",
                citations=("111", "333"),
                evidence_records=tuple(),
                rewritten_query="asthma biomarkers",
            ),
            source="model",
        )

    def fake_run_rag_pipeline(query: Query, *, data_dir=None, documents=None, settings=None):
        documents, candidates, evidence, source = fake_retrieval_stack(
            query,
            data_dir=data_dir,
            documents=documents,
            settings=settings,
        )
        return agent_workflow.WorkflowResult(
            query=query,
            documents=tuple(documents),
            retrieved_candidates=tuple(candidates),
            evidence_records=tuple(evidence),
            answer=AnswerBundle(
                answer_text="Baseline answer",
                citations=tuple(record.pmid for record in evidence),
                evidence_records=tuple(evidence),
                rewritten_query=query.text,
            ),
            source=source,
        )

    monkeypatch.setattr(agent_workflow, "run_rag_pipeline", fake_run_rag_pipeline)
    monkeypatch.setattr(agent_workflow, "run_retrieval_stack", fake_retrieval_stack)
    monkeypatch.setattr(agent_workflow, "plan_next_steps_with_trace", fake_plan_next_steps_with_trace)
    monkeypatch.setattr(agent_workflow, "synthesize_agent_answer_with_trace", fake_synthesize)
    monkeypatch.setattr(agent_workflow, "create_agent_client", lambda settings: object())

    result = agent_workflow.run_agent_workflow(Query(text="asthma corticosteroids"), settings=_settings())

    assert result.branch_results
    assert len(result.branch_results) == 1
    assert result.state.sufficient is True
    assert result.state.stop_reason == "sufficient_evidence"
    assert result.answer.answer_text == "Agent synthesis"
    assert result.comparison["branch_count"] == 1
    assert result.comparison["unique_pmid_coverage"] == 3
    assert result.comparison["agent_improved_retrieval_coverage"] is True
    assert result.comparison["new_pmids_over_baseline"] == ["333"]
    assert result.comparison["baseline_citations"] == ["111", "222"]
    assert result.comparison["agent_citations"] == ["111", "333"]
    assert result.planning_steps[0].rationale == "Search for biomarker evidence that may add coverage."
    assert result.branch_results[0].diagnostics["new_pmids"] == ["333"]
    assert result.branch_results[0].diagnostics["overlap_pmids"] == ["222"]
    assert result.branch_results[0].diagnostics["stop_reason_after_branch"] == "sufficient_evidence"

    events = list(agent_workflow.stream_agent_workflow(Query(text="asthma corticosteroids"), settings=_settings()))

    assert [event["event"] for event in events] == [
        "run_started",
        "baseline_completed",
        "graph_discovery_completed",
        "planner_completed",
        "branch_retrieval_completed",
        "synthesis_completed",
        "run_completed",
        "result",
    ]
    assert isinstance(events[-1]["result"], agent_workflow.AgentWorkflowResult)
    assert events[0]["run_id"] == events[-1]["run_id"]


def test_graph_discovery_only_degrades_declared_operational_errors(monkeypatch) -> None:
    query = Query(text="asthma")
    baseline = agent_workflow.WorkflowResult(
        query=query,
        documents=tuple(),
        retrieved_candidates=tuple(),
        evidence_records=tuple(),
        answer=AnswerBundle(answer_text="No evidence", rewritten_query=query.text),
        source="local_corpus",
    )

    class UnavailableProvider:
        def discover(self, query_text: str):
            raise GraphDiscoveryError("Neo4j unavailable")

        def close(self) -> None:
            return None

    monkeypatch.setattr(agent_workflow, "run_rag_pipeline", lambda *args, **kwargs: baseline)
    result = agent_workflow.run_agent_workflow(
        query,
        settings=replace(_settings(), agent_max_iterations=0),
        graph_provider=UnavailableProvider(),
    )

    assert result.graph_discovery is not None
    assert result.graph_discovery.status == "unavailable"
    assert result.graph_discovery.diagnostics["error_type"] == "GraphDiscoveryError"


def test_graph_discovery_does_not_hide_programming_errors(monkeypatch) -> None:
    query = Query(text="asthma")
    baseline = agent_workflow.WorkflowResult(
        query=query,
        documents=tuple(),
        retrieved_candidates=tuple(),
        evidence_records=tuple(),
        answer=AnswerBundle(answer_text="No evidence", rewritten_query=query.text),
        source="local_corpus",
    )

    class BrokenProvider:
        def discover(self, query_text: str):
            raise KeyError("bug")

        def close(self) -> None:
            return None

    monkeypatch.setattr(agent_workflow, "run_rag_pipeline", lambda *args, **kwargs: baseline)

    with pytest.raises(KeyError, match="bug"):
        agent_workflow.run_agent_workflow(query, settings=_settings(), graph_provider=BrokenProvider())
