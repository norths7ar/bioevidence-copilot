from pathlib import Path

import bioevidence.agent.workflow as workflow_module
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


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
    def fake_retrieval_stack(query: Query, *, data_dir=None, settings=None):
        del data_dir, settings
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

    def fake_plan_next_steps(state: AgentState, *, settings=None, client=None, branch_count=2):
        del settings, client, branch_count
        return ["asthma biomarkers"]

    def fake_synthesize(state: AgentState, baseline_answer: str, *, settings=None, client=None):
        del state, baseline_answer, settings, client
        return AnswerBundle(
            answer_text="Agent synthesis",
            citations=("111", "333"),
            evidence_records=tuple(),
            rewritten_query="asthma biomarkers",
        )

    monkeypatch.setattr(workflow_module, "_run_retrieval_stack", fake_retrieval_stack)
    monkeypatch.setattr(workflow_module, "plan_next_steps", fake_plan_next_steps)
    monkeypatch.setattr(workflow_module, "synthesize_agent_answer", fake_synthesize)
    monkeypatch.setattr(workflow_module, "create_agent_client", lambda settings: object())

    result = workflow_module.run_agent_workflow(Query(text="asthma corticosteroids"), settings=_settings())

    assert result.branch_results
    assert len(result.branch_results) == 1
    assert result.state.sufficient is True
    assert result.state.stop_reason == "sufficient_evidence"
    assert result.answer.answer_text == "Agent synthesis"
    assert result.comparison["branch_count"] == 1
    assert result.comparison["unique_pmid_coverage"] == 3
    assert result.comparison["baseline_citations"] == ["111", "222"]
    assert result.comparison["agent_citations"] == ["111", "333"]
