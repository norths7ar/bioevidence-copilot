from pathlib import Path

import bioevidence.agent.planner as planner_module
from bioevidence.agent.llm import AgentLLMError
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings
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


def test_plan_next_steps_uses_model_queries(monkeypatch):
    state = AgentState(
        query=Query(text="asthma corticosteroids"),
        branch_queries=["asthma corticosteroids review"],
        evidence_records=[
            EvidenceRecord(
                pmid="111",
                title="Title 111",
                year=2024,
                journal="Journal",
                entities=("asthma",),
                summary="Summary 111",
                relevance_score=0.9,
            )
        ],
    )

    monkeypatch.setattr(planner_module, "load_settings", _settings)
    monkeypatch.setattr(planner_module, "create_agent_client", lambda settings: object())
    monkeypatch.setattr(
        planner_module,
        "chat_json",
        lambda client, *, model, messages, max_tokens, temperature: {
            "branch_queries": [" asthma mechanisms ", "asthma mechanisms", "asthma biomarkers"],
        },
    )

    branch_queries = planner_module.plan_next_steps(state, branch_count=2)

    assert branch_queries == ["asthma mechanisms", "asthma biomarkers"]


def test_plan_next_steps_falls_back_when_llm_unavailable(monkeypatch):
    state = AgentState(query=Query(text="asthma corticosteroids"))

    monkeypatch.setattr(planner_module, "load_settings", _settings)
    monkeypatch.setattr(planner_module, "create_agent_client", lambda settings: object())
    monkeypatch.setattr(planner_module, "chat_json", lambda *args, **kwargs: (_ for _ in ()).throw(AgentLLMError("boom")))

    branch_queries = planner_module.plan_next_steps(state, branch_count=2)

    assert branch_queries == [
        "asthma corticosteroids review",
        "asthma corticosteroids recent literature",
    ]
