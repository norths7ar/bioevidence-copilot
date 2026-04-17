from pathlib import Path

import bioevidence.generation.agent_answerer as agent_answerer_module
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings
from bioevidence.generation.agent_answerer import synthesize_agent_answer
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


def test_synthesize_agent_answer_uses_llm_payload(monkeypatch):
    state = AgentState(
        query=Query(text="asthma corticosteroids"),
        evidence_records=[
            EvidenceRecord(
                pmid="111",
                title="Title 111",
                year=2024,
                journal="Journal",
                entities=("asthma",),
                summary="Summary 111",
                relevance_score=0.9,
            ),
            EvidenceRecord(
                pmid="222",
                title="Title 222",
                year=2024,
                journal="Journal",
                entities=("corticosteroids",),
                summary="Summary 222",
                relevance_score=0.8,
            ),
        ],
        branch_queries=["asthma corticosteroids review"],
    )

    monkeypatch.setattr(agent_answerer_module, "load_settings", _settings)
    monkeypatch.setattr(agent_answerer_module, "create_agent_client", lambda settings: object())
    monkeypatch.setattr(
        agent_answerer_module,
        "chat_json",
        lambda client, *, model, messages, max_tokens, temperature: {
            "answer_text": "Agent answer",
            "citations": ["111", "999", "222"],
            "rewritten_query": "asthma corticosteroids review",
        },
    )

    answer = synthesize_agent_answer(state, "baseline answer", client=object())

    assert answer.answer_text == "Agent answer"
    assert answer.citations == ("111", "222")
    assert answer.rewritten_query == "asthma corticosteroids review"


def test_synthesize_agent_answer_falls_back_to_template(monkeypatch):
    state = AgentState(
        query=Query(text="asthma corticosteroids"),
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

    monkeypatch.setattr(agent_answerer_module, "load_settings", _settings)
    monkeypatch.setattr(agent_answerer_module, "create_agent_client", lambda settings: object())
    monkeypatch.setattr(
        agent_answerer_module,
        "chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(agent_answerer_module.AgentLLMError("boom")),
    )

    answer = synthesize_agent_answer(state, "baseline answer", client=object())

    assert answer.answer_text.startswith("Top retrieved evidence for 'asthma corticosteroids'")
    assert answer.citations == ("111",)
