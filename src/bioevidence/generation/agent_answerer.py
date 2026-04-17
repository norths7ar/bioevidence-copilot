from __future__ import annotations

from collections.abc import Sequence

from bioevidence.agent.llm import AgentLLMError, chat_json, create_agent_client
from bioevidence.agent.prompts import build_synthesis_messages
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings, load_settings
from bioevidence.generation.answerer import generate_answer
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord


def synthesize_agent_answer(
    state: AgentState,
    baseline_answer: str,
    *,
    settings: Settings | None = None,
    client=None,
) -> AnswerBundle:
    settings = settings or load_settings()
    client = client or create_agent_client(settings)
    evidence_records = tuple(state.top_evidence_records())
    if not settings.agent_model:
        raise AgentLLMError("BIOEVIDENCE_AGENT_MODEL is required for agent synthesis")
    try:
        payload = chat_json(
            client,
            model=settings.agent_model,
            messages=build_synthesis_messages(state, baseline_answer),
            max_tokens=settings.agent_max_output_tokens,
            temperature=settings.agent_temperature,
        )
        answer_text = _require_non_empty_str(payload.get("answer_text"), fallback=baseline_answer)
        citations = _normalize_citations(payload.get("citations"), evidence_records)
        rewritten_query = _normalize_rewritten_query(payload.get("rewritten_query"), state)
        return AnswerBundle(
            answer_text=answer_text,
            citations=citations,
            evidence_records=evidence_records,
            rewritten_query=rewritten_query,
        )
    except (AgentLLMError, ValueError, TypeError):
        return generate_answer(state.query, list(evidence_records))


def _require_non_empty_str(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _normalize_citations(raw_citations: object, evidence_records: Sequence[EvidenceRecord]) -> tuple[str, ...]:
    allowed_pmids = {record.pmid for record in evidence_records}
    if raw_citations is None:
        return tuple(record.pmid for record in evidence_records)
    if isinstance(raw_citations, str):
        raw_citations = [raw_citations]
    if not isinstance(raw_citations, Sequence):
        return tuple(record.pmid for record in evidence_records)

    citations: list[str] = []
    seen: set[str] = set()
    for value in raw_citations:
        pmid = str(value).strip()
        if not pmid or pmid in seen or pmid not in allowed_pmids:
            continue
        seen.add(pmid)
        citations.append(pmid)
    return tuple(citations) if citations else tuple(record.pmid for record in evidence_records)


def _normalize_rewritten_query(raw_value: object, state: AgentState) -> str:
    if isinstance(raw_value, str) and raw_value.strip():
        return " ".join(raw_value.split()).strip()
    if state.branch_queries:
        return state.branch_queries[-1]
    return state.query.text
