from __future__ import annotations

from collections.abc import Sequence

from bioevidence.agent.llm import AgentLLMError, create_agent_client, chat_json
from bioevidence.agent.prompts import build_planning_messages
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings, load_settings
from bioevidence.schemas.evidence import EvidenceRecord


def plan_next_steps(
    state: AgentState,
    *,
    settings: Settings | None = None,
    client=None,
    branch_count: int = 2,
) -> list[str]:
    if state.sufficient or state.iterations >= state.max_iterations:
        return []

    settings = settings or load_settings()
    client = client or create_agent_client(settings)
    if not settings.agent_model:
        raise AgentLLMError("BIOEVIDENCE_AGENT_MODEL is required for agent planning")
    try:
        payload = chat_json(
            client,
            model=settings.agent_model,
            messages=build_planning_messages(state, branch_count=branch_count),
            max_tokens=settings.agent_max_output_tokens,
            temperature=settings.agent_temperature,
        )
        branch_queries = _normalize_branch_queries(payload.get("branch_queries"))
    except (AgentLLMError, ValueError, TypeError):
        branch_queries = _fallback_branch_queries(state, branch_count=branch_count)

    return _deduplicate_queries(branch_queries, state.branch_queries)[:branch_count]


def _normalize_branch_queries(raw_queries: object) -> list[str]:
    if raw_queries is None:
        return []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, Sequence):
        raise ValueError("branch_queries must be a string or sequence of strings")
    normalized: list[str] = []
    for value in raw_queries:
        text = " ".join(str(value).split()).strip()
        if text:
            normalized.append(text)
    return normalized


def _fallback_branch_queries(state: AgentState, *, branch_count: int) -> list[str]:
    base_query = state.query.text.strip()
    evidence_hint = _evidence_hint(state.evidence_records)
    variants = [
        f"{base_query} review",
        f"{base_query} recent literature",
        f"{base_query} clinical evidence",
        f"{base_query} mechanism",
        f"{base_query} evidence {evidence_hint}".strip(),
    ]
    return variants[:branch_count]


def _evidence_hint(records: Sequence[EvidenceRecord]) -> str:
    if not records:
        return ""
    top_terms: list[str] = []
    for record in records[:3]:
        for token in record.entities:
            if token not in top_terms:
                top_terms.append(token)
    return " ".join(top_terms[:2])


def _deduplicate_queries(candidate_queries: Sequence[str], existing_queries: Sequence[str]) -> list[str]:
    seen = {" ".join(query.split()).strip() for query in existing_queries}
    unique_queries: list[str] = []
    for query in candidate_queries:
        normalized = " ".join(query.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_queries.append(normalized)
    return unique_queries
