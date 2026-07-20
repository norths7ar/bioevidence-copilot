from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging

from bioevidence.agent.llm import AgentLLMError, create_agent_client, chat_json
from bioevidence.agent.prompts import build_planning_messages
from bioevidence.agent.state import AgentState
from bioevidence.config import Settings, load_settings
from bioevidence.schemas.evidence import EvidenceRecord


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlanningResult:
    proposed_queries: tuple[str, ...]
    accepted_queries: tuple[str, ...]
    rationale: str
    source: str
    fallback_reason: str | None = None


def plan_next_steps(
    state: AgentState,
    *,
    settings: Settings | None = None,
    client=None,
    branch_count: int = 2,
) -> list[str]:
    return list(
        plan_next_steps_with_trace(
            state,
            settings=settings,
            client=client,
            branch_count=branch_count,
        ).accepted_queries
    )


def plan_next_steps_with_trace(
    state: AgentState,
    *,
    settings: Settings | None = None,
    client=None,
    branch_count: int = 2,
) -> PlanningResult:
    if state.sufficient or state.iterations >= state.max_iterations:
        return PlanningResult(
            proposed_queries=tuple(),
            accepted_queries=tuple(),
            rationale="Planning skipped because the agent stop condition is already met.",
            source="skipped",
        )

    settings = settings or load_settings()
    try:
        client = client or create_agent_client(settings)
        if not settings.agent_model:
            raise AgentLLMError("AGENT_MODEL is required for agent planning")
        payload = chat_json(
            client,
            model=settings.agent_model,
            messages=build_planning_messages(state, branch_count=branch_count),
            max_tokens=settings.agent_max_output_tokens,
            temperature=settings.agent_temperature,
        )
        branch_queries = _normalize_branch_queries(payload.get("branch_queries"))
        rationale = _normalize_rationale(payload.get("rationale"))
        source = "model"
        fallback_reason = None
    except (AgentLLMError, ValueError, TypeError) as exc:
        LOGGER.warning("planner_fallback reason=%s detail=%s", type(exc).__name__, exc)
        branch_queries = _fallback_branch_queries(state, branch_count=branch_count)
        rationale = _fallback_rationale(state)
        source = "fallback"
        fallback_reason = str(exc)

    accepted_queries = _deduplicate_queries(branch_queries, state.branch_queries)[:branch_count]
    return PlanningResult(
        proposed_queries=tuple(branch_queries),
        accepted_queries=tuple(accepted_queries),
        rationale=rationale,
        source=source,
        fallback_reason=fallback_reason,
    )


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


def _normalize_rationale(raw_rationale: object) -> str:
    text = " ".join(str(raw_rationale or "").split()).strip()
    if text:
        return text
    return "Planner proposed follow-up queries to cover evidence gaps in the current retrieval set."


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


def _fallback_rationale(state: AgentState) -> str:
    if not state.evidence_records:
        return "Fallback planning broadened the original query because no evidence records were available yet."
    return "Fallback planning broadened the original query using generic review, recency, and clinical-evidence variants."


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
