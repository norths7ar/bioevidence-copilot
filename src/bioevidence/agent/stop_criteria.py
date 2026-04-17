from __future__ import annotations

from bioevidence.agent.state import AgentState


def should_stop(
    state: AgentState,
    *,
    minimum_unique_pmids: int = 3,
    minimum_relevance_score: float = 0.6,
) -> bool:
    if state.iterations >= state.max_iterations:
        state.sufficient = False
        state.stop_reason = "max_iterations"
        return True

    if state.unique_pmid_count() < minimum_unique_pmids:
        return False

    top_scores = state.top_relevance_scores(limit=minimum_unique_pmids)
    if len(top_scores) < minimum_unique_pmids:
        return False

    if min(top_scores) >= minimum_relevance_score:
        state.sufficient = True
        state.stop_reason = "sufficient_evidence"
        return True

    return False
