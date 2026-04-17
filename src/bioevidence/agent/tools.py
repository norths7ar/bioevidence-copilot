from __future__ import annotations

from collections.abc import Sequence

from bioevidence.agent.state import AgentState
from bioevidence.schemas.document import RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord


def available_tools() -> list[str]:
    return ["plan", "retrieve", "deduplicate", "extract", "synthesize", "stop"]


def merge_candidates(state: AgentState, candidates: Sequence[RetrievedCandidate]) -> None:
    state.merge_candidates(candidates)


def merge_evidence_records(state: AgentState, records: Sequence[EvidenceRecord]) -> None:
    state.merge_evidence_records(records)


def summarize_candidates(candidates: Sequence[RetrievedCandidate], limit: int = 5) -> str:
    lines: list[str] = []
    for candidate in candidates[:limit]:
        year_text = str(candidate.document.year) if candidate.document.year is not None else "n.d."
        lines.append(
            f"{candidate.document.pmid}: {candidate.document.title} ({year_text}) | score={candidate.score:.3f}"
        )
    return "\n".join(lines) if lines else "(no candidates)"


def summarize_evidence(records: Sequence[EvidenceRecord], limit: int = 5) -> str:
    lines: list[str] = []
    for record in records[:limit]:
        year_text = str(record.year) if record.year is not None else "n.d."
        lines.append(
            f"{record.pmid}: {record.title} ({year_text}) | score={record.relevance_score:.3f} | {record.summary}"
        )
    return "\n".join(lines) if lines else "(no evidence)"


def unique_pmids(candidates: Sequence[RetrievedCandidate]) -> tuple[str, ...]:
    seen: set[str] = set()
    pmids: list[str] = []
    for candidate in candidates:
        pmid = candidate.document.pmid
        if pmid in seen:
            continue
        seen.add(pmid)
        pmids.append(pmid)
    return tuple(pmids)
