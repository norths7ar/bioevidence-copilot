from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query
from bioevidence.schemas.document import RetrievedCandidate


@dataclass(slots=True)
class AgentState:
    query: Query
    iterations: int = 0
    max_iterations: int = 3
    branch_queries: list[str] = field(default_factory=list)
    all_candidates: list[RetrievedCandidate] = field(default_factory=list)
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    seen_pmids: set[str] = field(default_factory=set)
    sufficient: bool = False
    stop_reason: str | None = None

    def record_branch_query(self, branch_query: str) -> bool:
        normalized_query = " ".join(branch_query.split()).strip()
        if not normalized_query or normalized_query in self.branch_queries:
            return False
        self.branch_queries.append(normalized_query)
        return True

    def merge_candidates(self, candidates: Sequence[RetrievedCandidate]) -> None:
        merged: dict[str, RetrievedCandidate] = {candidate.document.pmid: candidate for candidate in self.all_candidates}
        for candidate in candidates:
            pmid = candidate.document.pmid
            current = merged.get(pmid)
            if current is None or candidate.score > current.score:
                merged[pmid] = candidate
            self.seen_pmids.add(pmid)
        self.all_candidates = sorted(merged.values(), key=lambda candidate: (-candidate.score, candidate.document.pmid))

    def merge_evidence_records(self, records: Sequence[EvidenceRecord]) -> None:
        merged: dict[str, EvidenceRecord] = {record.pmid: record for record in self.evidence_records}
        for record in records:
            current = merged.get(record.pmid)
            if current is None or record.relevance_score > current.relevance_score:
                merged[record.pmid] = record
            self.seen_pmids.add(record.pmid)
        self.evidence_records = sorted(merged.values(), key=lambda record: (-record.relevance_score, record.pmid))

    def top_candidates(self, limit: int | None = None) -> tuple[RetrievedCandidate, ...]:
        if limit is None:
            return tuple(self.all_candidates)
        return tuple(self.all_candidates[:limit])

    def top_evidence_records(self, limit: int | None = None) -> tuple[EvidenceRecord, ...]:
        if limit is None:
            return tuple(self.evidence_records)
        return tuple(self.evidence_records[:limit])

    def unique_pmid_count(self) -> int:
        return len(self.seen_pmids)

    def best_relevance_score(self) -> float:
        return max((record.relevance_score for record in self.evidence_records), default=0.0)

    def top_relevance_scores(self, limit: int) -> list[float]:
        return [record.relevance_score for record in self.evidence_records[:limit]]
