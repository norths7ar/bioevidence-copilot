from __future__ import annotations

from dataclasses import dataclass

from bioevidence.agent.state import AgentState
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    query: Query
    documents: tuple[Document, ...]
    retrieved_candidates: tuple[RetrievedCandidate, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    answer: AnswerBundle
    source: str


@dataclass(frozen=True, slots=True)
class AgentBranchResult:
    query: Query
    documents: tuple[Document, ...]
    retrieved_candidates: tuple[RetrievedCandidate, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query.text,
            "rewritten_query": self.query.rewritten_text or self.query.text,
            "source": self.source,
            "retrieved_pmids": [candidate.document.pmid for candidate in self.retrieved_candidates],
            "evidence_pmids": [record.pmid for record in self.evidence_records],
        }


@dataclass(frozen=True, slots=True)
class AgentWorkflowResult:
    query: Query
    baseline: WorkflowResult
    branch_results: tuple[AgentBranchResult, ...]
    documents: tuple[Document, ...]
    retrieved_candidates: tuple[RetrievedCandidate, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    answer: AnswerBundle
    source: str
    state: AgentState
    comparison: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query.text,
            "baseline": {
                "source": self.baseline.source,
                "answer": self.baseline.answer.answer_text,
                "citations": list(self.baseline.answer.citations),
                "retrieved_pmids": [candidate.document.pmid for candidate in self.baseline.retrieved_candidates],
            },
            "branches": [branch.to_dict() for branch in self.branch_results],
            "state": {
                "iterations": self.state.iterations,
                "max_iterations": self.state.max_iterations,
                "branch_queries": list(self.state.branch_queries),
                "unique_pmids": sorted(self.state.seen_pmids),
                "sufficient": self.state.sufficient,
                "stop_reason": self.state.stop_reason,
            },
            "retrieved_papers": [
                {
                    "pmid": candidate.document.pmid,
                    "title": candidate.document.title,
                    "journal": candidate.document.journal,
                    "year": candidate.document.year,
                    "score": round(candidate.score, 4),
                    "rank": candidate.rank,
                }
                for candidate in self.retrieved_candidates
            ],
            "evidence_table": [
                {
                    "pmid": record.pmid,
                    "title": record.title,
                    "year": record.year,
                    "journal": record.journal,
                    "entities": list(record.entities),
                    "summary": record.summary,
                    "relevance_score": round(record.relevance_score, 4),
                }
                for record in self.evidence_records
            ],
            "answer": self.answer.answer_text,
            "citations": list(self.answer.citations),
            "comparison": self.comparison,
            "source": self.source,
        }
