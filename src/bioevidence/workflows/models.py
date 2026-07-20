from __future__ import annotations

from dataclasses import dataclass, field

from bioevidence.agent.state import AgentState
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query
from bioevidence.graph.models import GraphDiscoveryResult


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
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query.text,
            "rewritten_query": self.query.rewritten_text or self.query.text,
            "source": self.source,
            "retrieved_pmids": [candidate.document.pmid for candidate in self.retrieved_candidates],
            "evidence_pmids": [record.pmid for record in self.evidence_records],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class AgentPlanningStep:
    iteration: int
    existing_queries: tuple[str, ...]
    proposed_queries: tuple[str, ...]
    accepted_queries: tuple[str, ...]
    rationale: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "existing_queries": list(self.existing_queries),
            "proposed_queries": list(self.proposed_queries),
            "accepted_queries": list(self.accepted_queries),
            "rationale": self.rationale,
            "source": self.source,
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
    planning_steps: tuple[AgentPlanningStep, ...] = tuple()
    graph_discovery: GraphDiscoveryResult | None = None
    run_id: str | None = None
    trace_events: tuple[dict[str, object], ...] = tuple()

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
            "trace": {
                "run_id": self.run_id,
                "events": [dict(event) for event in self.trace_events],
                "original_query": self.query.text,
                "rewritten_query": self.answer.rewritten_query or self.query.text,
                "planning_steps": [step.to_dict() for step in self.planning_steps],
                "branch_diagnostics": [branch.to_dict() for branch in self.branch_results],
                "graph_discovery": self.graph_discovery.to_dict() if self.graph_discovery else None,
                "retrieval_coverage": self.comparison.get("retrieval_coverage", {}),
                "stop": {
                    "reason": self.state.stop_reason,
                    "sufficient": self.state.sufficient,
                    "iterations": self.state.iterations,
                    "max_iterations": self.state.max_iterations,
                },
            },
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
