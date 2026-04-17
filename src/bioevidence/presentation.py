from __future__ import annotations

from dataclasses import dataclass

from bioevidence.agent.workflow import AgentWorkflowResult, WorkflowResult
from bioevidence.extraction.table import evidence_table_rows, render_evidence_table


@dataclass(frozen=True, slots=True)
class ResultView:
    query: str
    rewritten_query: str
    retrieval_source: str
    retrieved_papers: tuple[dict[str, object], ...]
    evidence_table: tuple[dict[str, object], ...]
    answer: str
    citations: tuple[str, ...]
    evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "retrieval_source": self.retrieval_source,
            "retrieved_papers": list(self.retrieved_papers),
            "evidence_table": list(self.evidence_table),
            "answer": self.answer,
            "citations": list(self.citations),
            "evidence_count": self.evidence_count,
        }


def build_result_view(result: WorkflowResult | AgentWorkflowResult) -> ResultView:
    return ResultView(
        query=result.query.text,
        rewritten_query=result.answer.rewritten_query or result.query.text,
        retrieval_source=result.source,
        retrieved_papers=tuple(
            {
                "pmid": candidate.document.pmid,
                "title": candidate.document.title,
                "journal": candidate.document.journal,
                "year": candidate.document.year,
                "score": round(candidate.score, 4),
                "rank": candidate.rank,
            }
            for candidate in result.retrieved_candidates[: result.query.top_k]
        ),
        evidence_table=tuple(evidence_table_rows(result.evidence_records)),
        answer=result.answer.answer_text,
        citations=tuple(result.answer.citations),
        evidence_count=len(result.evidence_records),
    )


def build_demo_payload(result: WorkflowResult) -> dict[str, object]:
    return build_result_view(result).to_dict()


def build_agent_comparison_payload(result: AgentWorkflowResult) -> dict[str, object]:
    return {
        "query": result.query.text,
        "baseline": build_result_view(result.baseline).to_dict(),
        "agent": build_result_view(result).to_dict(),
        "comparison": result.comparison,
        "branches": [branch.to_dict() for branch in result.branch_results],
        "state": {
            "iterations": result.state.iterations,
            "max_iterations": result.state.max_iterations,
            "branch_queries": list(result.state.branch_queries),
            "unique_pmids": sorted(result.state.seen_pmids),
            "sufficient": result.state.sufficient,
            "stop_reason": result.state.stop_reason,
        },
    }


def render_demo_output(result: WorkflowResult) -> str:
    return render_evidence_table(result.evidence_records)
