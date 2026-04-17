from __future__ import annotations

from bioevidence.agent.workflow import WorkflowResult
from bioevidence.extraction.table import evidence_table_rows, render_evidence_table
from bioevidence.schemas.query import Query


def build_demo_payload(query: Query, result: WorkflowResult) -> dict[str, object]:
    return {
        "query": query.text,
        "rewritten_query": result.answer.rewritten_query,
        "retrieval_source": result.source,
        "retrieved_papers": [
            {
                "pmid": candidate.document.pmid,
                "title": candidate.document.title,
                "journal": candidate.document.journal,
                "year": candidate.document.year,
                "score": round(candidate.score, 4),
                "rank": candidate.rank,
            }
            for candidate in result.retrieved_candidates[: query.top_k]
        ],
        "evidence_table": evidence_table_rows(result.evidence_records),
        "answer": result.answer.answer_text,
        "citations": list(result.answer.citations),
        "evidence_count": len(result.evidence_records),
    }


def render_demo_output(result: WorkflowResult) -> str:
    return render_evidence_table(result.evidence_records)
