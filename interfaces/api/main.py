from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bioevidence.workflows import AgentWorkflowResult, WorkflowResult, run_agent_workflow, run_rag_pipeline
from bioevidence.extraction.table import evidence_table_rows
from bioevidence.presentation import build_agent_trace_payload
from bioevidence.schemas.query import Query


app = FastAPI(
    title="BioEvidence Copilot API",
    version="0.1.0",
    description="Thin API layer over the local BioEvidence retrieval and agent workflows.",
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)
    data_dir: str | None = None


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "bioevidence-copilot"}


@app.post("/api/v1/query/baseline")
def query_baseline(request: QueryRequest) -> dict[str, Any]:
    try:
        result = run_rag_pipeline(_to_query(request), data_dir=_data_dir(request))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="baseline workflow failed") from exc
    return _workflow_response(result)


@app.post("/api/v1/query/agent")
def query_agent(request: QueryRequest) -> dict[str, Any]:
    try:
        result = run_agent_workflow(_to_query(request), data_dir=_data_dir(request))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="agent workflow failed") from exc
    return _agent_response(result)


def _to_query(request: QueryRequest) -> Query:
    return Query(text=request.query.strip(), top_k=request.top_k)


def _data_dir(request: QueryRequest) -> Path | None:
    if request.data_dir is None:
        return None
    return Path(request.data_dir)


def _workflow_response(result: WorkflowResult) -> dict[str, Any]:
    return {
        "query": result.query.text,
        "rewritten_query": result.answer.rewritten_query or result.query.text,
        "source": result.source,
        "answer": result.answer.answer_text,
        "citations": list(result.answer.citations),
        "retrieved_papers": _retrieved_papers(result),
        "evidence_table": evidence_table_rows(result.evidence_records),
    }


def _agent_response(result: AgentWorkflowResult) -> dict[str, Any]:
    response = _workflow_response(result)
    response.update(
        {
            "baseline": _workflow_response(result.baseline),
            "branches": [branch.to_dict() for branch in result.branch_results],
            "state": {
                "iterations": result.state.iterations,
                "max_iterations": result.state.max_iterations,
                "branch_queries": list(result.state.branch_queries),
                "unique_pmids": sorted(result.state.seen_pmids),
                "sufficient": result.state.sufficient,
                "stop_reason": result.state.stop_reason,
            },
            "comparison": result.comparison,
            "trace": build_agent_trace_payload(result),
        }
    )
    return response


def _retrieved_papers(result: WorkflowResult | AgentWorkflowResult) -> list[dict[str, Any]]:
    return [
        {
            "pmid": candidate.document.pmid,
            "title": candidate.document.title,
            "journal": candidate.document.journal,
            "year": candidate.document.year,
            "score": round(candidate.score, 4),
            "rank": candidate.rank,
        }
        for candidate in result.retrieved_candidates[: result.query.top_k]
    ]
