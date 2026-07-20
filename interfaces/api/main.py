from __future__ import annotations

from collections.abc import Iterator
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bioevidence.workflows import (
    AgentWorkflowResult,
    WorkflowResult,
    run_agent_workflow,
    run_rag_pipeline,
    stream_agent_workflow,
)
from bioevidence.extraction.table import evidence_table_rows
from bioevidence.presentation import build_agent_trace_payload
from bioevidence.schemas.query import Query


logger = logging.getLogger(__name__)


app = FastAPI(
    title="BioEvidence Copilot API",
    version="0.2.0",
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


@app.post("/api/v1/query/agent/stream")
def query_agent_stream(request: QueryRequest) -> StreamingResponse:
    try:
        event_iterator = iter(stream_agent_workflow(_to_query(request), data_dir=_data_dir(request)))
        first_event = next(event_iterator)
    except StopIteration:
        return StreamingResponse(iter(()), media_type="application/x-ndjson")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent stream failed before the first event")
        raise HTTPException(status_code=500, detail="agent workflow failed") from exc

    def events() -> Iterator[str]:
        try:
            yield _serialize_stream_event(first_event)
            for event in event_iterator:
                yield _serialize_stream_event(event)
        except (FileNotFoundError, ValueError) as exc:
            yield _serialize_stream_error(400, str(exc))
        except Exception:
            logger.exception("Agent stream failed after streaming started")
            yield _serialize_stream_error(500, "agent workflow failed")
        finally:
            close = getattr(event_iterator, "close", None)
            if close is not None:
                close()

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _to_query(request: QueryRequest) -> Query:
    return Query(text=request.query.strip(), top_k=request.top_k)


def _data_dir(request: QueryRequest) -> Path | None:
    if request.data_dir is None:
        return None
    return Path(request.data_dir)


def _serialize_stream_event(event: dict[str, object]) -> str:
    payload = dict(event)
    result = payload.get("result")
    if isinstance(result, AgentWorkflowResult):
        payload["result"] = _agent_response(result)
    return json.dumps(payload, ensure_ascii=True) + "\n"


def _serialize_stream_error(status_code: int, detail: str) -> str:
    return json.dumps(
        {"node": "error", "error": {"status_code": status_code, "detail": detail}},
        ensure_ascii=True,
    ) + "\n"


def _workflow_response(result: WorkflowResult | AgentWorkflowResult) -> dict[str, Any]:
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
            "graph_discovery": result.graph_discovery.to_dict() if result.graph_discovery else None,
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
