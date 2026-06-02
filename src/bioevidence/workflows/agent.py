from __future__ import annotations

from pathlib import Path

from bioevidence.agent.llm import AgentLLMError, create_agent_client
from bioevidence.agent.planner import plan_next_steps
from bioevidence.agent.state import AgentState
from bioevidence.agent.stop_criteria import should_stop
from bioevidence.agent.tools import merge_candidates, merge_evidence_records
from bioevidence.config import Settings, load_settings
from bioevidence.generation.agent_answerer import synthesize_agent_answer
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.workflows.baseline import run_rag_pipeline
from bioevidence.workflows.models import AgentBranchResult, AgentWorkflowResult, WorkflowResult
from bioevidence.workflows.retrieval_stack import run_retrieval_stack


def run_agent_workflow(
    query: Query,
    *,
    data_dir: Path | None = None,
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
) -> AgentWorkflowResult:
    settings = settings or load_settings()
    try:
        agent_client = create_agent_client(settings)
    except AgentLLMError:
        agent_client = None
    baseline = run_rag_pipeline(query, data_dir=data_dir, documents=documents, settings=settings)
    state = AgentState(query=query, max_iterations=settings.agent_max_iterations)
    state.record_branch_query(query.text)
    state.merge_candidates(baseline.retrieved_candidates)
    state.merge_evidence_records(baseline.evidence_records)

    branch_results: list[AgentBranchResult] = []

    while not should_stop(
        state,
        minimum_unique_pmids=settings.agent_min_unique_pmids,
        minimum_relevance_score=settings.agent_min_relevance_score,
    ):
        planned_queries = plan_next_steps(
            state,
            settings=settings,
            client=agent_client,
        )
        if not planned_queries:
            if state.stop_reason is None:
                state.stop_reason = "planner_exhausted"
            break

        branch_added = False
        for branch_query in planned_queries:
            if not state.record_branch_query(branch_query):
                continue
            branch_added = True
            branch_query_obj = Query(
                text=branch_query,
                rewritten_text=branch_query,
                top_k=query.top_k,
            )
            documents, ranked_candidates, evidence_records, source = run_retrieval_stack(
                branch_query_obj,
                data_dir=data_dir,
                documents=documents,
                settings=settings,
            )
            branch_result = AgentBranchResult(
                query=branch_query_obj,
                documents=tuple(documents),
                retrieved_candidates=tuple(ranked_candidates),
                evidence_records=tuple(evidence_records),
                source=source,
            )
            branch_results.append(branch_result)
            merge_candidates(state, branch_result.retrieved_candidates)
            merge_evidence_records(state, branch_result.evidence_records)
            if should_stop(
                state,
                minimum_unique_pmids=settings.agent_min_unique_pmids,
                minimum_relevance_score=settings.agent_min_relevance_score,
            ):
                break

        state.iterations += 1
        if not branch_added and state.stop_reason is None:
            state.stop_reason = "planner_exhausted"
            break
        if state.stop_reason is not None:
            break

    if state.stop_reason is None:
        state.stop_reason = "max_iterations" if state.iterations >= state.max_iterations else "planner_exhausted"

    answer = synthesize_agent_answer(
        state,
        baseline.answer.answer_text,
        settings=settings,
        client=agent_client,
    )
    comparison = _build_comparison(
        baseline,
        answer,
        state,
        branch_results,
        agent_backend_ready=bool(settings.agent_api_key and settings.agent_base_url and settings.agent_model),
    )
    return AgentWorkflowResult(
        query=query,
        baseline=baseline,
        branch_results=tuple(branch_results),
        documents=tuple(_merge_documents([baseline.documents, *(branch.documents for branch in branch_results)])),
        retrieved_candidates=tuple(state.top_candidates()),
        evidence_records=tuple(state.top_evidence_records()),
        answer=answer,
        source=f"agent:{baseline.source}",
        state=state,
        comparison=comparison,
    )


def _merge_documents(document_groups: list[tuple[Document, ...]]) -> list[Document]:
    merged: dict[str, Document] = {}
    for group in document_groups:
        for document in group:
            merged.setdefault(document.pmid, document)
    return list(merged.values())


def _build_comparison(
    baseline: WorkflowResult,
    answer: AnswerBundle,
    state: AgentState,
    branch_results: list[AgentBranchResult],
    *,
    agent_backend_ready: bool,
) -> dict[str, object]:
    baseline_pmids = [candidate.document.pmid for candidate in baseline.retrieved_candidates]
    agent_pmids = [candidate.document.pmid for candidate in state.all_candidates]
    baseline_citations = set(baseline.answer.citations)
    agent_citations = set(answer.citations)
    return {
        "branch_count": len(branch_results),
        "iterations": state.iterations,
        "unique_pmid_coverage": len(state.seen_pmids),
        "baseline_unique_pmids": len({pmid for pmid in baseline_pmids}),
        "agent_unique_pmids": len({pmid for pmid in agent_pmids}),
        "citation_overlap": sorted(baseline_citations & agent_citations),
        "baseline_citations": list(baseline.answer.citations),
        "agent_citations": list(answer.citations),
        "stop_reason": state.stop_reason,
        "agent_backend_ready": agent_backend_ready,
    }
