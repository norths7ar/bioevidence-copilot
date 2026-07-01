from __future__ import annotations

from pathlib import Path

from bioevidence.agent.llm import AgentLLMError, create_agent_client
from bioevidence.agent.planner import plan_next_steps_with_trace
from bioevidence.agent.state import AgentState
from bioevidence.agent.stop_criteria import should_stop
from bioevidence.agent.tools import merge_candidates, merge_evidence_records
from bioevidence.config import Settings, load_settings
from bioevidence.generation.agent_answerer import synthesize_agent_answer
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.workflows.baseline import run_rag_pipeline
from bioevidence.workflows.models import AgentBranchResult, AgentPlanningStep, AgentWorkflowResult, WorkflowResult
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
    planning_steps: list[AgentPlanningStep] = []

    while not should_stop(
        state,
        minimum_unique_pmids=settings.agent_min_unique_pmids,
        minimum_relevance_score=settings.agent_min_relevance_score,
    ):
        planning_result = plan_next_steps_with_trace(
            state,
            settings=settings,
            client=agent_client,
        )
        planned_queries = list(planning_result.accepted_queries)
        planning_steps.append(
            AgentPlanningStep(
                iteration=state.iterations,
                existing_queries=tuple(state.branch_queries),
                proposed_queries=planning_result.proposed_queries,
                accepted_queries=planning_result.accepted_queries,
                rationale=planning_result.rationale,
                source=planning_result.source,
            )
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
            pmids_before_branch = set(state.seen_pmids)
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
                diagnostics=_build_branch_diagnostics(
                    branch_query=branch_query,
                    iteration=state.iterations,
                    pmids_before_branch=pmids_before_branch,
                    candidates=ranked_candidates,
                    evidence_records=evidence_records,
                    planner_rationale=planning_result.rationale,
                    planner_source=planning_result.source,
                    stop_reason_after_branch=None,
                ),
            )
            merge_candidates(state, branch_result.retrieved_candidates)
            merge_evidence_records(state, branch_result.evidence_records)
            if should_stop(
                state,
                minimum_unique_pmids=settings.agent_min_unique_pmids,
                minimum_relevance_score=settings.agent_min_relevance_score,
            ):
                branch_result = _with_stop_reason(branch_result, state.stop_reason)
                branch_results.append(branch_result)
                break
            branch_results.append(branch_result)

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
        planning_steps=tuple(planning_steps),
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
    baseline_unique_pmids = {pmid for pmid in baseline_pmids}
    agent_unique_pmids = {pmid for pmid in agent_pmids}
    new_pmids_over_baseline = sorted(agent_unique_pmids - baseline_unique_pmids)
    baseline_citations = set(baseline.answer.citations)
    agent_citations = set(answer.citations)
    return {
        "branch_count": len(branch_results),
        "iterations": state.iterations,
        "unique_pmid_coverage": len(state.seen_pmids),
        "baseline_unique_pmids": len(baseline_unique_pmids),
        "agent_unique_pmids": len(agent_unique_pmids),
        "new_pmids_over_baseline": new_pmids_over_baseline,
        "agent_improved_retrieval_coverage": len(agent_unique_pmids) > len(baseline_unique_pmids),
        "retrieval_coverage": {
            "baseline_unique_pmids": sorted(baseline_unique_pmids),
            "agent_unique_pmids": sorted(agent_unique_pmids),
            "new_pmids_over_baseline": new_pmids_over_baseline,
            "overlap_pmids": sorted(baseline_unique_pmids & agent_unique_pmids),
        },
        "citation_overlap": sorted(baseline_citations & agent_citations),
        "baseline_citations": list(baseline.answer.citations),
        "agent_citations": list(answer.citations),
        "stop_reason": state.stop_reason,
        "agent_backend_ready": agent_backend_ready,
    }


def _build_branch_diagnostics(
    *,
    branch_query: str,
    iteration: int,
    pmids_before_branch: set[str],
    candidates: list,
    evidence_records: list,
    planner_rationale: str,
    planner_source: str,
    stop_reason_after_branch: str | None,
) -> dict[str, object]:
    retrieved_pmids = [candidate.document.pmid for candidate in candidates]
    evidence_pmids = [record.pmid for record in evidence_records]
    retrieved_set = set(retrieved_pmids)
    top_relevance_score = max((record.relevance_score for record in evidence_records), default=0.0)
    top_retrieval_score = max((candidate.score for candidate in candidates), default=0.0)
    return {
        "iteration": iteration,
        "branch_query": branch_query,
        "planner_source": planner_source,
        "retrieval_rationale": planner_rationale,
        "retrieved_count": len(retrieved_pmids),
        "evidence_count": len(evidence_pmids),
        "new_pmids": sorted(retrieved_set - pmids_before_branch),
        "overlap_pmids": sorted(retrieved_set & pmids_before_branch),
        "top_retrieval_score": round(top_retrieval_score, 4),
        "top_relevance_score": round(top_relevance_score, 4),
        "stop_reason_after_branch": stop_reason_after_branch,
    }


def _with_stop_reason(branch_result: AgentBranchResult, stop_reason: str | None) -> AgentBranchResult:
    diagnostics = dict(branch_result.diagnostics)
    diagnostics["stop_reason_after_branch"] = stop_reason
    return AgentBranchResult(
        query=branch_result.query,
        documents=branch_result.documents,
        retrieved_candidates=branch_result.retrieved_candidates,
        evidence_records=branch_result.evidence_records,
        source=branch_result.source,
        diagnostics=diagnostics,
    )
