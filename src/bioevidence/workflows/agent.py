from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from bioevidence.agent.llm import AgentLLMError, create_agent_client
from bioevidence.agent.planner import plan_next_steps_with_trace
from bioevidence.agent.state import AgentState
from bioevidence.agent.stop_criteria import should_stop
from bioevidence.agent.tools import merge_candidates, merge_evidence_records
from bioevidence.config import Settings, load_settings
from bioevidence.generation.agent_answerer import synthesize_agent_answer
from bioevidence.graph.models import GraphDiscoveryResult
from bioevidence.graph.provider import GraphDiscoveryProvider, create_graph_provider
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.workflows.baseline import run_rag_pipeline
from bioevidence.workflows.models import AgentBranchResult, AgentPlanningStep, AgentWorkflowResult, WorkflowResult
from bioevidence.workflows.retrieval_stack import run_retrieval_stack


class AgentGraphState(TypedDict, total=False):
    baseline: WorkflowResult
    branch_results: list[AgentBranchResult]
    documents: tuple[Document, ...] | list[Document] | None
    graph_discovery: GraphDiscoveryResult
    planning_steps: list[AgentPlanningStep]
    planned_queries: list[str]
    result: AgentWorkflowResult
    state: AgentState


def run_agent_workflow(
    query: Query,
    *,
    data_dir: Path | None = None,
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
    graph_provider: GraphDiscoveryProvider | None = None,
) -> AgentWorkflowResult:
    settings = settings or load_settings()
    provider = graph_provider or create_graph_provider(settings)
    owns_provider = graph_provider is None
    try:
        graph = _build_agent_graph(query, data_dir=data_dir, settings=settings, graph_provider=provider)
        output = graph.invoke(_initial_graph_state(query, documents))
        return output["result"]
    finally:
        if owns_provider:
            provider.close()


def stream_agent_workflow(
    query: Query,
    *,
    data_dir: Path | None = None,
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
    graph_provider: GraphDiscoveryProvider | None = None,
) -> Iterator[dict[str, object]]:
    settings = settings or load_settings()
    provider = graph_provider or create_graph_provider(settings)
    owns_provider = graph_provider is None
    graph = _build_agent_graph(query, data_dir=data_dir, settings=settings, graph_provider=provider)
    try:
        for update in graph.stream(_initial_graph_state(query, documents), stream_mode="updates"):
            node_name, node_update = next(iter(update.items()))
            yield _stream_event(node_name, node_update)
    finally:
        if owns_provider:
            provider.close()


def _initial_graph_state(
    query: Query,
    documents: tuple[Document, ...] | list[Document] | None,
) -> AgentGraphState:
    return {
        "branch_results": [],
        "documents": documents,
        "planning_steps": [],
        "planned_queries": [],
        "state": AgentState(query=query),
    }


def _build_agent_graph(
    query: Query,
    *,
    data_dir: Path | None,
    settings: Settings,
    graph_provider: GraphDiscoveryProvider,
):
    try:
        agent_client = create_agent_client(settings)
    except AgentLLMError:
        agent_client = None

    def baseline_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        baseline = run_rag_pipeline(
            query,
            data_dir=data_dir,
            documents=runtime.get("documents"),
            settings=settings,
        )
        state = AgentState(query=query, max_iterations=settings.agent_max_iterations)
        state.record_branch_query(query.text)
        state.merge_candidates(baseline.retrieved_candidates)
        state.merge_evidence_records(baseline.evidence_records)
        return {"baseline": baseline, "state": state}

    def discover_graph_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        try:
            discovery = graph_provider.discover(query.text)
        except Exception as exc:
            discovery = GraphDiscoveryResult(
                query=query.text,
                status="unavailable",
                diagnostics={"error_type": type(exc).__name__, "message": str(exc)},
            )
        planned_queries = list(discovery.expanded_queries)
        planning_steps = list(runtime.get("planning_steps", []))
        state = runtime["state"]
        accepted_queries = tuple(
            branch_query for branch_query in planned_queries if branch_query not in state.branch_queries
        )
        if accepted_queries:
            planning_steps.append(
                AgentPlanningStep(
                    iteration=state.iterations,
                    existing_queries=tuple(state.branch_queries),
                    proposed_queries=tuple(planned_queries),
                    accepted_queries=accepted_queries,
                    rationale="Hetionet paths supplied related biomedical entities for literature query expansion.",
                    source="knowledge_graph",
                )
            )
        return {
            "graph_discovery": discovery,
            "planned_queries": list(accepted_queries),
            "planning_steps": planning_steps,
        }

    def plan_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        state = runtime["state"]
        planning_result = plan_next_steps_with_trace(state, settings=settings, client=agent_client)
        planning_steps = list(runtime.get("planning_steps", []))
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
        if not planning_result.accepted_queries and state.stop_reason is None:
            state.stop_reason = "planner_exhausted"
        return {
            "planned_queries": list(planning_result.accepted_queries),
            "planning_steps": planning_steps,
            "state": state,
        }

    def retrieve_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        state = runtime["state"]
        branch_results = list(runtime.get("branch_results", []))
        runtime_documents = runtime.get("documents")
        planner_step = runtime["planning_steps"][-1]
        branch_added = False
        for branch_query in runtime.get("planned_queries", []):
            if not state.record_branch_query(branch_query):
                continue
            branch_added = True
            branch_query_obj = Query(text=branch_query, rewritten_text=branch_query, top_k=query.top_k)
            pmids_before_branch = set(state.seen_pmids)
            returned_documents, ranked_candidates, evidence_records, source = run_retrieval_stack(
                branch_query_obj,
                data_dir=data_dir,
                documents=runtime_documents,
                settings=settings,
            )
            runtime_documents = returned_documents
            branch_result = AgentBranchResult(
                query=branch_query_obj,
                documents=tuple(returned_documents),
                retrieved_candidates=tuple(ranked_candidates),
                evidence_records=tuple(evidence_records),
                source=source,
                diagnostics=_build_branch_diagnostics(
                    branch_query=branch_query,
                    iteration=state.iterations,
                    pmids_before_branch=pmids_before_branch,
                    candidates=ranked_candidates,
                    evidence_records=evidence_records,
                    planner_rationale=planner_step.rationale,
                    planner_source=planner_step.source,
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
                branch_results.append(_with_stop_reason(branch_result, state.stop_reason))
                break
            branch_results.append(branch_result)
        state.iterations += 1
        if not branch_added and state.stop_reason is None:
            state.stop_reason = "planner_exhausted"
        return {
            "branch_results": branch_results,
            "documents": runtime_documents,
            "planned_queries": [],
            "state": state,
        }

    def synthesize_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        state = runtime["state"]
        if state.stop_reason is None:
            state.stop_reason = "max_iterations" if state.iterations >= state.max_iterations else "planner_exhausted"
        baseline = runtime["baseline"]
        branch_results = runtime.get("branch_results", [])
        answer = synthesize_agent_answer(state, baseline.answer.answer_text, settings=settings, client=agent_client)
        result = AgentWorkflowResult(
            query=query,
            baseline=baseline,
            branch_results=tuple(branch_results),
            documents=tuple(_merge_documents([baseline.documents, *(branch.documents for branch in branch_results)])),
            retrieved_candidates=tuple(state.top_candidates()),
            evidence_records=tuple(state.top_evidence_records()),
            answer=answer,
            source=f"agent:{baseline.source}",
            state=state,
            comparison=_build_comparison(
                baseline,
                answer,
                state,
                branch_results,
                agent_backend_ready=bool(settings.agent_api_key and settings.agent_base_url and settings.agent_model),
            ),
            planning_steps=tuple(runtime.get("planning_steps", [])),
            graph_discovery=runtime.get("graph_discovery"),
        )
        return {"result": result, "state": state}

    graph = StateGraph(AgentGraphState)
    # LangGraph's callable overloads do not currently accept partial TypedDict returns in mypy.
    graph.add_node("retrieve_baseline", cast(Any, baseline_node))
    graph.add_node("discover_graph", cast(Any, discover_graph_node))
    graph.add_node("plan", cast(Any, plan_node))
    graph.add_node("retrieve_branches", cast(Any, retrieve_node))
    graph.add_node("synthesize", cast(Any, synthesize_node))
    graph.add_edge(START, "retrieve_baseline")
    graph.add_edge("retrieve_baseline", "discover_graph")
    graph.add_conditional_edges("discover_graph", lambda state: _route_after_discovery(state, settings))
    graph.add_conditional_edges("plan", _route_after_plan)
    graph.add_conditional_edges("retrieve_branches", lambda state: _route_after_retrieval(state, settings))
    graph.add_edge("synthesize", END)
    return graph.compile()


def _route_after_discovery(graph_state: AgentGraphState, settings: Settings) -> str:
    if _stop(graph_state["state"], settings):
        return "synthesize"
    return "retrieve_branches" if graph_state.get("planned_queries") else "plan"


def _route_after_plan(graph_state: AgentGraphState) -> str:
    return "retrieve_branches" if graph_state.get("planned_queries") else "synthesize"


def _route_after_retrieval(graph_state: AgentGraphState, settings: Settings) -> str:
    return "synthesize" if _stop(graph_state["state"], settings) else "plan"


def _stop(state: AgentState, settings: Settings) -> bool:
    return should_stop(
        state,
        minimum_unique_pmids=settings.agent_min_unique_pmids,
        minimum_relevance_score=settings.agent_min_relevance_score,
    )


def _stream_event(node_name: str, update: AgentGraphState) -> dict[str, object]:
    event: dict[str, object] = {"node": node_name}
    if node_name == "discover_graph" and "graph_discovery" in update:
        event["graph_discovery"] = update["graph_discovery"].to_dict()
    elif node_name in {"plan", "retrieve_branches"}:
        event["planned_queries"] = list(update.get("planned_queries", []))
        event["branch_count"] = len(update.get("branch_results", []))
    elif node_name == "synthesize" and "result" in update:
        event["result"] = update["result"]
    return event


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
    baseline_unique_pmids = set(baseline_pmids)
    agent_unique_pmids = set(agent_pmids)
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
    return {
        "iteration": iteration,
        "branch_query": branch_query,
        "planner_source": planner_source,
        "retrieval_rationale": planner_rationale,
        "retrieved_count": len(retrieved_pmids),
        "evidence_count": len(evidence_pmids),
        "new_pmids": sorted(retrieved_set - pmids_before_branch),
        "overlap_pmids": sorted(retrieved_set & pmids_before_branch),
        "top_retrieval_score": round(max((candidate.score for candidate in candidates), default=0.0), 4),
        "top_relevance_score": round(max((record.relevance_score for record in evidence_records), default=0.0), 4),
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
