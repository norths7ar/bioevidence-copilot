from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from bioevidence.agent.llm import AgentLLMError, create_agent_client
from bioevidence.agent.planner import plan_next_steps_with_trace
from bioevidence.agent.state import AgentState
from bioevidence.agent.stop_criteria import should_stop
from bioevidence.agent.tools import merge_candidates, merge_evidence_records
from bioevidence.config import Settings, load_settings
from bioevidence.extraction.model_backend import ExtractionBackend, create_product_extraction_backend
from bioevidence.generation.agent_answerer import synthesize_agent_answer_with_trace
from bioevidence.graph.models import GraphDiscoveryResult
from bioevidence.graph.provider import GraphDiscoveryError, GraphDiscoveryProvider, create_graph_provider
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.trace import TraceRecorder
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


LOGGER = logging.getLogger(__name__)


def run_agent_workflow(
    query: Query,
    *,
    data_dir: Path | None = None,
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
    graph_provider: GraphDiscoveryProvider | None = None,
    trace_recorder: TraceRecorder | None = None,
    extraction_backend: ExtractionBackend | None = None,
) -> AgentWorkflowResult:
    settings = settings or load_settings()
    extraction_backend = extraction_backend or create_product_extraction_backend(settings)
    recorder = trace_recorder or TraceRecorder()
    _record_run_started(recorder, query, settings)
    provider = graph_provider or create_graph_provider(settings)
    owns_provider = graph_provider is None
    LOGGER.info(
        "agent_started top_k=%d graph_enabled=%s max_iterations=%d",
        query.top_k,
        settings.graph_enabled,
        settings.agent_max_iterations,
    )
    try:
        graph = _build_agent_graph(
            query,
            data_dir=data_dir,
            settings=settings,
            graph_provider=provider,
            trace_recorder=recorder,
            extraction_backend=extraction_backend,
        )
        output = graph.invoke(_initial_graph_state(query, documents))
        result = output["result"]
        LOGGER.info(
            "agent_completed iterations=%d branches=%d evidence=%d citations=%d stop_reason=%s",
            result.state.iterations,
            len(result.branch_results),
            len(result.evidence_records),
            len(result.answer.citations),
            result.state.stop_reason,
        )
        return result
    except Exception as exc:
        recorder.emit("run_failed", error_type=type(exc).__name__)
        LOGGER.exception("agent_failed")
        raise
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
    trace_recorder: TraceRecorder | None = None,
    extraction_backend: ExtractionBackend | None = None,
) -> Iterator[dict[str, object]]:
    settings = settings or load_settings()
    extraction_backend = extraction_backend or create_product_extraction_backend(settings)
    recorder = trace_recorder or TraceRecorder()
    _record_run_started(recorder, query, settings)
    provider = graph_provider or create_graph_provider(settings)
    owns_provider = graph_provider is None
    graph = _build_agent_graph(
        query,
        data_dir=data_dir,
        settings=settings,
        graph_provider=provider,
        trace_recorder=recorder,
        extraction_backend=extraction_backend,
    )
    emitted_event_count = 0
    try:
        for event in recorder.events():
            yield event
            emitted_event_count += 1
        for update in graph.stream(_initial_graph_state(query, documents), stream_mode="updates"):
            for event in recorder.events()[emitted_event_count:]:
                yield event
                emitted_event_count += 1
            node_name, node_update = next(iter(update.items()))
            if node_name == "synthesize" and "result" in node_update:
                yield {
                    "event": "result",
                    "run_id": recorder.run_id,
                    "result": node_update["result"],
                }
    except Exception as exc:
        failure_event = recorder.emit("run_failed", error_type=type(exc).__name__)
        yield failure_event
        raise
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
    trace_recorder: TraceRecorder,
    extraction_backend: ExtractionBackend | None,
):
    try:
        agent_client = create_agent_client(settings)
    except AgentLLMError as exc:
        LOGGER.warning("agent_backend_unavailable reason=%s", exc)
        agent_client = None

    def baseline_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        started_at = trace_recorder.start_timer()
        baseline_kwargs: dict[str, Any] = {
            "data_dir": data_dir,
            "documents": runtime.get("documents"),
            "settings": settings,
        }
        if extraction_backend is not None:
            baseline_kwargs["extraction_backend"] = extraction_backend
        baseline = run_rag_pipeline(query, **baseline_kwargs)
        state = AgentState(query=query, max_iterations=settings.agent_max_iterations)
        state.record_branch_query(query.text)
        state.merge_candidates(baseline.retrieved_candidates)
        state.merge_evidence_records(baseline.evidence_records)
        trace_recorder.emit(
            "baseline_completed",
            duration_ms=trace_recorder.elapsed_ms(started_at),
            source=baseline.source,
            candidates=len(baseline.retrieved_candidates),
            evidence=len(baseline.evidence_records),
            citations=len(baseline.answer.citations),
        )
        return {"baseline": baseline, "state": state}

    def discover_graph_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        started_at = trace_recorder.start_timer()
        try:
            discovery = graph_provider.discover(query.text)
        except GraphDiscoveryError as exc:
            LOGGER.warning("graph_discovery_fallback reason=%s", type(exc).__name__)
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
        LOGGER.info(
            "graph_discovery_completed status=%s linked_entities=%d paths=%d expansion_queries=%d",
            discovery.status,
            len(discovery.linked_entities),
            len(discovery.paths),
            len(discovery.expanded_queries),
        )
        trace_recorder.emit(
            "graph_discovery_completed",
            duration_ms=trace_recorder.elapsed_ms(started_at),
            status=discovery.status,
            linked_entities=len(discovery.linked_entities),
            paths=len(discovery.paths),
            expansion_queries=list(accepted_queries),
        )
        return {
            "graph_discovery": discovery,
            "planned_queries": list(accepted_queries),
            "planning_steps": planning_steps,
        }

    def plan_node(graph_state: AgentGraphState) -> AgentGraphState:
        runtime = graph_state
        state = runtime["state"]
        started_at = trace_recorder.start_timer()
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
        LOGGER.info(
            "planner_completed iteration=%d source=%s proposed=%d accepted=%d",
            state.iterations,
            planning_result.source,
            len(planning_result.proposed_queries),
            len(planning_result.accepted_queries),
        )
        trace_recorder.emit(
            "planner_completed",
            duration_ms=trace_recorder.elapsed_ms(started_at),
            iteration=state.iterations,
            source=planning_result.source,
            proposed=len(planning_result.proposed_queries),
            accepted_queries=list(planning_result.accepted_queries),
            fallback_reason=planning_result.fallback_reason,
        )
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
            started_at = trace_recorder.start_timer()
            branch_query_obj = Query(text=branch_query, rewritten_text=branch_query, top_k=query.top_k)
            pmids_before_branch = set(state.seen_pmids)
            retrieval_kwargs: dict[str, Any] = {
                "data_dir": data_dir,
                "documents": runtime_documents,
                "settings": settings,
            }
            if extraction_backend is not None:
                retrieval_kwargs["extraction_backend"] = extraction_backend
            returned_documents, ranked_candidates, evidence_records, source = run_retrieval_stack(
                branch_query_obj,
                **retrieval_kwargs,
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
            new_pmids = sorted(
                {candidate.document.pmid for candidate in ranked_candidates} - pmids_before_branch
            )
            LOGGER.info(
                "branch_retrieval_completed iteration=%d source=%s candidates=%d evidence=%d new_pmids=%d",
                state.iterations,
                source,
                len(ranked_candidates),
                len(evidence_records),
                len(new_pmids),
            )
            trace_recorder.emit(
                "branch_retrieval_completed",
                duration_ms=trace_recorder.elapsed_ms(started_at),
                iteration=state.iterations,
                query=branch_query,
                source=source,
                candidates=len(ranked_candidates),
                evidence=len(evidence_records),
                new_pmids=new_pmids,
            )
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
        started_at = trace_recorder.start_timer()
        synthesis = synthesize_agent_answer_with_trace(
            state,
            baseline.answer.answer_text,
            settings=settings,
            client=agent_client,
        )
        answer = synthesis.answer
        trace_recorder.emit(
            "synthesis_completed",
            duration_ms=trace_recorder.elapsed_ms(started_at),
            source=synthesis.source,
            evidence=len(state.top_evidence_records()),
            citations=len(answer.citations),
            fallback_reason=synthesis.fallback_reason,
        )
        trace_recorder.emit(
            "run_completed",
            iterations=state.iterations,
            branches=len(branch_results),
            evidence=len(state.top_evidence_records()),
            citations=len(answer.citations),
            stop_reason=state.stop_reason,
        )
        LOGGER.info(
            "synthesis_completed source=%s evidence=%d citations=%d stop_reason=%s",
            synthesis.source,
            len(state.top_evidence_records()),
            len(answer.citations),
            state.stop_reason,
        )
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
            run_id=trace_recorder.run_id,
            trace_events=trace_recorder.events(),
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


def _record_run_started(recorder: TraceRecorder, query: Query, settings: Settings) -> None:
    if recorder.events():
        return
    recorder.emit(
        "run_started",
        query=query.text,
        top_k=query.top_k,
        graph_enabled=settings.graph_enabled,
        agent_model=settings.agent_model or None,
        max_iterations=settings.agent_max_iterations,
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
