from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from bioevidence.workflows import AgentWorkflowResult, WorkflowResult
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
        "trace": build_agent_trace_payload(result),
        "state": {
            "iterations": result.state.iterations,
            "max_iterations": result.state.max_iterations,
            "branch_queries": list(result.state.branch_queries),
            "unique_pmids": sorted(result.state.seen_pmids),
            "sufficient": result.state.sufficient,
            "stop_reason": result.state.stop_reason,
        },
    }


def build_agent_report_payload(result: AgentWorkflowResult) -> dict[str, object]:
    baseline_evidence_pmids = [record.pmid for record in result.baseline.evidence_records]
    agent_evidence_pmids = [record.pmid for record in result.evidence_records]
    baseline_evidence_set = set(baseline_evidence_pmids)
    agent_evidence_set = set(agent_evidence_pmids)
    baseline_citations = set(result.baseline.answer.citations)
    agent_citations = set(result.answer.citations)

    evidence_by_pmid = {record.pmid: record for record in result.baseline.evidence_records}
    evidence_by_pmid.update({record.pmid: record for record in result.evidence_records})
    evidence = []
    for pmid, record in evidence_by_pmid.items():
        evidence.append(
            {
                "pmid": pmid,
                "title": record.title,
                "year": record.year,
                "journal": record.journal,
                "entities": list(record.entities),
                "summary": record.summary,
                "relevance_score": round(record.relevance_score, 4),
                "in_baseline": pmid in baseline_evidence_set,
                "in_agent": pmid in agent_evidence_set,
                "cited_by": [
                    name
                    for name, citations in (("baseline", baseline_citations), ("agent", agent_citations))
                    if pmid in citations
                ],
            }
        )

    graph = result.graph_discovery
    graph_summary = None
    if graph is not None:
        graph_summary = {
            "status": graph.status,
            "linked_entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "label": entity.label,
                    "score": entity.score,
                    "match_type": entity.match_type,
                }
                for entity in graph.linked_entities
            ],
            "path_count": len(graph.paths),
            "expanded_queries": list(graph.expanded_queries),
        }

    return {
        "schema_version": 1,
        "run": {"run_id": result.run_id, "status": "completed"},
        "query": {
            "original": result.query.text,
            "rewritten": result.answer.rewritten_query or result.query.text,
        },
        "baseline": {
            "source": result.baseline.source,
            "answer": result.baseline.answer.answer_text,
            "citations": list(result.baseline.answer.citations),
            "evidence_pmids": baseline_evidence_pmids,
        },
        "agent": {
            "source": result.source,
            "answer": result.answer.answer_text,
            "citations": list(result.answer.citations),
            "evidence_pmids": agent_evidence_pmids,
        },
        "comparison": {
            "branch_count": len(result.branch_results),
            "iterations": result.state.iterations,
            "baseline_evidence_count": len(baseline_evidence_set),
            "agent_evidence_count": len(agent_evidence_set),
            "new_evidence_pmids": sorted(agent_evidence_set - baseline_evidence_set),
            "citation_overlap": sorted(baseline_citations & agent_citations),
            "agent_backend_ready": result.comparison.get("agent_backend_ready", False),
        },
        "planning": [
            {
                "iteration": step.iteration,
                "source": step.source,
                "accepted_queries": list(step.accepted_queries),
                "rationale": step.rationale,
            }
            for step in result.planning_steps
        ],
        "graph": graph_summary,
        "stop": {
            "reason": result.state.stop_reason,
            "sufficient": result.state.sufficient,
            "iterations": result.state.iterations,
            "max_iterations": result.state.max_iterations,
        },
        "evidence": evidence,
    }


def build_agent_trace_payload(result: AgentWorkflowResult) -> dict[str, object]:
    return {
        "original_query": result.query.text,
        "rewritten_query": result.answer.rewritten_query or result.query.text,
        "planning_steps": [step.to_dict() for step in result.planning_steps],
        "branch_diagnostics": [branch.to_dict() for branch in result.branch_results],
        "graph_discovery": result.graph_discovery.to_dict() if result.graph_discovery else None,
        "retrieval_coverage": result.comparison.get("retrieval_coverage", {}),
        "stop": {
            "reason": result.state.stop_reason,
            "sufficient": result.state.sufficient,
            "iterations": result.state.iterations,
            "max_iterations": result.state.max_iterations,
        },
    }


def render_agent_run_summary(result: AgentWorkflowResult) -> str:
    graph = result.graph_discovery
    graph_text = "unavailable"
    if graph is not None:
        graph_text = (
            f"{graph.status}, {len(graph.linked_entities)} entities, "
            f"{len(graph.paths)} paths, {len(graph.expanded_queries)} expansions"
        )
    planner_sources = " -> ".join(step.source for step in result.planning_steps) or "none"
    baseline_pmids = {record.pmid for record in result.baseline.evidence_records}
    agent_pmids = {record.pmid for record in result.evidence_records}
    synthesis_event = next(
        (event for event in reversed(result.trace_events) if event.get("event") == "synthesis_completed"),
        {},
    )
    return "\n".join(
        [
            f"Run ID: {result.run_id or 'unavailable'}",
            f"Graph: {graph_text}",
            f"Planner: {planner_sources}",
            f"Evidence: baseline {len(baseline_pmids)} -> agent {len(agent_pmids)}",
            f"New evidence: {len(agent_pmids - baseline_pmids)}",
            f"Synthesis: {synthesis_event.get('source', 'unknown')}",
            f"Stop: {result.state.stop_reason} after {result.state.iterations} iterations",
        ]
    )


def build_markdown_report(payload: dict[str, object]) -> str:
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    agent = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    stop = trace.get("stop") if isinstance(trace.get("stop"), dict) else {}

    lines = [
        "# BioEvidence Copilot Report",
        "",
        f"Query: {payload.get('query', '')}",
        f"Agent status: {comparison.get('stop_reason', stop.get('reason', 'unknown'))}",
        f"Agent improved retrieval coverage: {comparison.get('agent_improved_retrieval_coverage', False)}",
        "",
        "## Baseline Answer",
        "",
        str(baseline.get("answer", "")),
        "",
        "## Agent Answer",
        "",
        str(agent.get("answer", "")),
        "",
        "## Evidence",
        "",
        "| PMID | Year | Journal | Relevance | Title |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in _normalize_evidence_rows(agent.get("evidence_table", [])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("pmid", "")),
                    "" if row.get("year") is None else str(row.get("year", "")),
                    _markdown_cell(row.get("journal", "")),
                    str(row.get("relevance_score", "")),
                    _markdown_cell(row.get("title", "")),
                ]
            )
            + " |"
        )

    planning_steps = trace.get("planning_steps", [])
    if isinstance(planning_steps, list) and planning_steps:
        lines.extend(["", "## Search Trace", ""])
        for step in planning_steps:
            if not isinstance(step, dict):
                continue
            accepted_queries = ", ".join(str(query) for query in step.get("accepted_queries", []))
            lines.append(
                f"- Iteration {step.get('iteration')}: {accepted_queries} "
                f"({step.get('source', 'unknown')}) - {step.get('rationale', '')}"
            )

    return "\n".join(lines).strip() + "\n"


def build_evidence_csv(rows: list[dict[str, object]]) -> str:
    output = StringIO()
    fieldnames = ["pmid", "year", "journal", "relevance_score", "entities", "title", "summary"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in _normalize_evidence_rows(rows):
        normalized = dict(row)
        entities = normalized.get("entities", [])
        if isinstance(entities, list):
            normalized["entities"] = "; ".join(str(entity) for entity in entities)
        writer.writerow(normalized)
    return output.getvalue()


def _normalize_evidence_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_demo_output(result: WorkflowResult) -> str:
    return render_evidence_table(result.evidence_records)
