from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable
from urllib.error import URLError

from bioevidence.workflows import run_agent_workflow
from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.presentation import build_agent_comparison_payload, build_evidence_csv, build_markdown_report
from bioevidence.schemas.query import Query
from bioevidence.utils.logging_config import configure_logging

try:  # pragma: no cover - optional runtime dependency
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - import smoke test path
    st = None  # type: ignore[assignment]


DEFAULT_QUERY = "What evidence exists for asthma corticosteroids?"
DEFAULT_DATA_DIR = "data/corpora/demo"
LOGGER = logging.getLogger(__name__)


def _cache_data(*decorator_args, **decorator_kwargs) -> Callable[[Callable[..., object]], Callable[..., object]]:
    if st is None:
        def _identity(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return _identity
    return st.cache_data(*decorator_args, **decorator_kwargs)


@_cache_data(show_spinner=False)
def load_demo_payload(query_text: str, data_dir: str | None = None) -> dict[str, object]:
    settings = load_settings()
    query = Query(text=query_text)
    data_path = Path(data_dir) if data_dir else None
    try:
        result = run_agent_workflow(query, data_dir=data_path, settings=settings)
    except (PubMedRequestError, URLError, OSError) as exc:
        LOGGER.warning("streamlit_agent_unavailable reason=%s", type(exc).__name__)
        return {
            "query": query.text,
            "baseline": None,
            "agent": None,
            "comparison": None,
            "branches": [],
            "state": None,
            "agent_notice": f"Agent workflow unavailable in the current environment: {exc}",
            "error": str(exc),
        }

    payload = build_agent_comparison_payload(result)
    comparison = payload.get("comparison", {})
    agent_notice = None
    if isinstance(comparison, dict) and not comparison.get("agent_backend_ready", True):
        agent_notice = "Agent backend is not configured in `.env`; showing a fallback comparison."
    payload["agent_notice"] = agent_notice
    payload["error"] = None
    return payload


def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError("streamlit is required to run the browser demo; install the project dependencies first")


def _render_result_tab(title: str, payload: dict[str, object]) -> None:
    st.subheader(title)
    st.markdown(f"**Retrieval source:** `{payload['retrieval_source']}`")
    st.markdown(f"**Rewritten query:** {payload['rewritten_query']}")
    st.markdown(f"**Evidence rows:** {payload['evidence_count']}")
    citations = ", ".join(payload["citations"]) if payload["citations"] else "(none)"
    st.markdown(f"**Citations:** {citations}")
    st.markdown("**Answer**")
    st.write(payload["answer"])

    st.markdown("**Retrieved papers**")
    st.dataframe(payload["retrieved_papers"], hide_index=True, use_container_width=True)

    st.markdown("**Structured evidence**")
    _render_evidence_console(payload, key_prefix=title.lower().replace(" ", "_"))


def _render_evidence_console(payload: dict[str, object], *, key_prefix: str) -> None:
    rows = _normalize_rows(payload.get("evidence_table", []))
    if not rows:
        st.info("No structured evidence rows are available.")
        return

    controls = st.columns([1, 1, 1, 1])
    entity_options = _entity_options(rows)
    journal_options = _journal_options(rows)
    selected_entities = controls[0].multiselect("Entity", entity_options, key=f"{key_prefix}_entities")
    selected_journal = controls[1].selectbox("Journal", ["All", *journal_options], key=f"{key_prefix}_journal")
    min_relevance = controls[2].slider(
        "Minimum relevance",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key=f"{key_prefix}_min_relevance",
    )
    sort_by = controls[3].selectbox(
        "Sort",
        ["Relevance high to low", "Year newest", "Year oldest", "PMID"],
        key=f"{key_prefix}_sort",
    )
    filtered_rows = _filter_sort_evidence_rows(
        rows,
        selected_entities=selected_entities,
        selected_journal=selected_journal,
        min_relevance=min_relevance,
        sort_by=sort_by,
    )
    st.caption(f"Showing {len(filtered_rows)} of {len(rows)} evidence rows.")
    st.dataframe(filtered_rows, hide_index=True, use_container_width=True)


def _build_run_summary(payload: dict[str, object]) -> dict[str, object]:
    baseline = payload.get("baseline")
    agent = payload.get("agent")
    comparison = payload.get("comparison")
    state = payload.get("state")
    branches = payload.get("branches", [])

    baseline_source = baseline.get("retrieval_source") if isinstance(baseline, dict) else "unavailable"
    agent_source = agent.get("retrieval_source") if isinstance(agent, dict) else "unavailable"
    comparison = comparison if isinstance(comparison, dict) else {}
    state = state if isinstance(state, dict) else {}
    branches = branches if isinstance(branches, list) else []

    branch_count = int(comparison.get("branch_count", len(branches)) or 0)
    iterations = int(comparison.get("iterations", state.get("iterations", 0)) or 0)
    stop_reason = str(comparison.get("stop_reason", state.get("stop_reason", "unknown")) or "unknown")
    backend_ready = bool(comparison.get("agent_backend_ready", False))
    agent_status = "expanded" if branch_count > 0 else "baseline sufficient"

    return {
        "baseline_source": baseline_source,
        "agent_source": agent_source,
        "agent_status": agent_status,
        "agent_backend": "configured" if backend_ready else "fallback",
        "branch_count": branch_count,
        "iterations": iterations,
        "stop_reason": stop_reason,
    }


def _normalize_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _entity_options(rows: list[dict[str, object]]) -> list[str]:
    entities: set[str] = set()
    for row in rows:
        row_entities = row.get("entities", [])
        if isinstance(row_entities, list):
            entities.update(str(entity) for entity in row_entities if str(entity).strip())
    return sorted(entities)


def _journal_options(rows: list[dict[str, object]]) -> list[str]:
    return sorted({str(row.get("journal", "")).strip() for row in rows if str(row.get("journal", "")).strip()})


def _filter_sort_evidence_rows(
    rows: list[dict[str, object]],
    *,
    selected_entities: list[str],
    selected_journal: str,
    min_relevance: float,
    sort_by: str,
) -> list[dict[str, object]]:
    selected_entity_set = set(selected_entities)
    filtered: list[dict[str, object]] = []
    for row in rows:
        row_entities = row.get("entities", [])
        row_entity_set = set(str(entity) for entity in row_entities) if isinstance(row_entities, list) else set()
        relevance = _as_float(row.get("relevance_score"))
        journal = str(row.get("journal", ""))
        if selected_entity_set and not selected_entity_set.intersection(row_entity_set):
            continue
        if selected_journal != "All" and journal != selected_journal:
            continue
        if relevance < min_relevance:
            continue
        filtered.append(row)

    if sort_by == "Year newest":
        return sorted(filtered, key=lambda row: (_as_int(row.get("year")), str(row.get("pmid", ""))), reverse=True)
    if sort_by == "Year oldest":
        return sorted(filtered, key=lambda row: (_as_int(row.get("year")), str(row.get("pmid", ""))))
    if sort_by == "PMID":
        return sorted(filtered, key=lambda row: str(row.get("pmid", "")))
    return sorted(filtered, key=lambda row: (_as_float(row.get("relevance_score")), str(row.get("pmid", ""))), reverse=True)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _render_run_summary(payload: dict[str, object]) -> None:
    summary = _build_run_summary(payload)
    st.markdown("**Run diagnostics**")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Data source", str(summary["baseline_source"]))
    metric_columns[1].metric("Agent status", str(summary["agent_status"]))
    metric_columns[2].metric("Branches", int(summary["branch_count"]))
    metric_columns[3].metric("Iterations", int(summary["iterations"]))

    st.caption(
        "Agent backend: "
        f"{summary['agent_backend']} | Agent source: {summary['agent_source']} | Stop reason: {summary['stop_reason']}"
    )


def _render_agent_diagnostics(payload: dict[str, object]) -> None:
    summary = _build_run_summary(payload)
    branch_count = int(summary["branch_count"])
    iterations = int(summary["iterations"])
    stop_reason = str(summary["stop_reason"])

    if branch_count > 0:
        st.success(f"Agent expanded retrieval with {branch_count} branch queries over {iterations} iteration(s).")
    else:
        st.info(f"Agent did not add branch retrieval because the baseline evidence was sufficient. Stop reason: {stop_reason}.")

    st.markdown("**Agent run summary**")
    st.table(
        [
            {"signal": "Agent backend", "value": summary["agent_backend"]},
            {"signal": "Agent status", "value": summary["agent_status"]},
            {"signal": "Branch queries", "value": str(branch_count)},
            {"signal": "Iterations", "value": str(iterations)},
            {"signal": "Stop reason", "value": stop_reason},
        ]
    )


def _render_agent_trace(payload: dict[str, object]) -> None:
    trace = payload.get("trace")
    if not isinstance(trace, dict):
        return

    st.markdown("**Search trace**")
    trace_summary = _build_trace_summary(payload)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Baseline PMIDs", int(trace_summary["baseline_unique_pmids"]))
    metric_columns[1].metric("Agent PMIDs", int(trace_summary["agent_unique_pmids"]))
    metric_columns[2].metric("New PMIDs", int(trace_summary["new_pmids"]))
    metric_columns[3].metric("Stop reason", str(trace_summary["stop_reason"]))
    st.table(_build_trace_rows(trace))

    planning_steps = trace.get("planning_steps", [])
    if isinstance(planning_steps, list) and planning_steps:
        st.markdown("**Planning steps**")
        st.dataframe(_build_planning_rows(planning_steps), hide_index=True, use_container_width=True)

    branches = trace.get("branch_diagnostics", [])
    if isinstance(branches, list) and branches:
        st.markdown("**Branch diagnostics**")
        st.dataframe(_build_branch_rows(branches), hide_index=True, use_container_width=True)


def _build_trace_summary(payload: dict[str, object]) -> dict[str, object]:
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    coverage = trace.get("retrieval_coverage") if isinstance(trace.get("retrieval_coverage"), dict) else {}
    stop = trace.get("stop") if isinstance(trace.get("stop"), dict) else {}
    return {
        "baseline_unique_pmids": len(coverage.get("baseline_unique_pmids", [])),
        "agent_unique_pmids": len(coverage.get("agent_unique_pmids", [])),
        "new_pmids": len(coverage.get("new_pmids_over_baseline", [])),
        "stop_reason": stop.get("reason", "unknown"),
    }


def _build_trace_rows(trace: dict[str, object]) -> list[dict[str, object]]:
    stop = trace.get("stop") if isinstance(trace.get("stop"), dict) else {}
    return [
        {"signal": "Original query", "value": trace.get("original_query", "")},
        {"signal": "Rewritten query", "value": trace.get("rewritten_query", "")},
        {"signal": "Evidence sufficient", "value": str(stop.get("sufficient", False))},
        {"signal": "Iterations", "value": f"{stop.get('iterations', 0)} / {stop.get('max_iterations', 0)}"},
    ]


def _build_planning_rows(planning_steps: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in planning_steps:
        if not isinstance(step, dict):
            continue
        rows.append(
            {
                "iteration": step.get("iteration"),
                "source": step.get("source"),
                "proposed_queries": ", ".join(str(query) for query in step.get("proposed_queries", [])),
                "accepted_queries": ", ".join(str(query) for query in step.get("accepted_queries", [])),
                "rationale": step.get("rationale"),
            }
        )
    return rows


def _build_branch_rows(branches: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        diagnostics = branch.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        rows.append(
            {
                "query": branch.get("query"),
                "new_pmids": ", ".join(str(pmid) for pmid in diagnostics.get("new_pmids", [])),
                "overlap_pmids": ", ".join(str(pmid) for pmid in diagnostics.get("overlap_pmids", [])),
                "retrieved_count": diagnostics.get("retrieved_count"),
                "evidence_count": diagnostics.get("evidence_count"),
                "top_relevance_score": diagnostics.get("top_relevance_score"),
                "stop_reason_after_branch": diagnostics.get("stop_reason_after_branch"),
            }
        )
    return rows


def _render_exports(payload: dict[str, object]) -> None:
    agent = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    evidence_rows = _normalize_rows(agent.get("evidence_table", []))
    export_columns = st.columns(3)
    export_columns[0].download_button(
        "Download JSON",
        data=json.dumps(payload, indent=2, sort_keys=True),
        file_name="bioevidence-report.json",
        mime="application/json",
    )
    export_columns[1].download_button(
        "Download Markdown",
        data=build_markdown_report(payload),
        file_name="bioevidence-report.md",
        mime="text/markdown",
    )
    export_columns[2].download_button(
        "Download evidence CSV",
        data=build_evidence_csv(evidence_rows),
        file_name="bioevidence-evidence.csv",
        mime="text/csv",
    )


def main() -> None:
    _require_streamlit()
    configure_logging(
        load_settings().log_level,
        log_file=Path("artifacts/logs/streamlit.log"),
    )
    st.set_page_config(
        page_title="BioEvidence Copilot",
        page_icon="🧬",
        layout="wide",
    )
    st.title("BioEvidence Copilot")
    st.caption("A biomedical evidence assistant built around PubMed retrieval, structured evidence, and agentic comparison.")

    with st.form("query_form", clear_on_submit=False):
        query_text = st.text_input("Biomedical question", value=DEFAULT_QUERY)
        data_dir = st.text_input("Optional data directory", value=DEFAULT_DATA_DIR)
        submitted = st.form_submit_button("Run demo")

    if not submitted:
        st.info("Enter a query and run the demo to compare baseline and agent outputs.")
        return

    query_text = query_text.strip()
    if not query_text:
        st.warning("Please enter a biomedical question.")
        return

    payload = load_demo_payload(query_text, data_dir.strip() or None)
    st.markdown(f"**Query:** {payload['query']}")

    if payload.get("agent_notice"):
        st.warning(payload["agent_notice"])
    if payload.get("error"):
        st.warning(payload["error"])

    baseline = payload.get("baseline")
    agent = payload.get("agent")

    if baseline is None:
        st.error("Could not build a demo result for this query in the current environment.")
        return

    _render_run_summary(payload)
    _render_exports(payload)

    baseline_tabs = st.tabs(["Baseline", "Agent"])

    with baseline_tabs[0]:
        _render_result_tab("Baseline RAG", baseline)

    with baseline_tabs[1]:
        if agent is None:
            st.info("Agent result is unavailable for this query.")
        else:
            _render_agent_diagnostics(payload)
            _render_result_tab("Agent workflow", agent)
            _render_agent_trace(payload)
            comparison = payload.get("comparison")
            if comparison:
                st.markdown("**Comparison metadata**")
                st.json(comparison)
            branches = payload.get("branches", [])
            if branches:
                st.markdown("**Agent branches**")
                st.table(branches)

    with st.expander("Raw JSON payload", expanded=False):
        st.json(payload)


if __name__ == "__main__":
    main()
