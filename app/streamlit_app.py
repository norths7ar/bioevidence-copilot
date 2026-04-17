from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.error import URLError

from bioevidence.agent.workflow import run_agent_workflow
from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.presentation import build_agent_comparison_payload
from bioevidence.schemas.query import Query

try:  # pragma: no cover - optional runtime dependency
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - import smoke test path
    st = None  # type: ignore[assignment]


DEFAULT_QUERY = "What evidence exists for asthma corticosteroids?"


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
    st.table(payload["retrieved_papers"])

    st.markdown("**Structured evidence**")
    st.table(payload["evidence_table"])


def main() -> None:
    _require_streamlit()
    st.set_page_config(
        page_title="BioEvidence Copilot",
        page_icon="🧬",
        layout="wide",
    )
    st.title("BioEvidence Copilot")
    st.caption("A biomedical evidence assistant built around PubMed retrieval, structured evidence, and agentic comparison.")

    with st.form("query_form", clear_on_submit=False):
        query_text = st.text_input("Biomedical question", value=DEFAULT_QUERY)
        data_dir = st.text_input("Optional data directory", value="")
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

    baseline_tabs = st.tabs(["Baseline", "Agent"])

    with baseline_tabs[0]:
        _render_result_tab("Baseline RAG", baseline)

    with baseline_tabs[1]:
        if agent is None:
            st.info("Agent result is unavailable for this query.")
        else:
            _render_result_tab("Agent workflow", agent)
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
