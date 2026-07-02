# DECISIONS

## 2026-04-14: Milestone 0 scaffold choices

- Use Python 3.12 as the project target.
- Use a `src/` layout for the importable package.
- Keep the first milestone flat and conservative: dataclass schemas, module stubs, and simple scripts rather than framework-heavy abstractions.
- Keep Stage 1 centered on PubMed metadata and abstracts; defer full-text parsing and real agent orchestration.

## 2026-04-14: PubMed ingestion implementation shape

- Use the NCBI E-utilities API directly from the standard library instead of adding a new HTTP client dependency.
- Keep PubMed ingestion inspectable by saving raw search JSON, raw fetch XML, processed JSONL documents, and a manifest.
- Keep parsing logic local and explicit in the ingestion module so tests can inject fake responses without network access.

## 2026-04-14: PubMed transport policy

- Use a module-local request policy for PubMed rather than a shared networking framework.
- Default to a fixed request timeout, retry transient network failures and 429/5xx responses, and surface a clear `PubMedRequestError` after retries are exhausted.
- Keep retry/backoff behavior deterministic and testable by injecting fake openers and monkeypatching sleep in tests.

## 2026-04-14: Milestone 2 retrieval baseline

- Load the baseline corpus from local `processed/*.documents.jsonl` artifacts under the configured corpus directory rather than introducing a database or external index.
- Use BM25-style lexical scoring over document title plus abstract, plus a deterministic overlap-based dense adapter so the hybrid retriever has two inspectable signals.
- Keep merge and final ranking behavior deterministic, and return the ranked candidates alongside the final answer so the app can expose intermediate artifacts.

## 2026-04-15: Dense embedding backend

- Use an OpenAI-compatible embedding backend configured through generic `BIOEVIDENCE_EMBEDDING_*` fields in `.env`.
- Keep the implementation provider-agnostic in code; the example configuration can point at Qwen `text-embedding-v4` or another OpenAI-compatible provider without code changes.
- Cache corpus embeddings on disk keyed by corpus signature, model, and dimensions so repeated dense retrieval does not re-embed unchanged documents.
- Fall back to lexical-only ranking when the dense backend is unavailable, rather than failing the whole retrieval flow.

## 2026-04-17: Structured evidence table output

- Keep evidence extraction deterministic for milestone 3 and derive structured `EvidenceRecord` rows directly from ranked retrieval output.
- Surface the evidence table in the app and demo output as a first-class artifact, alongside the final answer and citations.
- Store real demo corpus and evaluation artifacts under `data/` so reviewers can inspect the evidence-table shape without relying on fabricated examples.

## 2026-04-17: Local evaluation harness

- Keep evaluation file-based and local by loading JSONL datasets from disk and running the existing RAG workflow per item.
- Score retrieval with hit@k, recall@k, and MRR, and score answers with citation precision / recall / F1 plus normalized exact match and token overlap when a reference answer is available.
- Return a structured evaluation report with per-item records and aggregate summary metrics, and make the CLI optionally write the full report as JSON.

## 2026-04-17: Custom agentic orchestration

- Keep the agent controller custom and lightweight instead of adopting LangChain or LangGraph for the first agent milestone.
- Use generic `BIOEVIDENCE_AGENT_*` environment variables for the agent backend so provider choice stays in `.env` rather than in code.
- Keep the agent backend OpenAI-compatible so DeepSeek, Qwen Chat, MiMo, and similar providers can be swapped without code changes.
- Keep sufficiency deterministic: stop when the loop has accumulated enough unique PMIDs with a minimum relevance floor, otherwise continue until max iterations.
- Surface the agent report in CLI / JSON form and let callers write real report artifacts for reviewability.

## 2026-04-17: Streamlit presentation layer

- Use Streamlit only as a thin presentation layer on top of the existing workflow outputs, not as a second place for business logic.
- Show baseline RAG and agent outputs in tabs so the comparison is easy to inspect in a browser.
- Keep the browser demo aligned with the CLI/demo helpers by normalizing workflow results into shared presentation payloads.
- Document baseline answer generation honestly as evidence stitching / templated synthesis and keep the final ranking step explicitly deterministic rather than learned.

## 2026-05-29: Demo evaluation and quality checks

- Track curated demo evaluation and converted corpus artifacts under `data/`, while keeping large raw downloads and caches ignored.
- Extend the existing file-based evaluation report instead of introducing a separate experiment tracker.
- Support both baseline and agent evaluation modes through the same runner so reports stay comparable.
- Add deterministic citation/evidence quality checks before considering any LLM-as-judge evaluation.
- Keep PICO-like enrichment as derived report metadata for now rather than expanding the core `EvidenceRecord` schema prematurely.

## 2026-05-29: Real demo data preparation

- Seed the first real corpus through PubMed E-utilities using a small set of biomedical demo topics.
- Store the combined PubMed corpus as `data/corpora/demo/processed/demo.documents.jsonl`, matching the existing retriever's `*.documents.jsonl` convention.
- Convert BioASQ Task B questions into the existing evaluation JSONL schema and convert snippets into the same document JSONL shape.
- Ignore local BioASQ raw zip/extracted files because they are large external benchmark inputs, not curated project artifacts.

## 2026-05-29: Evaluation corpus reuse

- Preload `data_dir` corpora once in `run_evaluation()` and pass the documents into each workflow call.
- Keep workflow functions able to accept explicit `documents` so evaluation does not repeatedly read the same `*.documents.jsonl` corpus.
- Add a CLI `--limit` option for BioASQ smoke runs because the full dataset is thousands of queries over tens of thousands of snippet documents.

## 2026-05-29: FastAPI service boundary

- Add FastAPI as a thin service layer around the existing workflow functions.
- Keep retrieval, generation, extraction, evaluation, and agent orchestration in `src/bioevidence/`.
- Expose baseline and agent query endpoints first; defer evaluation endpoints, background jobs, auth, Docker, and Streamlit API-client conversion.
- Treat the API as an additional backend interface, not a replacement for the local CLI and Streamlit demo paths.

## 2026-06-02: Interface and workflow cleanup

- Move external entrypoints under `interfaces/` so the Streamlit UI and FastAPI API are clearly separate from core package logic.
- Move baseline and agent orchestration into `src/bioevidence/workflows/`, keeping `src/bioevidence/agent/` focused on agent-specific planner, state, tools, and LLM helpers.
- Rename the deterministic ranking step from `rerank.py` to `ranking.py` to avoid implying a learned reranking model.
- Remove placeholder notebooks, fake milestone examples, empty app pages, stale bytecode caches, and unused scaffold scripts.

## 2026-07-01: Agent traceability surface

- Keep agent traceability as structured workflow output instead of adding a separate tracing framework.
- Preserve the existing agent report shape while adding a `trace` payload with original query, rewritten query, planning steps, branch diagnostics, retrieval coverage, and deterministic stop metadata.
- Keep planner compatibility by retaining the list-returning `plan_next_steps()` helper and adding a traced planner result for workflow use.
- Surface the same trace payload through CLI JSON, FastAPI responses, and the Streamlit review console so branch planning and coverage improvements are inspectable from every demo path.

## 2026-07-01: Streamlit review console polish

- Keep Streamlit as a read-only review console over normalized presentation payloads, not as a second workflow implementation.
- Add evidence-table filtering, sorting, and wider dataframe views in the web interface while keeping the underlying evidence rows unchanged.
- Add Markdown, JSON, and CSV exports from presentation payloads so demo results can be shared without rerunning the workflow.
- Keep trace summaries and branch diagnostics table-shaped for reviewer inspection instead of relying on raw JSON as the primary view.

## 2026-07-02: FastAPI Docker packaging

- Package only the FastAPI service path in Docker; keep Streamlit and conda-based commands as local development and demo workflows.
- Install the project with the `serve` extra so `uvicorn` is available inside the image while keeping business logic in `src/bioevidence/`.
- Copy curated local corpus artifacts into the image so `/api/v1/health` and local-corpus query paths can run without external downloads.
- Route embedding cache writes to `/tmp/bioevidence-cache` and run the service as a non-root user.
- Keep Docker configuration environment-driven so `.env` can be supplied at runtime without baking secrets into the image.
