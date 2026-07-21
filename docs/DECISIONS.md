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

- Use an OpenAI-compatible embedding backend configured through generic `EMBEDDING_*` fields in `.env`.
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
- Use generic `AGENT_*` environment variables for the agent backend so provider choice stays in `.env` rather than in code.
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

## 2026-07-02: CI quality gates and documentation closeout

- Run GitHub Actions on push and pull request with Ruff linting, focused mypy type checking, the pytest suite, and a one-item baseline evaluation smoke test.
- Keep the type-checking gate focused on stable schema, evaluation, and workflow modules rather than forcing whole-repository strict typing before the exploratory layers settle.
- Use `--no-sqlite-cache --no-incremental` for mypy because the local Windows environment showed SQLite cache I/O errors; the focused check remains deterministic without cache.
- Treat the evaluation smoke test as a workflow integrity check, not a benchmark or model-quality claim.
- Keep documentation split by audience: README for the shortest path, `DEMO_SCRIPT.md` for walkthroughs, `EVALUATION.md` for metrics and datasets, and `LIMITATIONS.md` for medical and engineering boundaries.

## 2026-07-20: v0.1 release boundary and GraphRAG integration

- Freeze the completed literature evidence assistant at the `v0.1.0` tag
  before introducing knowledge-graph and orchestration-runtime changes.
- Integrate useful Hetionet capabilities into this repository rather than
  preserving a second product-shaped API, Docker image, CI workflow, and
  evaluation shell from `biomedical-graphrag`.
- Treat Hetionet as a discovery and query-expansion source. Graph paths can
  explain why a literature search was broadened, but only retrieved papers can
  support final answer citations.
- Access Neo4j behind an optional provider boundary so baseline workflows,
  local fixtures, and CI do not require a graph database.
- Use LangGraph for routing and streaming while retaining project-owned
  planner, retrieval, evidence, stopping, and synthesis functions as domain
  nodes.
- Evaluate graph augmentation against relevant PMIDs and report deltas over the
  literature baseline. Do not reuse graph-generated pseudo ground truth.
- Add Docker Compose only for local FastAPI plus Neo4j composition. Defer React,
  hosted deployment, and a general-purpose application database.
- Keep Streamlit in a `web` extra so the FastAPI container does not install the
  UI-only Pandas, PyArrow, and Streamlit dependency chain.

## 2026-07-20: Canonical Neo4j runtime and environment contract

- Use the Docker Compose Neo4j service and its named volumes as the canonical
  local product runtime. Neo4j Desktop is not required to run the product.
- Keep Hetionet ingestion explicit: Compose creates an empty database, and the
  import script rebuilds it from the external Hetionet source files.
- Use purpose-grouped environment variables (`AGENT_*`, `EMBEDDING_*`,
  `NEO4J_*`, and `GRAPH_*`) instead of a repository-wide `BIOEVIDENCE_*`
  prefix.
- Reserve `NEO4J_*` for connection settings and `GRAPH_*` for discovery-layer
  behavior so local scripts and containers share one unambiguous contract.
- Default agent output capacity to 8192 tokens because reasoning models consume
  the same completion budget for internal reasoning and structured response
  text. Keep a bounded default rather than allowing unbounded model output.
- Let the Compose API service read the local `.env` for provider credentials,
  while overriding container-only paths and the Neo4j service-network address.

## 2026-07-20: Application logging boundary

- Keep logging helpers local to `bioevidence.utils` instead of restoring the
  separately installed `myutils` dependency.
- Configure the root logger once at each application entrypoint and use
  module-level `logging.getLogger(__name__)` instances throughout the package.
- Write logs to the process stream by default so Docker and local runners use
  the same behavior; do not create implicit per-script log directories.
- Keep third-party HTTP and Neo4j INFO traffic quiet and log application
  lifecycle counts, fallback reasons, and failures without prompts, abstracts,
  credentials, or complete PMID collections.
- Retain CLI logs beside their report and trace in a timestamped run directory;
  keep Streamlit logs in one rotating file and use Docker log rotation for the
  long-running API service.
- Separate user-facing reports from execution traces. Store evidence once in
  the compact report, write ordered execution events as JSONL, and generate the
  full internal payload only when `--debug` is explicitly requested.
- Reuse the same event schema for saved traces and the FastAPI NDJSON stream so
  the service does not maintain a second tracing contract.
- This supersedes the Milestone 9 decision to keep trace, report, and complete
  internal state in one JSON payload; the Streamlit review payload remains an
  in-memory presentation model rather than the persisted report format.

## 2026-07-21: Fine-tuned evidence extraction boundary

- Keep fine-tuning inside the BioEvidence Copilot repository because the model
  is one evidence-pipeline capability, not an independently released product.
- Keep future training and offline evaluation under `training/`, runtime model
  adapters under `src/bioevidence/extraction/`, and model weights in an external
  registry or ignored local artifact directory.
- Define a separate `ModelEvidenceExtraction` contract for semantic predictions;
  do not ask the model to regenerate PMID, title, year, journal, or retrieval
  relevance scores that the workflow already owns deterministically.
- Require query-focused evidence status, explicit nulls, closed enums, and
  verbatim abstract evidence spans so structured output can be validated and
  evaluated beyond JSON parse success.
- Preserve the current deterministic extractor as a baseline and compare it
  with a prompted base model before making fine-tuning quality claims.
- Treat the first tracked annotations as schema-development drafts and retain
  annotation/review provenance when promoting any labels into a benchmark.
- Keep the v0.3 sequence evaluation-first: stabilize the contract and pilot,
  establish baselines, expand reviewed data, train, then add optional inference.

## 2026-07-21: Shared extraction baseline interface

- Put rule-based and prompt-only semantic extraction behind the same typed
  `ExtractionBackend` contract and validate both against
  `ModelEvidenceExtraction`.
- Keep extraction model credentials and model selection under `EXTRACTION_*`
  rather than coupling experiments to the agent planner/synthesizer settings.
- Record parse, schema, grounding, field-quality, and latency results per item;
  preserve raw failed model output for diagnosis.
- Treat provider cost as experiment metadata rather than guessing it from token
  counts when an OpenAI-compatible endpoint has no stable price contract.
