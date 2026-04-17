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

- Load the baseline corpus from local `data/processed/*.documents.jsonl` artifacts rather than introducing a database or external index.
- Use BM25-style lexical scoring over document title plus abstract, plus a deterministic overlap-based dense adapter so the hybrid retriever has two inspectable signals.
- Keep merge and rerank behavior deterministic, and return the ranked candidates alongside the final answer so the app can expose intermediate artifacts.

## 2026-04-15: Dense embedding backend

- Use Qwen `text-embedding-v4` through the OpenAI-compatible SDK for true dense retrieval.
- Load local `.env` values with `python-dotenv` so `QWEN_API_KEY` and related embedding settings work in local development.
- Cache corpus embeddings on disk keyed by corpus signature, model, and dimensions so repeated dense retrieval does not re-embed unchanged documents.
- Fall back to lexical-only ranking when the dense backend is unavailable, rather than failing the whole retrieval flow.

## 2026-04-17: Structured evidence table output

- Keep evidence extraction deterministic for milestone 3 and derive structured `EvidenceRecord` rows directly from ranked retrieval output.
- Surface the evidence table in the app and demo output as a first-class artifact, alongside the final answer and citations.
- Store a curated example output under `examples/` so reviewers can inspect the evidence-table shape without running the pipeline.

## 2026-04-17: Local evaluation harness

- Keep evaluation file-based and local by loading JSONL datasets from disk and running the existing RAG workflow per item.
- Score retrieval with hit@k, recall@k, and MRR, and score answers with citation precision / recall / F1 plus normalized exact match and token overlap when a reference answer is available.
- Return a structured evaluation report with per-item records and aggregate summary metrics, and make the CLI optionally write the full report as JSON.
