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
