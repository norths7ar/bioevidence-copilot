# ARCHITECTURE

## System overview

The system is organized into six main layers:

1. Ingestion
2. Retrieval
3. Generation
4. Extraction
5. Agent orchestration
6. Evaluation

These layers should remain loosely coupled.

## Recommended initial flow

### Stage 1 flow
User query
-> retrieval query normalization
-> PubMed candidate fetch / local index lookup
-> hybrid retrieval
-> deterministic final ranking
-> evidence extraction
-> answer generation
-> app output

### Stage 2 flow
User query
-> planner
-> one or more retrieval branches
-> planning and branch diagnostics
-> deduplication / merge
-> sufficiency check
-> evidence extraction
-> answer generation
-> final output

The agent implementation should remain a custom controller over the existing
retrieval and extraction layers. The baseline templated answer path remains
available for comparison, while the agent uses an OpenAI-compatible LLM backend
for planning and final synthesis. Agent workflow output includes a structured
trace payload with planning steps, branch-level retrieval diagnostics, coverage
comparison against the baseline, and deterministic stopping metadata. The app
surface stays lightweight; agent comparison is exposed through CLI / JSON report
artifacts and a read-only Streamlit review console rather than a heavier
interactive UI.

## Data model expectations
At minimum define schemas for:
- Query
- Document
- RetrievedCandidate
- EvidenceRecord
- AnswerBundle

## App contract
The app should make intermediate artifacts visible:
- rewritten query if any
- top retrieved papers
- evidence table
- final answer

The browser demo is a thin Streamlit presentation layer that renders baseline
and agent outputs in tabs while reusing the same normalized view payloads as
the CLI and demo scripts.

Agent demo payloads should also expose:
- original and rewritten query
- planner source and rationale for each planning step
- accepted branch queries
- branch-level new / overlapping PMIDs
- stopping reason and evidence sufficiency status

## Interface layout

External entrypoints are grouped under `interfaces/`:

- `interfaces/web/`: Streamlit browser demo
- `interfaces/api/`: FastAPI service boundary

Core workflow orchestration lives under `src/bioevidence/workflows/` so the
baseline RAG path, agent workflow, and retrieval stack are not coupled to the
UI, API, or agent-specific helper modules.

## Evaluation flow
Evaluation should stay local and file-based:

- load JSONL items from disk
- run the existing workflow per item
- compute retrieval and answer metrics
- emit a summary report plus per-item results that can be written as JSON

## Initial implementation constraints
- start with local development only
- use environment variables for secrets
- keep external dependencies moderate
- do not commit to a heavyweight framework too early
