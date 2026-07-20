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

LangGraph controls routing between the existing retrieval, graph discovery,
stopping, and extraction layers. The baseline templated answer path remains
available for comparison, while the agent can use an OpenAI-compatible LLM
backend for planning and final synthesis. Each run produces ordered execution
events with a shared run ID. The CLI can retain those events as JSONL beside a
compact report and run log, while the FastAPI streaming endpoint emits the same
event schema. The complete internal workflow payload is debug-only. The app
surface stays lightweight; agent comparison is exposed through CLI artifacts
and a read-only Streamlit review console rather than a heavier interactive UI.

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

The Streamlit interface can add review-console ergonomics such as evidence
filtering, sorting, summary metrics, and report exports, but should continue to
consume presentation payloads rather than calling lower-level retrieval,
generation, or agent helpers directly.

## Interface layout

External entrypoints are grouped under `interfaces/`:

- `interfaces/web/`: Streamlit browser demo
- `interfaces/api/`: FastAPI service boundary

Core workflow orchestration lives under `src/bioevidence/workflows/` so the
baseline RAG path, agent workflow, and retrieval stack are not coupled to the
UI, API, or agent-specific helper modules.

The FastAPI service is the deployable backend boundary for portfolio purposes.
Docker packages that API service with the curated local demo corpus and a health
check, while the Streamlit review console remains a local presentation surface
over normalized workflow payloads.

## Evaluation flow
Evaluation should stay local and file-based:

- load JSONL items from disk
- run the existing workflow per item
- compute retrieval and answer metrics
- emit a summary report plus per-item results that can be written as JSON

## Implementation constraints
- keep local development first-class
- use environment variables for secrets
- keep external dependencies moderate
- keep Docker focused on FastAPI service packaging, not as the only workflow
- do not commit to a heavyweight framework too early

## Quality gates
CI should remain lightweight and deterministic:

- lint with Ruff
- run focused mypy checks over stable schema, evaluation, and workflow modules
- run the pytest suite
- run a small evaluation smoke test over tracked demo artifacts

These checks are intended to catch regressions in the code and evidence workflow
without requiring external model-provider credentials.

## Graph-augmented discovery

The knowledge graph is a discovery source, not a substitute for citable
literature. Hetionet entity linking and path traversal can identify related
diseases, compounds, genes, pathways, and biological processes. Those graph
results are converted into follow-up literature queries and fed back through
the existing PubMed retrieval and evidence pipeline.

```text
question
-> baseline literature retrieval
-> optional Hetionet entity linking and path discovery
-> graph-derived query expansion
-> follow-up literature retrieval
-> deduplication and sufficiency check
-> structured evidence extraction
-> citation-grounded synthesis
```

Graph paths remain visible in the workflow trace, including linked entities,
relationships, and generated branch queries. Final answer citations continue
to identify PubMed records only.

Neo4j is accessed behind an optional provider boundary. A disabled, empty, or
unavailable graph must not break the baseline literature workflow. This keeps
local fixtures and CI deterministic while allowing a composed API plus Neo4j
runtime for graph-enabled demos.

## Agent runtime

LangGraph owns workflow routing, node execution, and streaming updates. Saved
JSONL traces and streamed API events use the same ordered event contract. Domain
behavior remains in project modules:

- the planner proposes follow-up searches
- the graph provider returns discovery context and query expansions
- the retrieval stack returns ranked PubMed candidates
- deterministic rules decide evidence sufficiency
- the answerer synthesizes only from accumulated evidence

This split uses a maintained orchestration runtime without hiding biomedical
retrieval decisions inside framework-specific agents or generic tool loops.
