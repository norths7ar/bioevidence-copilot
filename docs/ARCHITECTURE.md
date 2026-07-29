# Architecture

## System shape

BioEvidence Copilot separates six concerns:

1. ingestion
2. retrieval
3. evidence extraction
4. answer generation
5. agent orchestration
6. evaluation

The baseline and agent paths share the same retrieval and extraction modules.
External interfaces call workflow functions rather than reimplementing domain
logic.

## Request flows

### Baseline RAG

```text
query
-> normalize retrieval query
-> load a local corpus or search PubMed
-> lexical retrieval with optional dense scores
-> deterministic final ranking
-> extract one evidence record per selected paper
-> build a templated answer and PMID citations
```

The baseline is intentionally inspectable: ranked papers, evidence rows, and
citations remain available alongside the answer.

### Agent workflow

```text
query
-> run the baseline
-> optional Hetionet entity/path discovery
-> plan one or more follow-up literature queries
-> retrieve and merge branch results
-> check deterministic stopping criteria
-> repeat or synthesize a final answer
```

LangGraph controls node routing and streaming. Project modules still own query
planning, graph access, retrieval, evidence extraction, stopping, and
synthesis. This keeps the workflow testable without encoding biomedical
behavior in generic framework nodes.

## Evidence boundary

`Document` owns PubMed metadata and abstract text. `RetrievedCandidate` adds
retrieval scores and rank. `EvidenceRecord` is the product-facing evidence row
used by answers and citations.

Optional semantic extraction reads:

```text
current query + one paper title + one abstract
```

and returns a validated `ModelEvidenceExtraction` containing query-specific
evidence status, study design, semantic fields, outcomes, and verbatim evidence
spans. The result is attached to the existing `EvidenceRecord`; it does not
regenerate PMID, title, year, journal, or retrieval score.

The extraction interface has four product modes:

- `legacy`: original compatibility path
- `rules`: deterministic output for the model schema
- `prompted`: an OpenAI-compatible model constrained by the same schema
- `local`: the published QLoRA adapter

One mode is selected for a workflow run. Baseline and agent retrieval branches
use that same selection, which makes their evidence rows comparable.

## Graph boundary

Hetionet is a discovery source, not an evidence source. Entity linking and graph
paths may produce related terms for follow-up PubMed searches, but final answers
cite retrieved PMID records only.

Neo4j sits behind an optional provider interface. Disabling the graph or losing
the connection must not break the literature-only baseline.

## Generation boundary

The baseline answerer stitches structured evidence into a deterministic
response. The agent answerer can use an OpenAI-compatible LLM to synthesize from
the accumulated evidence.

Neither answer path owns retrieval. Citation checks compare returned PMIDs with
the evidence records available to that run.

## Interfaces and observability

Core orchestration lives in `src/bioevidence/workflows/`. External entrypoints
are:

- `scripts/`: CLI workflows and evaluation utilities
- `interfaces/api/`: FastAPI service and streamed agent events
- `interfaces/web/`: Streamlit review console

Workflow results are normalized into presentation payloads before they reach
the API or UI. Agent runs also emit ordered events with a shared run ID so CLI
artifacts and the streaming API expose the same execution history.

## Evaluation boundary

Evaluation remains file-based:

```text
versioned JSONL items
-> run an existing workflow or extraction backend
-> compute per-item metrics and checks
-> aggregate a machine-readable report
```

Retrieval, citation, graph-gain, and extraction experiments use explicit
datasets and preserve intermediate predictions. CI runs a small deterministic
smoke test; model-quality comparisons are separate offline experiments.

## Dependency boundaries

- provider credentials and runtime choices stay in environment configuration;
- the default product environment does not install the local QLoRA training stack;
- Docker packages the FastAPI service, while Streamlit remains a local review interface;
- large model weights and external datasets stay outside Git.
