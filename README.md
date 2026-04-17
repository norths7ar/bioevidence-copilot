# BioEvidence Copilot

BioEvidence Copilot is a portfolio project for biomedical literature evidence retrieval, structured evidence, evaluation, and custom agentic orchestration.

It is intentionally built in two stages:

- Stage 1: citation-grounded RAG over PubMed abstracts
- Stage 2: custom agentic orchestration over the same retrieval pipeline

## Goals
- build an inspectable RAG baseline
- extract structured evidence
- generate citation-grounded answers
- evolve toward agentic workflows without losing modularity
- keep the agent layer custom rather than framework-first
- keep the project evaluation-friendly

## Core demo flow
1. user enters a biomedical question
2. system retrieves PubMed evidence
3. system shows a structured evidence table
4. system returns a final answer with citations

## Repository status
Milestone 6 Streamlit browser demo is now in place on top of milestone 5 agent orchestration, milestone 4 evaluation, milestone 3 structured evidence output, milestone 2 RAG baseline, and milestone 1 PubMed ingestion scaffold.

## Planned modules
- ingestion
- retrieval
- generation
- extraction
- agent
- evaluation

## Quickstart
This repository uses a `src/` layout and targets Python 3.12.

Suggested local setup:

1. create and activate a Python 3.12 environment
2. install the project in editable mode with test extras
3. run the app as a module or run the tests

Example commands:

```powershell
python -m pip install -e .[dev]
python -m app.main
streamlit run app/streamlit_app.py
python scripts/ingest_pubmed.py "asthma corticosteroids" --retmax 5
pytest
```

For a portfolio or interview demo, `streamlit run app/streamlit_app.py` is the
best first command because it shows the baseline/agent comparison in tabs.
The CLI entrypoints remain useful for debugging and automated checks.

Editable install is the supported local workflow. Direct execution via
`python app/main.py` is intentionally not the primary path.

To enable the dense retriever, configure the embedding backend via the generic
`.env` fields:

- `BIOEVIDENCE_EMBEDDING_API_KEY`
- `BIOEVIDENCE_EMBEDDING_BASE_URL`
- `BIOEVIDENCE_EMBEDDING_MODEL`
- `BIOEVIDENCE_EMBEDDING_DIMENSIONS`

To enable the agent planner and final synthesis path, configure the generic
agent variables and pick a provider in `.env`:

- `BIOEVIDENCE_AGENT_API_KEY`
- `BIOEVIDENCE_AGENT_BASE_URL`
- `BIOEVIDENCE_AGENT_MODEL`
- `BIOEVIDENCE_AGENT_MAX_ITERATIONS=3`
- `BIOEVIDENCE_AGENT_MAX_OUTPUT_TOKENS=800`
- `BIOEVIDENCE_AGENT_MIN_RELEVANCE_SCORE=0.6`
- `BIOEVIDENCE_AGENT_MIN_UNIQUE_PMIDS=3`
- `BIOEVIDENCE_AGENT_TEMPERATURE=0.2`

Example provider mappings are documented in `.env.example` for Qwen embedding,
DeepSeek, Qwen Chat, and MiMo.

The demo app now shows:
- the query and rewritten query
- the top retrieved papers with scores and ranks
- a structured evidence table with PMID, title, year, journal, entities, summary, and relevance score
- the final answer and citation list
- baseline and agent comparisons in browser tabs via Streamlit

The agent CLI adds:
- multi-step branch planning
- branch-level retrieval traces
- baseline vs agent comparison metadata
- JSON output that can be written to disk with `--output`

Implementation note:
- the baseline RAG answer path is evidence stitching / templated synthesis over structured evidence
- the agent path is the LLM-backed synthesis path
- `rerank.py` is a deterministic post-process, not a learned cross-encoder reranker

## Evaluation
The evaluation harness is file-based and local:

- runtime datasets live under `data/eval/`
- a schema reference dataset is tracked at `examples/milestone4_eval_dataset.jsonl`
- run the harness with `python scripts/run_eval.py --dataset data/eval/dataset.jsonl`
- optionally add `--output path/to/report.json` to write the full report artifact
- run the agent workflow directly with `python scripts/run_agent.py --query "asthma corticosteroids" --output data/eval/agent-report.json`

Each JSONL dataset row uses:

- `id`
- `query`
- `gold_pmids` or `gold_citations`
- optional `reference_answer`
- optional `top_k`

The agent report fixture lives at `examples/milestone5_agent_report.json`.

The ingestion script writes raw artifacts under `data/raw/` and processed
documents under `data/processed/`.

The milestone 2 retrieval baseline reads the local corpus from
`data/processed/*.documents.jsonl` when available and falls back to live
PubMed search only when the local corpus is empty.

Dense embeddings are cached locally under the configured data directory so
repeated queries do not re-embed the same corpus documents.

## Project structure

```text
app/                lightweight application entrypoint
src/bioevidence/    importable package stubs
docs/               project brief, architecture, roadmap, decisions
scripts/            small helper scripts for local workflows and ingestion runs
tests/              placeholder test shape
data/               local-only raw, processed, and eval artifacts
notebooks/          exploration notebook
```

## Notes
This is a portfolio and research-engineering project.
It is not a clinical product and should not be used for medical decision-making.
