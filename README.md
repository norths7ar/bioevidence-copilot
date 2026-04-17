# BioEvidence Copilot

BioEvidence Copilot is a portfolio project for biomedical literature evidence retrieval and synthesis.

It is intentionally built in two stages:

- Stage 1: citation-grounded RAG over PubMed abstracts
- Stage 2: agentic orchestration over the same retrieval pipeline

## Goals
- build an inspectable RAG baseline
- extract structured evidence
- generate citation-grounded answers
- evolve toward agentic workflows without losing modularity
- keep the project evaluation-friendly

## Core demo flow
1. user enters a biomedical question
2. system retrieves PubMed evidence
3. system shows a structured evidence table
4. system returns a final answer with citations

## Repository status
Milestone 2 baseline retrieval is now in place on top of the milestone 1 PubMed ingestion scaffold.

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
python scripts/ingest_pubmed.py "asthma corticosteroids" --retmax 5
pytest
```

Editable install is the supported local workflow. Direct execution via
`python app/main.py` is intentionally not the primary path.

To enable the dense retriever, add the Qwen embedding variables to your local
`.env` file:

- `QWEN_API_KEY`
- `QWEN_BASE_URL` if you are using a non-default endpoint
- `QWEN_EMBEDDING_MODEL=text-embedding-v4`
- `QWEN_EMBEDDING_DIMENSIONS=1024`

The demo app now shows:
- the query and rewritten query
- the top retrieved papers with scores and ranks
- the final answer and citation list

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
