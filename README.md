# BioEvidence Copilot

[![CI](https://github.com/norths7ar/bioevidence-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/norths7ar/bioevidence-copilot/actions/workflows/ci.yml)

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
3. run the baseline CLI, Streamlit UI, API, or tests

Example commands:

```powershell
python -m pip install -e .[dev]
python scripts/run_baseline.py
streamlit run interfaces/web/streamlit_app.py
python scripts/ingest_pubmed.py "asthma corticosteroids" --retmax 5
pytest
```

For a portfolio or interview demo, `streamlit run interfaces/web/streamlit_app.py` is the
best first command because it shows the baseline/agent comparison in tabs.
The CLI entrypoints remain useful for debugging and automated checks.

Editable install is the supported local workflow. CLI entrypoints live under
`scripts/`, and external interfaces live under `interfaces/`.

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
- `ranking.py` is a deterministic final ordering step, not a learned cross-encoder reranker
- the FastAPI service is a thin backend boundary over the same workflow functions;
  Streamlit remains a lightweight local presentation surface and does not need
  to call the API for the current local demo

## Evaluation
The evaluation harness is file-based and local:

- runtime evaluation datasets live under `data/evaluations/`
- the demo evaluation dataset is tracked at `data/evaluations/demo/demo_eval_dataset.jsonl`
- an example demo report is tracked at `data/evaluations/demo/demo_eval_report.json`
- seed a real PubMed demo corpus with `scripts/seed_demo_corpus.py`
- convert BioASQ Task B data with `scripts/convert_bioasq.py`
- run the harness with `python scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl`
- add `--data-dir data/corpora/demo` to evaluate against the seeded demo corpus
- add `--mode agent` to evaluate the agent workflow instead of the baseline
- add `--limit N` for BioASQ smoke runs before attempting the full dataset
- optionally add `--output path/to/report.json` to write the full report artifact
- run the agent workflow directly with `python scripts/run_agent.py --query "asthma corticosteroids" --data-dir data/corpora/demo --output data/evaluations/demo/agent-report.json`

Each JSONL dataset row uses:

- `id`
- `query`
- `gold_pmids` or `gold_citations`
- optional `reference_answer`
- optional `top_k`

The ingestion script writes raw artifacts and processed documents under the
configured corpus directory, such as `data/corpora/demo/raw/` and
`data/corpora/demo/processed/`.

The milestone 2 retrieval baseline reads the local corpus from
`processed/*.documents.jsonl` under the configured corpus directory and falls
back to live PubMed search only when the local corpus is empty.

Dense embeddings are cached locally under the configured data directory so
repeated queries do not re-embed the same corpus documents.

See `docs/EVALUATION.md` for demo evaluation commands, real PubMed/BioASQ
data preparation, metric definitions, and deterministic evidence quality checks.

## API
The FastAPI service exposes the core workflow without moving business logic out
of `src/bioevidence/`.

Install the service runtime extra when needed:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m pip install -e .[dev,serve]
```

Run the API locally:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m uvicorn interfaces.api.main:app --reload
```

Build and run the FastAPI service with Docker:

```powershell
docker build -t bioevidence-copilot-api .
docker run --rm --name bioevidence-api -p 8000:8000 --env-file .env bioevidence-copilot-api
```

If you want to run the container without local secrets or model-provider
configuration, omit `--env-file .env`. The health endpoint and local-corpus
fallback paths still work without external LLM or embedding credentials:

```powershell
docker run --rm --name bioevidence-api -p 8000:8000 bioevidence-copilot-api
Invoke-WebRequest -UseBasicParsing http://localhost:8000/api/v1/health
```

Initial endpoints:

- `GET /api/v1/health`
- `POST /api/v1/query/baseline`
- `POST /api/v1/query/agent`

## Project structure

```text
interfaces/         external API and Streamlit UI entrypoints
src/bioevidence/    importable package with retrieval, extraction, generation, agent, and workflow layers
docs/               project brief, architecture, roadmap, decisions, and evaluation notes
scripts/            local CLI helpers for baseline, agent, ingestion, conversion, and evaluation
tests/              unit and integration tests
data/               curated corpora/evaluation artifacts plus ignored runtime cache
```

## Notes
This is a portfolio and research-engineering project.
It is not a clinical product and should not be used for medical decision-making.
