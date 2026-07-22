# BioEvidence Copilot

[![CI](https://github.com/norths7ar/bioevidence-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/norths7ar/bioevidence-copilot/actions/workflows/ci.yml)

BioEvidence Copilot is a portfolio project for biomedical literature evidence retrieval, structured evidence, evaluation, LangGraph orchestration, and optional Hetionet discovery.

It is intentionally built in two stages:

- Stage 1: citation-grounded RAG over PubMed abstracts
- Stage 2: LangGraph orchestration over the same retrieval pipeline
- Stage 3: Hetionet knowledge-graph discovery that expands literature searches

## Goals
- build an inspectable RAG baseline
- extract structured evidence
- generate citation-grounded answers
- evolve toward agentic workflows without losing modularity
- use a maintained orchestration runtime while keeping domain logic explicit
- keep the project evaluation-friendly

## Core demo flow
1. user enters a biomedical question
2. system retrieves PubMed evidence
3. system shows a structured evidence table
4. optional graph paths propose related biomedical search terms
5. system returns a final answer with PubMed citations

## Repository status
The project now includes the Stage 1 RAG baseline, Stage 2 LangGraph agentic
orchestration, reproducible demo/evaluation artifacts, evidence faithfulness
checks, agent traceability, a polished Streamlit review console, a FastAPI
service boundary, optional Hetionet/Neo4j query expansion, local Docker Compose,
and GitHub Actions quality gates. The literature-only baseline is preserved at
the `v0.1.0` release tag. The `v0.3.0` release adds a versioned semantic evidence
schema, PMID-safe fine-tuning data, a published local QLoRA adapter, and an
optional fine-tuned extraction backend.

The recommended extraction adapter is published on Hugging Face as
[`n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2`](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2).
The
[`v1 adapter`](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v1)
remains available as the comparison checkpoint.

## Implemented modules
- ingestion
- retrieval
- generation
- extraction
- agent
- graph discovery
- evaluation
- API
- web review console

## Quickstart
This repository uses a `src/` layout and targets Python 3.12.

Suggested local setup:

1. create and activate a Python 3.12 environment
2. install the project in editable mode with test extras
3. run the baseline CLI, Streamlit UI, API, or tests

Example commands:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m pip install -e ".[dev,serve,graph,web]"
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_baseline.py
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m streamlit run interfaces/web/streamlit_app.py
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/ingest_pubmed.py "asthma corticosteroids" --retmax 5
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m pytest
```

For a portfolio or interview demo, `streamlit run interfaces/web/streamlit_app.py` is the
best first command because it shows the baseline/agent comparison in tabs.
The CLI entrypoints remain useful for debugging and automated checks.

Editable install is the supported local workflow. CLI entrypoints live under
`scripts/`, and external interfaces live under `interfaces/`.

To enable the dense retriever, configure the embedding backend via the generic
`.env` fields:

- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`

To enable the agent planner and final synthesis path, configure the generic
agent variables and pick a provider in `.env`:

- `AGENT_API_KEY`
- `AGENT_BASE_URL`
- `AGENT_MODEL`
- `AGENT_MAX_ITERATIONS=3`
- `AGENT_MAX_OUTPUT_TOKENS=8192`
- `AGENT_MIN_RELEVANCE_SCORE=0.6`
- `AGENT_MIN_UNIQUE_PMIDS=3`
- `AGENT_TEMPERATURE=0.2`

Example provider mappings are documented in `.env.example` for Qwen embedding,
DeepSeek, Qwen Chat, and MiMo.

Semantic extraction defaults to the compatibility-preserving `legacy` mode.
Set `EXTRACTION_BACKEND` to `rules`, `prompted`, or `local` to attach validated
query-focused fields to evidence rows. Local QLoRA inference also requires
`EXTRACTION_ADAPTER_PATH` and the separate `bioevidence-training` environment;
the normal API installation remains GPU-toolchain-free.

Run a short end-to-end local-adapter demo from the training environment:

```powershell
conda activate bioevidence-training
$env:HF_HOME="E:/huggingface-cache"
$env:HF_HUB_CACHE="$env:HF_HOME/hub"
$env:HF_XET_CACHE="$env:HF_HOME/xet"
python scripts/setup_extraction_adapter.py
$env:EXTRACTION_BACKEND="local"
$env:EXTRACTION_ADAPTER_PATH="artifacts/models/bioevidence-qwen3-4b-extraction-lora-v2"
python scripts/run_baseline.py `
  --query "asthma corticosteroids exacerbations randomized trial" `
  --top-k 3 `
  --output artifacts/runs/extraction_demo/local.json
```

Replace `HF_HOME` with the directory containing the existing Hugging Face model
cache. Explicit cache variables also prevent Unsloth's Windows startup probe
from falling back to a slow or unwritable default cache. Omit them when the
activated environment already defines the same locations. The first local run
loads the base model once; the backend then reuses it across all evidence rows
in that process.

The second local adapter experiment extends the draft dataset from 60 to 120
rows while preserving every v1 PMID split. On the resulting 13-row shared test
set, adapter v2 reached 100% JSON/schema validity, 92.3% grounded spans, 61.5%
evidence-status accuracy, and 0.681 semantic-field token F1. The prompted base
model had higher status accuracy (76.9%) but only 92.3% schema validity, 61.5%
grounding, 0.512 semantic F1, and slower mean generation (19.58 s versus
11.23 s). These remain small draft-label results, not biomedical quality claims.
The complete comparison and failure analysis are summarized in
[`docs/EXTRACTION_MODEL_REPORT.md`](docs/EXTRACTION_MODEL_REPORT.md).

Evidence rows produced by an optional extractor include
`extraction_attempted_backend`, `extraction_backend`, and
`extraction_fallback_reason`, so a schema or grounding fallback remains visible
in normal JSON exports. To inspect one failed model response without placing raw
model output in the API payload, run the opt-in diagnostic command; its default
destination is ignored by Git:

```powershell
python scripts/diagnose_extraction.py `
  --query "asthma corticosteroids exacerbations randomized trial" `
  --pmid 41772161 `
  --output artifacts/evaluations/extraction_diagnostic_41772161.json
```

Set `LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` to control
application logs. Logs go to the process stream by default, including under
Docker. Agent run bundles retain `run.log`, Streamlit uses a rotating local log,
and Docker rotates API logs. Provider credentials, prompts, abstracts, and full
PMID lists are not logged.

The demo app now shows:
- the query and rewritten query
- the top retrieved papers with scores and ranks
- a structured evidence table with PMID, title, year, journal, entities, summary, and relevance score
- the final answer and citation list
- baseline and agent comparisons in browser tabs via Streamlit
- evidence filtering, sorting, readable trace tables, and JSON / Markdown / CSV exports

The LangGraph agent adds:
- multi-step branch planning
- branch-level retrieval traces
- baseline vs agent comparison metadata
- compact JSON reports that can be written with `--output`
- per-run log, report, and JSONL trace bundles under `--artifacts-dir`
- optional Hetionet entity/path discovery before follow-up literature searches
- streaming execution events through the FastAPI NDJSON endpoint

Implementation note:
- the baseline RAG answer path is evidence stitching / templated synthesis over structured evidence
- the agent path is the LLM-backed synthesis path
- `ranking.py` is a deterministic final ordering step, not a learned cross-encoder reranker
- graph paths guide discovery but never replace PMID-backed evidence
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
- run the harness with `C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl`
- add `--data-dir data/corpora/demo` to evaluate against the seeded demo corpus
- add `--mode agent` to evaluate the agent workflow instead of the baseline
- add `--limit N` for BioASQ smoke runs before attempting the full dataset
- optionally add `--output path/to/report.json` to write a compact report artifact
- run the agent workflow with `C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_agent.py --query "asthma corticosteroids" --data-dir data/corpora/demo --artifacts-dir artifacts/runs`
- with Neo4j configured, compare baseline and graph-expanded PMID retrieval with `C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_graph_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --limit 5`

Agent CLI output is intentionally split by responsibility. The console prints a
short run summary. With `--artifacts-dir`, each run creates a timestamped,
Git-ignored directory containing `run.log`, `report.json`, and `trace.jsonl`.
Add `--debug` to include `debug.json`, which preserves the full internal payload
only when it is needed for troubleshooting. Evidence rows appear once in the
compact report; baseline and agent sections reference them by PMID.

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

## Quality gates
GitHub Actions runs the project quality gate on push and pull request:

- `ruff` lint checks
- focused `mypy` type checks over stable schema, evaluation, and workflow modules
- the full pytest suite
- a one-item baseline evaluation smoke test over the tracked demo dataset

Run the same checks locally:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m ruff check --no-cache .
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m mypy src/bioevidence/schemas src/bioevidence/evaluation src/bioevidence/workflows src/bioevidence/graph --no-sqlite-cache --no-incremental
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m pytest
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --mode baseline --limit 1
```

## API
The FastAPI service exposes the core workflow without moving business logic out
of `src/bioevidence/`.

Install the service runtime extra when needed:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m pip install -e ".[dev,serve,graph,web]"
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
- `POST /api/v1/query/agent/stream` (newline-delimited execution events; startup failures use HTTP 400/500,
  while failures after streaming starts produce a terminal `error` event)

Run the graph-enabled local service composition:

```powershell
docker compose up --build -d
$env:NEO4J_PASSWORD="bioevidence-local"
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/import_hetionet.py --hetionet-root E:/GitHub-Repos/hetionet-main
```

Compose starts FastAPI on port `8000`, Neo4j Browser on `7474`, and Bolt on
`7687`. The database initially contains no Hetionet data; import is an explicit
one-time preparation step. Override `NEO4J_PASSWORD` in `.env` or
the shell instead of using the documented local default outside development.
The API service reads provider credentials and runtime settings from `.env`,
then uses container-specific data, cache, and Neo4j addresses internally.

## Project structure

```text
interfaces/         external API and Streamlit UI entrypoints
src/bioevidence/    importable package with literature, graph, evidence, agent, and workflow layers
docs/               project brief, architecture, roadmap, decisions, evaluation, demo, and limitation notes
scripts/            local CLI helpers for baseline, agent, ingestion, conversion, and evaluation
tests/              unit and integration tests
data/               curated corpora/evaluation artifacts plus ignored runtime cache
```

## Documentation
- `docs/DEMO_SCRIPT.md`: portfolio and interview walkthrough commands
- `docs/EVALUATION.md`: dataset format, metrics, and evaluation commands
- `docs/LIMITATIONS.md`: medical, data, model, agent, and deployment boundaries
- `docs/ARCHITECTURE.md`: system organization and interface boundaries
- `docs/DECISIONS.md`: dated architecture decisions
- `docs/ROADMAP.md`: completed milestones and optional future work

## Notes
This is a portfolio and research-engineering project.
It is not a clinical product and should not be used for medical decision-making.
