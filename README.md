# BioEvidence Copilot

[![CI](https://github.com/norths7ar/bioevidence-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/norths7ar/bioevidence-copilot/actions/workflows/ci.yml)

BioEvidence Copilot retrieves biomedical literature, turns PubMed abstracts
into inspectable evidence records, and produces citation-grounded answers.

The system has two comparable paths:

- a deterministic RAG baseline over PubMed abstracts;
- a LangGraph workflow that can expand the search, merge retrieval branches,
  stop on explicit criteria, and synthesize from the accumulated evidence.

Optional Hetionet discovery supplies related biomedical terms to the agent
search. Graph paths guide retrieval but never replace PubMed citations.

## How it works

```text
question
  -> PubMed or local-corpus retrieval
  -> lexical/dense ranking
  -> query-focused evidence extraction
  -> evidence table
  -> citation-grounded answer

agent path only:
  -> optional Hetionet discovery
  -> planned follow-up queries
  -> branch retrieval and deduplication
  -> deterministic sufficiency check
  -> final synthesis
```

The baseline uses templated evidence stitching. The agent can use an
OpenAI-compatible LLM for planning and synthesis while keeping retrieval,
stopping, extraction, and citation checks in project-owned modules.

## Current capabilities

- PubMed E-utilities ingestion and tracked local corpora
- lexical retrieval with optional dense embeddings
- baseline and agent workflows over the same retrieval stack
- structured evidence tables with PMID-backed citations
- optional Hetionet/Neo4j query expansion
- deterministic, prompted, and local QLoRA extraction modes
- file-based retrieval, citation, faithfulness, graph-gain, and extraction evaluation
- CLI, FastAPI, and Streamlit interfaces
- Ruff, mypy, pytest, and evaluation smoke checks in CI

## Fine-tuned evidence extraction

The extraction experiment takes a biomedical query, one PubMed title, and its
abstract, then predicts a versioned `ModelEvidenceExtraction` JSON object. The
target includes evidence status, study design, population, intervention or
exposure, comparator, outcomes, and verbatim supporting spans.

On the 13-row PMID-held-out draft test set, the recommended QLoRA v2 adapter
reached 100% schema validity, 92.3% grounded outputs, and 0.681 semantic-field
token F1. The prompted base model retained higher evidence-status accuracy
(76.9% versus 61.5%), while QLoRA v2 was faster on average (11.23 seconds versus
19.58 seconds). The small model-assisted test set supports an engineering
comparison, not a broad biomedical-quality claim.

See [the extraction model report](docs/EXTRACTION_MODEL_REPORT.md) for the full
rules/prompted/v1/v2 comparison and failure analysis.

Published adapters:

- [QLoRA v2](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2)
- [QLoRA v1](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v1)

## Quickstart

The project targets Python 3.12 and uses the committed `uv.lock` for
reproducible local and CI environments.

```powershell
uv sync --locked --all-extras --no-managed-python
uv run python scripts/run_baseline.py
uv run streamlit run interfaces/web/streamlit_app.py
uv run pytest
```

The baseline works with the tracked local demo corpus. Optional providers and
runtime settings are documented in `.env.example`:

- `EMBEDDING_*` enables dense retrieval;
- `AGENT_*` enables LLM planning and synthesis;
- `GRAPH_*` and `NEO4J_*` enable Hetionet discovery;
- `EXTRACTION_*` selects semantic extraction.

Semantic extraction defaults to `legacy`. Set `EXTRACTION_BACKEND` to `rules`,
`prompted`, or `local` to attach the versioned query-focused fields. Local QLoRA
inference uses the separate `bioevidence-training` environment and an adapter
path; the normal API installation does not require the GPU training stack.

```powershell
conda activate bioevidence-training
python scripts/setup_extraction_adapter.py
$env:EXTRACTION_BACKEND="local"
$env:EXTRACTION_ADAPTER_PATH="artifacts/models/bioevidence-qwen3-4b-extraction-lora-v2"
python scripts/run_baseline.py `
  --query "asthma corticosteroids exacerbations randomized trial" `
  --top-k 3
```

Detailed training and adapter commands live in
[`training/evidence_extraction/README.md`](training/evidence_extraction/README.md).

## Evaluation

Run the tracked demo evaluation against the local corpus:

```powershell
uv run python scripts/run_eval.py `
  --dataset data/evaluations/demo/demo_eval_dataset.jsonl `
  --data-dir data/corpora/demo `
  --mode baseline
```

The same harness supports agent mode, BioASQ conversion, graph-gain evaluation,
and extraction-backend comparison. See [Evaluation](docs/EVALUATION.md) for
dataset formats, commands, and metric definitions.

## API and local services

Run the FastAPI boundary locally:

```powershell
uv run uvicorn interfaces.api.main:app --reload
```

The API exposes health, baseline query, agent query, and streamed agent-event
endpoints under `/api/v1/`.

Build the API image:

```powershell
docker build -t bioevidence-copilot-api .
docker run --rm --name bioevidence-api -p 8000:8000 bioevidence-copilot-api
```

For graph-enabled local composition, start FastAPI and Neo4j together, then
import Hetionet explicitly:

```powershell
docker compose up --build -d
python scripts/import_hetionet.py --hetionet-root "<path-to-hetionet>"
```

## Quality checks

```powershell
uv run ruff check --no-cache .
uv run mypy src/bioevidence/schemas src/bioevidence/evaluation src/bioevidence/workflows src/bioevidence/graph --no-sqlite-cache --no-incremental
uv run pytest
uv run python scripts/run_eval.py `
  --dataset data/evaluations/demo/demo_eval_dataset.jsonl `
  --data-dir data/corpora/demo `
  --mode baseline `
  --limit 1
```

## Repository layout

```text
interfaces/         FastAPI and Streamlit entrypoints
src/bioevidence/    retrieval, evidence, graph, agent, and workflow modules
data/               tracked corpora and evaluation artifacts
training/           evidence-extraction data, model cards, and QLoRA scripts
scripts/            ingestion, workflow, evaluation, and release utilities
tests/              unit and integration tests
docs/               current architecture, evaluation, decisions, and limitations
```

## Documentation

Suggested reading order:

1. [Architecture](docs/ARCHITECTURE.md) — request flows and module boundaries
2. [Extraction model report](docs/EXTRACTION_MODEL_REPORT.md) — QLoRA experiment and results
3. [Evaluation](docs/EVALUATION.md) — datasets, commands, and metrics
4. [Limitations](docs/LIMITATIONS.md) — evidence and engineering boundaries
5. [Decisions](docs/DECISIONS.md) — the small set of choices that shape the current system

BioEvidence Copilot is a research-engineering system over literature metadata
and abstracts. It is not a clinical decision system.
