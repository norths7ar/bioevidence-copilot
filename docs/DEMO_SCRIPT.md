# Demo Script

This script is the short portfolio walkthrough path for BioEvidence Copilot.
It assumes the repository has already been installed locally and the curated
demo corpus under `data/corpora/demo` is available.

## Positioning

BioEvidence Copilot is a biomedical literature evidence assistant over PubMed
metadata and abstracts. The project demonstrates an inspectable RAG baseline,
structured evidence extraction, local evaluation, LangGraph orchestration,
a Hetionet discovery branch,
a lightweight review console, a FastAPI service boundary, Docker packaging, and
CI quality gates.

It is not a clinical decision system.

## One-minute Walkthrough

1. Start with the README badge and explain that CI runs lint, focused type
   checks, tests, and a small evaluation smoke test.
2. Show the Streamlit review console.
3. Run a query such as `What evidence exists for asthma corticosteroids?`.
4. Compare the Baseline and Agent tabs.
5. Point out that graph-linked entities create follow-up literature queries,
   while final citations still identify PubMed papers.
6. Show the evidence table, trace summary, branch diagnostics, and exports.
7. Show the FastAPI health endpoint or streaming agent endpoint.
8. Mention Docker Compose as the local FastAPI plus Neo4j service shape.
9. Close with evaluation and limitations: local demo data, abstract-only
   evidence, deterministic checks, and no medical-decision use.

## Local Demo Commands

Install or refresh local development dependencies:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m pip install -e ".[dev,serve,graph,web]"
```

Run the review console:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m streamlit run interfaces/web/streamlit_app.py
```

Run the FastAPI service locally:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe -m uvicorn interfaces.api.main:app --reload
```

Check the API:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/api/v1/health
```

Run the agent workflow with retained run artifacts:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_agent.py --query "asthma corticosteroids" --data-dir data/corpora/demo --artifacts-dir artifacts/runs
```

The command prints a short summary and creates one timestamped directory with
`run.log`, `report.json`, and `trace.jsonl`. Add `--debug` only when the complete
internal workflow payload is needed.

Run the evaluation smoke path:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --mode baseline --limit 1
```

Build and run the API container:

```powershell
docker build -t bioevidence-copilot-api .
docker run --rm --name bioevidence-api -p 8000:8000 bioevidence-copilot-api
```

Run the graph-enabled service pair:

```powershell
docker compose up --build -d
```

## What to Highlight

- The baseline path is inspectable: retrieved papers, evidence rows, answer,
  and citations are all visible.
- The LangGraph path is controlled: graph discovery, branch planning,
  deduplication, deterministic stopping, and coverage comparison are exposed as
  trace data.
- Graph-derived terms are evaluated by whether they recover additional relevant
  PMIDs, not by graph-path count alone.
- The evidence table is a review surface, not hidden prompt context.
- The API is a thin boundary over core package workflows.
- Docker packages the FastAPI service without replacing the local conda
  development flow.
- CI verifies code quality and a small evidence workflow smoke path on every
  push or pull request.

## Expected Tradeoffs

- Dense embeddings and LLM synthesis use provider-agnostic OpenAI-compatible
  settings, but the project can fall back when credentials are not configured.
- The demo corpus is intentionally small and curated for repeatability.
- Abstract-only evidence is useful for portfolio demonstration but incomplete
  for clinical or systematic-review conclusions.
