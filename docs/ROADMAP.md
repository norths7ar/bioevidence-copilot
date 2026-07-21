# ROADMAP

## Milestone 0: Scaffold [done]
- create repository layout
- create importable package
- add pyproject.toml
- add placeholder tests
- add basic app entrypoint
- add docs

## Milestone 1: PubMed ingestion [done]
- implement PubMed search client
- normalize metadata into internal schema
- save sample raw and processed artifacts

## Milestone 2: RAG baseline [done]
- implement lexical retrieval
- implement dense retrieval interface
- implement hybrid merge
- implement deterministic final ranking
- implement answer generation with citations
- show results in app

## Milestone 3: Structured evidence [done]
- implement evidence extraction
- render evidence table in app
- store real demo/evaluation artifacts

## Milestone 4: Evaluation [done]
- define small eval dataset format
- add retrieval and answer checks
- implement evaluation runner

## Milestone 5: Agentic orchestration [done]
- add a custom LLM-backed planner and synthesis path
- add multi-query branch execution over the existing retrieval stack
- add deduplication, deterministic stopping, and baseline comparison
- expose agent reports through CLI / JSON artifacts instead of a heavier UI

## Milestone 6: Streamlit demo surface [done]
- add a thin browser UI for baseline vs agent comparison
- reuse the existing presentation helpers and workflow outputs
- keep the browser view presentation-only and read-only

## Next phase: Portfolio productization

The next phase should make the project stronger in two complementary ways:

1. deepen the biomedical evidence-assistant value of the project
2. broaden the visible engineering stack for portfolio and job-search purposes

The product-depth work should stay close to the repository's original identity:
a citation-grounded biomedical literature evidence assistant over PubMed
metadata and abstracts. The engineering-stack work should expose the same
core pipeline through conventional backend service boundaries without turning
the project into a generic chatbot or framework-first demo.

## Track A: Product depth

### Milestone 7: Reproducible demo and evaluation suite [done]
- define a fixed interview/demo query set
- build or document a fixed local PubMed abstract corpus for repeatable demos
- expand the evaluation dataset beyond the minimal example fixture
- generate a saved evaluation report with aggregate and per-query results
- compare baseline, hybrid retrieval, and agent workflow outputs
- document expected demo commands and sample outputs

### Milestone 8: Evidence quality and faithfulness [done]
- add citation faithfulness checks for generated answers
- detect answers that cite unsupported or weakly supported claims
- add an explicit insufficient-evidence outcome where appropriate
- enrich evidence records with fields such as population, intervention,
  comparator, outcome, study type, and effect direction when available
- surface conflicting or mixed evidence instead of forcing a single conclusion
- keep these checks inspectable and testable with local fixtures

### Milestone 9: Search strategy and agent traceability [done]
- make query rewriting and branch planning easier to inspect
- show original query, rewritten query, branch queries, and retrieval rationale
- add branch-level retrieval diagnostics and stopping reasons
- report whether the agent improved retrieval coverage over the baseline
- keep agent stopping deterministic and tied to evidence sufficiency

### Milestone 10: Demo console polish [done]
- improve the Streamlit demo as a lightweight review console
- prioritize evidence table filtering, sorting, and readable trace views
- add export paths for Markdown, JSON, or CSV reports
- keep Streamlit as a presentation layer over existing workflow outputs
- do not spend early effort converting Streamlit into a full API client unless
  there is a concrete demo or architecture need

## Track B: Engineering stack expansion

### Milestone 11: FastAPI service layer [done]
- add a thin FastAPI API layer around the existing core package
- keep retrieval, generation, extraction, evaluation, and agent logic in
  `src/bioevidence/`
- define typed request and response schemas for baseline and agent queries
- add a health endpoint for local service checks
- return retrieved evidence, structured evidence rows, citations, final answer,
  and trace metadata in API responses
- add API tests using FastAPI's test client

Current endpoint shape:

```text
GET  /api/v1/health
POST /api/v1/query/baseline
POST /api/v1/query/agent
```

Deferred endpoint candidates:

```text
GET  /api/v1/corpora
POST /api/v1/evaluations/run
```

### Milestone 12: Docker packaging [done]
- add a Dockerfile for the FastAPI service
- add a `.dockerignore`
- document local Python startup and Docker startup separately
- make configuration flow through environment variables
- include a health check path
- keep Docker focused on portfolio/backend-stack breadth, not as the primary
  local development workflow

### Milestone 13: Optional service composition
- optionally add Docker Compose for API plus Streamlit if it becomes useful
- avoid making Compose a blocker for the core product-depth milestones
- keep Streamlit API-client conversion optional and lower priority
- document that API and UI are architecturally decoupled even if Streamlit
  continues to call local package functions during development

## Supporting work

### Quality gates [done]
- add CI for tests
- add linting checks
- add focused type-checking for stable core modules
- add an evaluation smoke test that can run quickly in CI

### Documentation [done]
- keep `docs/DECISIONS.md` updated for meaningful architecture decisions
- maintain `docs/EVALUATION.md` for evaluation datasets, metrics, and commands
- add `docs/DEMO_SCRIPT.md` for interview or portfolio walkthroughs
- add `docs/LIMITATIONS.md` to document medical, data, and model limitations
- keep README focused on the shortest practical demo path first

### Engineering hygiene [done]
- split `agent/workflow.py` into `workflows/` package: `models.py`,
  `baseline.py`, `agent.py`, `retrieval_stack.py`
- move `retrieval/rerank.py` into `retrieval/ranking.py` with a cleaner name
- consolidate `api/` and `app/` into `interfaces/api/` and `interfaces/web/`
- remove placeholder files: `examples/milestone*.json`, `notebooks/exploration.ipynb`,
  `scripts/bootstrap.py`, `scripts/build_index.py`, `scripts/demo_query.py`
- align `data/` layout to a source-first naming convention:
  `data/corpora/` for corpora, `data/evaluations/` for eval sets

## Priority guidance

Completed productization priorities:
- Milestone 7: reproducible demo and evaluation suite
- Milestone 8: evidence quality and faithfulness
- Milestone 9: search strategy and agent traceability
- Milestone 10: demo console polish
- Milestone 11: FastAPI service layer
- Milestone 12: Docker packaging
- quality gates and documentation polish

Lower priority unless needed later:
- Docker Compose for API plus Streamlit presentation
- Streamlit-as-API-client conversion
- full-text ingestion, PDF parsing, browser automation, or database adoption

## v0.2: Graph-augmented evidence discovery

The v0.1.0 release freezes the literature-only product baseline. The next
version integrates the useful domain capabilities from `biomedical-graphrag`
without treating knowledge-graph paths as citable literature evidence.

### Milestone 14: Hetionet discovery layer [done]
- migrate typed Hetionet nodes, entity linking, and deterministic path retrieval
- expose Neo4j through an optional graph provider boundary
- use graph paths to propose biomedical terms and follow-up literature queries
- keep PubMed papers and PMIDs as the evidence source for final answers
- keep baseline workflows operational when Neo4j is disabled or unavailable

### Milestone 15: LangGraph orchestration runtime [done]
- replace the handwritten agent loop with an explicit state graph
- keep retrieval, graph discovery, sufficiency checks, extraction, and synthesis
  as independently testable domain functions
- expose workflow updates as a streaming API surface
- preserve deterministic stopping and the existing inspectable trace contract

### Milestone 16: Graph augmentation evaluation [done]
- compare baseline literature retrieval with graph-expanded retrieval
- report recall@k, MRR, new relevant PMIDs, and per-query deltas
- keep graph evaluation ground truth independent of graph traversal output
- do not reuse Hetionet path-template pseudo ground truth as biomedical evidence

### Milestone 17: Local service composition [done]
- add Docker Compose for FastAPI plus Neo4j
- keep Hetionet import as an explicit, repeatable data-preparation step
- keep React, hosted deployment, and general application persistence out of
  scope until the retrieval and evaluation work is validated

## v0.3: Fine-tuned evidence extraction

The next version deepens the biomedical evidence layer. Fine-tuning is an
implementation technique within the existing product, not a separate chatbot
or model-training demo.

### Milestone 18: Extraction contract and annotation pilot [completed]
- define a versioned, query-focused JSON Schema for model-generated evidence
- keep document metadata and retrieval scores outside the model target
- add annotation rules, exact abstract-span validation, and review status
- pressure-test the contract on tracked direct, indirect, and negative pairs

### Milestone 19: Comparable extraction baselines [in progress]
- evaluate the deterministic extractor and a prompted base model against the
  same versioned pilot labels
- report JSON parse rate, schema validity, field metrics, outcome direction,
  evidence-span support, latency, and cost
- distinguish workflow-integrity checks from model-quality benchmarks

The shared backend contract, prompt-only adapter, rule adapter, and offline
metrics runner are implemented. Milestone 19 remains in progress until a
prompted model run is recorded against the versioned pilot labels.

### Milestone 20: Training dataset
- expand reviewed annotations only after the pilot contract stabilizes
- split by PMID to prevent document leakage across train, dev, and test sets
- preserve source, license, transformation, and annotation provenance
- do not treat unreviewed synthetic labels as benchmark ground truth

### Milestone 21: Fine-tuning and offline evaluation
- keep training code under `training/evidence_extraction/` in this repository
- use a separate environment and optional dependencies for the training stack
- compare the fine-tuned model against both established baselines
- publish weights externally with a model card rather than committing them to Git

### Milestone 22: Optional product inference backend
- add deterministic, prompted, and fine-tuned extraction backends behind one
  validated output contract
- preserve deterministic fallback behavior when model inference is unavailable
- surface grounded structured fields without making the core API image depend
  on the training toolchain
