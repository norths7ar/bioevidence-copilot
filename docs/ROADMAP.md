# ROADMAP

## Milestone 0: Scaffold ✅
- create repository layout
- create importable package
- add pyproject.toml
- add placeholder tests
- add basic app entrypoint
- add docs

## Milestone 1: PubMed ingestion ✅
- implement PubMed search client
- normalize metadata into internal schema
- save sample raw and processed artifacts

## Milestone 2: RAG baseline ✅
- implement lexical retrieval
- implement dense retrieval interface
- implement hybrid merge
- implement deterministic final ranking
- implement answer generation with citations
- show results in app

## Milestone 3: Structured evidence ✅
- implement evidence extraction
- render evidence table in app
- store real demo/evaluation artifacts

## Milestone 4: Evaluation ✅
- define small eval dataset format
- add retrieval and answer checks
- implement evaluation runner

## Milestone 5: Agentic orchestration ✅
- add a custom LLM-backed planner and synthesis path
- add multi-query branch execution over the existing retrieval stack
- add deduplication, deterministic stopping, and baseline comparison
- expose agent reports through CLI / JSON artifacts instead of a heavier UI

## Milestone 6: Streamlit demo surface ✅
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

### Milestone 7: Reproducible demo and evaluation suite ✅
- define a fixed interview/demo query set
- build or document a fixed local PubMed abstract corpus for repeatable demos
- expand the evaluation dataset beyond the minimal example fixture
- generate a saved evaluation report with aggregate and per-query results
- compare baseline, hybrid retrieval, and agent workflow outputs
- document expected demo commands and sample outputs

### Milestone 8: Evidence quality and faithfulness ✅
- add citation faithfulness checks for generated answers
- detect answers that cite unsupported or weakly supported claims
- add an explicit insufficient-evidence outcome where appropriate
- enrich evidence records with fields such as population, intervention,
  comparator, outcome, study type, and effect direction when available
- surface conflicting or mixed evidence instead of forcing a single conclusion
- keep these checks inspectable and testable with local fixtures

### Milestone 9: Search strategy and agent traceability 鉁?
- make query rewriting and branch planning easier to inspect
- show original query, rewritten query, branch queries, and retrieval rationale
- add branch-level retrieval diagnostics and stopping reasons
- report whether the agent improved retrieval coverage over the baseline
- keep agent stopping deterministic and tied to evidence sufficiency

### Milestone 10: Demo console polish ✅
- improve the Streamlit demo as a lightweight review console
- prioritize evidence table filtering, sorting, and readable trace views
- add export paths for Markdown, JSON, or CSV reports
- keep Streamlit as a presentation layer over existing workflow outputs
- do not spend early effort converting Streamlit into a full API client unless
  there is a concrete demo or architecture need

## Track B: Engineering stack expansion

### Milestone 11: FastAPI service layer ✅
- add a thin FastAPI API layer around the existing core package
- keep retrieval, generation, extraction, evaluation, and agent logic in
  `src/bioevidence/`
- define typed request and response schemas for baseline and agent queries
- add health and metadata endpoints for local service checks
- return retrieved evidence, structured evidence rows, citations, final answer,
  and trace metadata in API responses
- add API tests using FastAPI's test client

Candidate endpoint shape:

```text
GET  /api/v1/health
GET  /api/v1/corpora
POST /api/v1/query/baseline
POST /api/v1/query/agent
POST /api/v1/evaluations/run
```

### Milestone 12: Docker packaging ✅
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

### Quality gates
- add or expand CI for tests
- add linting and formatting checks
- consider a focused type-checking pass for stable core modules
- add an evaluation smoke test that can run quickly in CI

### Documentation
- keep `docs/DECISIONS.md` updated for meaningful architecture decisions
- add `docs/EVALUATION.md` when evaluation reports become central
- add `docs/DEMO_SCRIPT.md` for interview or portfolio walkthroughs
- add `docs/LIMITATIONS.md` to document medical, data, and model limitations
- keep README focused on the shortest practical demo path first

### Engineering hygiene ✅
- split `agent/workflow.py` into `workflows/` package: `models.py`,
  `baseline.py`, `agent.py`, `retrieval_stack.py`
- move `retrieval/rerank.py` into `retrieval/ranking.py` with a cleaner name
- consolidate `api/` and `app/` into `interfaces/api/` and `interfaces/web/`
- remove placeholder files: `examples/milestone*.json`, `notebooks/exploration.ipynb`,
  `scripts/bootstrap.py`, `scripts/build_index.py`, `scripts/demo_query.py`
- align `data/` layout to a source-first naming convention:
  `data/corpora/` for corpora, `data/evaluations/` for eval sets

## Priority guidance

Short-term priority:
- Milestone 7: reproducible demo and evaluation suite
- Milestone 8: evidence quality and faithfulness
- Milestone 11: FastAPI service layer

Medium-term priority:
- Milestone 9: search strategy and agent traceability
- Milestone 12: Docker packaging
- quality gates and documentation polish

Lower priority unless needed later:
- Streamlit-as-API-client conversion
- Docker Compose for multiple services
- full-text ingestion, PDF parsing, browser automation, or database adoption
