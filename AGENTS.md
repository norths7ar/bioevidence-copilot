# AGENTS.md

## Project identity
This repository is a product-style portfolio project, not a throwaway demo.
The goal is to build a biomedical literature evidence assistant in two stages:

1. Stage 1: citation-grounded RAG baseline
2. Stage 2: agentic orchestration on top of the same RAG system

Do not redesign the project into an unrelated chatbot, general assistant, or toy tool-calling demo.

## Working style
- Prefer small, reviewable changes.
- Before large edits, inspect the surrounding files and preserve architectural consistency.
- When adding a new module, also add minimal tests or stubs.
- Keep code readable and boring over clever.
- Avoid over-engineering, speculative abstractions, and unnecessary framework complexity.

## Engineering principles
- Python 3.12
- Full interpreter path required: C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe
- Use pathlib.Path instead of os.path when practical.
- Prefer typed Python and dataclasses / pydantic where appropriate.
- Keep side effects at the edges.
- Separate retrieval, generation, extraction, and agent orchestration concerns.
- Do not tightly couple the agent layer to a single model provider.
- Make intermediate outputs inspectable whenever possible.

## Scope constraints
Stage 1 should work on PubMed metadata + abstracts first.
Do not start with PDF parsing, browser automation, or full-text ingestion unless explicitly requested.
Do not introduce a database unless there is a clear need.
Local file-based artifacts are acceptable for the initial scaffold.

## Evaluation requirements
This project must support evaluation.
Whenever retrieval or answer generation logic is added, leave clear seams for:
- retrieval metrics
- answer faithfulness / citation checks
- future RAG evaluation datasets

## App expectations
The app UI should be lightweight.
Prioritize a usable demo flow over polished frontend design.
The core demo should show:
- user query
- retrieved evidence
- structured evidence table
- final answer with citations

## Documentation discipline
If you make a meaningful architectural decision, update docs/DECISIONS.md.
If implementation diverges from docs, update the docs in the same change.

## What to avoid
- giant God classes
- hidden global state
- hardcoded magic values without explanation
- framework-first architecture
- fake “agent” behavior that is just one LLM call with tool wrappers