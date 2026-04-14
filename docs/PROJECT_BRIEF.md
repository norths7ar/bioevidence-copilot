# PROJECT_BRIEF

## Name
BioEvidence Copilot

## One-line summary
A biomedical literature evidence assistant that first provides a grounded RAG baseline over PubMed abstracts, then evolves into an agentic workflow that can iteratively search, refine, deduplicate, extract, and synthesize evidence.

## Why this project exists
This repository is intended as a portfolio project for AI application engineering roles.

The project should demonstrate:
- retrieval-augmented generation (RAG)
- structured evidence extraction
- citation-grounded answer generation
- agentic orchestration over a multi-step workflow
- evaluation-aware engineering

It is not intended to be a production medical product or a clinical decision system.

## Product framing
The user asks a biomedical question such as:
- What evidence exists linking Drug X to Adverse Event Y?
- What recent literature discusses Drug X for Disease Y?
- What are the main literature-supported findings around a given biomedical topic?

The system should:
1. search PubMed
2. retrieve candidate abstracts and metadata
3. rank and rerank evidence
4. extract structured evidence records
5. generate an answer with citations
6. in a later stage, automatically refine search strategy and evidence gathering through an agentic workflow

## Two-stage roadmap

### Stage 1: RAG baseline
Goal: build a reliable, inspectable, citation-grounded QA pipeline.

Core capabilities:
- PubMed search client
- metadata + abstract ingestion
- hybrid retrieval (lexical + dense)
- reranking
- answer generation with citations
- evidence table output
- basic evaluation hooks

This stage should work without agent orchestration.

### Stage 2: Agentic RAG
Goal: add controlled multi-step orchestration on top of the existing pipeline.

Possible agent responsibilities:
- query reformulation
- multi-query planning
- iterative retrieval when evidence is insufficient
- deduplication across search branches
- structured evidence consolidation
- stopping when evidence is sufficient

The agent should orchestrate existing modules, not replace them.

## Non-goals
- clinical diagnosis
- treatment recommendation
- full autonomous scientific discovery
- full-text PDF parsing in the initial milestone
- generalized internet search
- browser-based research automation in v1

## Intended output format
The system should return:
1. a concise answer
2. citations linked to retrieved evidence
3. a structured evidence table with fields such as:
   - PMID
   - title
   - year
   - journal
   - entities of interest
   - short evidence summary
   - relevance score

## Technical preferences
- Python-first
- modular architecture
- provider-agnostic LLM interface
- evaluation-friendly design
- simple app/demo layer

## Success criteria for initial scaffold
A successful initial scaffold should include:
- repository structure
- configuration files
- importable Python package
- stubs for all major modules
- a lightweight app entrypoint
- placeholder tests
- basic documentation