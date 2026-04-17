# ROADMAP

## Milestone 0: Scaffold
- create repository layout
- create importable package
- add pyproject.toml
- add placeholder tests
- add basic app entrypoint
- add docs

## Milestone 1: PubMed ingestion
- implement PubMed search client
- normalize metadata into internal schema
- save sample raw and processed artifacts

## Milestone 2: RAG baseline
- implement lexical retrieval
- implement dense retrieval interface
- implement hybrid merge
- implement rerank stub
- implement answer generation with citations
- show results in app

## Milestone 3: Structured evidence
- implement evidence extraction
- render evidence table in app
- store example outputs

## Milestone 4: Evaluation
- define small eval dataset format
- add retrieval and answer checks
- implement evaluation runner

## Milestone 5: Agentic orchestration
- add planner
- add multi-query workflow
- add deduplication and stopping criteria
- compare with baseline
