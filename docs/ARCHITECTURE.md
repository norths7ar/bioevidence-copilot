# ARCHITECTURE

## System overview

The system is organized into six main layers:

1. Ingestion
2. Retrieval
3. Generation
4. Extraction
5. Agent orchestration
6. Evaluation

These layers should remain loosely coupled.

## Recommended initial flow

### Stage 1 flow
User query
-> retrieval query normalization
-> PubMed candidate fetch / local index lookup
-> hybrid retrieval
-> rerank
-> evidence extraction
-> answer generation
-> app output

### Stage 2 flow
User query
-> planner
-> one or more retrieval branches
-> deduplication / merge
-> sufficiency check
-> evidence extraction
-> answer generation
-> final output

## Data model expectations
At minimum define schemas for:
- Query
- Document
- RetrievedCandidate
- EvidenceRecord
- AnswerBundle

## App contract
The app should make intermediate artifacts visible:
- rewritten query if any
- top retrieved papers
- evidence table
- final answer

## Initial implementation constraints
- start with local development only
- use environment variables for secrets
- keep external dependencies moderate
- do not commit to a heavyweight framework too early