# Limitations

BioEvidence Copilot is a research-engineering project. It is not
a clinical product, medical device, diagnostic tool, treatment recommender, or
substitute for professional medical judgment.

## Data Limitations

- The current core pipeline is designed around PubMed metadata and abstracts.
  It does not ingest full-text articles, supplements, trial registries, PDFs, or
  clinical guidelines.
- The curated demo corpus is small and optimized for repeatable demos. It is
  not a complete evidence base for any biomedical question.
- BioASQ conversion support is useful for evaluation experiments, but converted
  snippets are not equivalent to a full literature review.
- PubMed search and metadata can change over time; tracked local artifacts are
  used to keep demos reproducible.

## Retrieval Limitations

- Lexical retrieval can miss relevant papers that use different terminology.
- Dense retrieval depends on an external embedding backend when configured. If
  the backend is unavailable, the system falls back to lexical-only ranking.
- Hybrid scoring and deterministic final ranking are inspectable but simple;
  they should not be interpreted as a comprehensive evidence ranking model.

## Generation and Citation Limitations

- Baseline answer generation is templated evidence stitching, not expert
  biomedical synthesis.
- Agent synthesis can use an LLM backend, which may omit nuance, overstate
  evidence, or fail to express uncertainty unless checks and prompts constrain
  it.
- Citation checks verify that returned PMIDs are present in the evidence table;
  they do not prove that every natural-language claim is fully supported by the
  cited abstract.
- Conflicting evidence is surfaced when detected by deterministic fields, but
  the system does not perform a full risk-of-bias or guideline-grade evidence
  assessment.

## Agent Limitations

- LangGraph provides workflow routing and streaming, but planning, stopping,
  and biomedical retrieval behavior remain project-specific and require
  evaluation. It is not a general-purpose autonomous research agent.
- Branch planning can broaden coverage, but it can also retrieve redundant or
  weakly relevant papers.
- Stopping is deterministic and tied to evidence count and relevance thresholds;
  it is a pragmatic demo control, not a clinical sufficiency standard.

## Knowledge Graph Limitations

- Hetionet is a bounded, historical biomedical graph. Missing or newer entities
  and relationships cannot be discovered through this branch.
- Entity linking is name-based and does not yet provide ontology synonyms,
  abbreviation disambiguation, or context-aware biomedical NER.
- A graph edge or path is treated as a discovery hint, not as citable evidence.
  Final answer support still depends on retrieved PubMed records.
- Graph query expansion can introduce broad or irrelevant terms. Its value must
  be judged by PMID recall deltas, not by the number of returned graph paths.

## Engineering Limitations

- The FastAPI service is a local service boundary. It does not include
  authentication, authorization, rate limiting, persistence, background job
  orchestration, or production observability.
- Docker packages the API, and Compose adds a local Neo4j service. Neither is a
  cloud deployment configuration, and Hetionet import remains a separate data
  preparation step.
- Streamlit remains a lightweight read-only review console and is not currently
  implemented as a separate API client.

## Appropriate Use

Appropriate uses include:

- demonstrating RAG, evidence extraction, evaluation, and agent orchestration
  architecture
- exploring local PubMed abstract retrieval behavior
- testing evaluation and citation-faithfulness checks on small fixtures
- engineering discussion and architecture review

Inappropriate uses include:

- making patient-specific medical decisions
- generating clinical recommendations
- claiming comprehensive systematic-review coverage
- treating generated answers as authoritative biomedical conclusions
