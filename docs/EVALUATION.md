# Evaluation

BioEvidence Copilot keeps evaluation local and file-based so demo runs are
repeatable and inspectable.

## Demo Dataset

The tracked demo dataset lives at:

```text
data/evaluations/demo/demo_eval_dataset.jsonl
```

Each row uses the same JSONL schema as the runtime evaluation harness:

- `id`
- `query`
- `gold_pmids` or `gold_citations`
- optional `reference_answer`
- optional `top_k`

The file contains stable demo questions for interview walkthroughs. Large raw
downloads and caches under `data/` remain ignored, but curated eval and corpus
artifacts are intentionally trackable.

## Evidence Extraction Pilot

The v0.3 schema-development pilot lives at:

```text
data/evaluations/evidence_extraction/pilot_annotations.jsonl
```

Each row references one PMID in the tracked demo corpus and contains a query,
review status, and `ModelEvidenceExtraction` target. The loader validates the
closed schema and rejects outcome evidence spans that are not verbatim
substrings of the referenced abstract.

The initial 20 rows intentionally mix direct, indirect, and unrelated pairs.
They remain `draft`; they pressure-test the annotation contract but are not yet
gold labels and must not support model-quality claims until human review. See
`docs/EVIDENCE_ANNOTATION_GUIDE.md` for field definitions and review rules.

Validate the tracked annotations against the current corpus:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/validate_extraction_annotations.py
```

## Building Real Local Data

Seed a small PubMed corpus with real E-utilities abstracts:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/seed_demo_corpus.py --retmax-per-topic 30 --output-dir data/corpora/demo
```

The default topics cover asthma, type 2 diabetes, statins, melanoma
immunotherapy, and dietary sodium / hypertension. The combined corpus is written
to:

```text
data/corpora/demo/processed/demo.documents.jsonl
```

Convert BioASQ Task B data into the same local file conventions:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/convert_bioasq.py --input tmp/BioASQ-training13b/training13b.json --corpus-output-dir data/corpora/bioasq --eval-output-dir data/evaluations/bioasq
```

The converter writes:

- `data/evaluations/bioasq/bioasq13b_eval.jsonl`
- `data/corpora/bioasq/processed/bioasq13b_snippets.documents.jsonl`

The eval file follows the existing `EvaluationItem` schema. The BioASQ snippet
corpus follows the same `*.documents.jsonl` shape as the PubMed corpus, so the
retriever can load it by using `data/corpora/bioasq` as the data directory.

## Running Evaluation

Baseline mode:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --mode baseline --output tmp/demo-baseline-report.json
```

Agent mode:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --mode agent --output tmp/demo-agent-report.json
```

BioASQ smoke mode:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_eval.py --dataset data/evaluations/bioasq/bioasq13b_eval.jsonl --data-dir data/corpora/bioasq --mode baseline --limit 20 --output tmp/bioasq-smoke-report.json
```

Use `--limit` for BioASQ smoke runs. Full `training13b` evaluation contains
5,389 questions over a 41,299-document snippet corpus and is intentionally much
larger than the demo dataset.

The tracked example report at `data/evaluations/demo/demo_eval_report.json` shows the
expected report shape without requiring a live run.

## CI Smoke Test

GitHub Actions runs a one-item baseline evaluation smoke test:

```powershell
python scripts/run_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --mode baseline --limit 1
```

This is intentionally a smoke test, not a benchmark. It verifies that the
tracked demo dataset, local corpus loading, retrieval workflow, evidence table,
metric computation, and report formatting still work in a clean CI environment.
When embedding credentials are not configured, the retrieval stack can fall back
to lexical-only ranking; that fallback is acceptable for the CI smoke path.

## Metrics

Retrieval metrics:

- `hit_at_k`: whether any gold PMID appears in the top retrieved PMIDs
- `recall_at_k`: share of gold PMIDs found in the top retrieved PMIDs
- `mrr`: reciprocal rank of the first matching gold PMID

Citation metrics:

- `precision`: share of returned citations that are gold PMIDs
- `recall`: share of gold PMIDs cited by the answer
- `f1`: harmonic mean of citation precision and recall

Answer metrics:

- `exact_match`: normalized exact match against `reference_answer`
- `token_overlap`: token-level overlap F1 against `reference_answer`

Quality checks:

- `is_faithful`: no unsupported citations, no citation-list mismatch, and no
  forced conclusion when evidence is absent
- `unsupported_citations`: citations not present in the evidence table
- `missing_citations`: inline PMID citations missing from the citation list
- `forced_conclusion_without_evidence`: answer gives a conclusion with no
  evidence records
- `evidence_metadata`: deterministic study-type and effect-direction hints
  derived from evidence title and summary text

## Graph Augmentation Evaluation

Graph augmentation is evaluated against the same independently supplied gold
PMIDs as the literature baseline. For each question the evaluator:

1. records the baseline PMID ranking
2. asks Hetionet for linked entities and related terms
3. runs literature retrieval for each graph-derived query
4. combines rankings with reciprocal rank fusion rather than hand-tuned weights
5. compares top-k recall, hit rate, MRR, and newly recovered relevant PMIDs

Run a graph-enabled comparison after Neo4j has been populated:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_graph_eval.py --dataset data/evaluations/demo/demo_eval_dataset.jsonl --data-dir data/corpora/demo --limit 5 --output tmp/graph-gain-report.json
```

The result is an ablation report, not a claim that graph augmentation always
helps. Queries with no linked Hetionet entity retain the baseline ranking and
are reported with their graph status. The old GraphRAG prototype's
path-template-generated targets are intentionally not used here because they
would make the KG evaluate against its own output.
