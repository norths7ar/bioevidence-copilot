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
They remain a small schema-development pilot, with source and review provenance
stored in `pilot_dataset_metadata.json` and per-row stability status in the
annotation JSONL. See `docs/EVIDENCE_ANNOTATION_GUIDE.md` for field definitions
and review rules.

The first expansion queue is tracked as
`data/evaluations/evidence_extraction/expansion_candidates.v1.jsonl`, with its
source hash and sampling configuration in the adjacent manifest. Selection
bands describe how a pair entered the queue; they are not evidence-status
labels. The model-assisted draft job must still produce schema-valid, grounded
extractions before any row can enter the annotation dataset.

Validate the tracked annotations against the current corpus:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/validate_extraction_annotations.py
```

Render a local review packet:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/render_extraction_review.py
```

## Evidence extraction baselines

The extraction evaluator uses the same query-document annotations for every
backend and writes predictions, failure categories, per-item metrics, latency,
and aggregate metrics to JSON. Run the inspectable rule baseline with:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_extraction_eval.py --backend rules
```

For the prompt-only baseline, configure `EXTRACTION_API_KEY`,
`EXTRACTION_BASE_URL`, and `EXTRACTION_MODEL`, then run:

```powershell
C:/Users/jnkyl/miniconda3/envs/bioevidence-copilot/python.exe scripts/run_extraction_eval.py --backend prompted
```

The prompt-only backend sends the query, title, abstract, and versioned output
schema to an OpenAI-compatible chat endpoint. It then applies Pydantic schema
validation and exact abstract-span grounding. The initial report covers JSON
parse rate, schema validity, evidence status, study design, semantic-field token
F1, outcome matching and direction, span overlap/support, and latency. Provider
cost is not inferred when the compatible endpoint does not expose a stable
price contract; model name and experiment configuration should be recorded
alongside published benchmark results.

The shared product-side local adapter can be evaluated from the separate
training environment without importing Unsloth into the normal API image:

```powershell
conda activate bioevidence-training
python scripts/run_extraction_eval.py `
  --backend local `
  --adapter-path artifacts/training/evidence_extraction/qwen3_4b_qlora_v2/adapter `
  --dataset artifacts/training/evidence_extraction/training_v1_sft/test.annotations.jsonl `
  --output artifacts/evaluations/extraction_local_adapter.json
```

For an opt-in product run, set `EXTRACTION_BACKEND=local` and
`EXTRACTION_ADAPTER_PATH` to the adapter directory. `rules` and `prompted` use
the same validated contract; `legacy` preserves the original evidence rows.
Optional model failures fall back to the deterministic structured extractor.
Successful structured predictions are exposed alongside the existing evidence
row fields and do not require the API container to install the training stack.

The published v1 adapter can be downloaded before running the same evaluator:

```powershell
hf download n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v1 `
  --revision e6a61cd9749f373fc6c4fcdc3563b417ea57b401 `
  --local-dir artifacts/models/bioevidence-qwen3-4b-extraction-lora-v1
```

The first local prompted run uses the pinned 4-bit
`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` snapshot. Reproduce it from
the separate training environment with:

```powershell
conda activate bioevidence-training
python training/evidence_extraction/scripts/smoke_test.py
python training/evidence_extraction/scripts/run_local_extraction_eval.py --model-label "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit@7744afa"
```

| Metric | Rules | Qwen3-4B prompted |
| --- | ---: | ---: |
| JSON parse rate | 1.000 | 0.950 |
| Schema validity | 1.000 | 0.950 |
| Evidence status accuracy | 0.400 | 0.650 |
| Study design accuracy | 0.750 | 0.800 |
| Semantic-field token F1 | 0.400 | 0.553 |
| Outcome direction accuracy | 0.450 | 0.379 |
| Evidence-span token F1 | 0.461 | 0.299 |
| Evidence-span support rate | 1.000 | 0.815 |

The prompted model averaged 22.26 seconds per item and peaked at 4.38 GiB of
allocated VRAM on an RTX 5070 12 GB. It over-extracted outcomes (2.4 predicted
versus 0.6 labeled per item), which lowered outcome-direction and span metrics.
One response hit the 1,024-token output limit and ended as incomplete JSON. The
environment lock, model download command, and full-run entry point are documented
in `training/evidence_extraction/README.md`. The tracked machine-readable
aggregate is `data/evaluations/evidence_extraction/baseline_summary.json`.

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
