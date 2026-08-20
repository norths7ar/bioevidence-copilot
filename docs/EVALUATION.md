# Evaluation

BioEvidence Copilot keeps evaluation file-based so every run can preserve its
inputs, per-item outputs, and aggregate metrics without a separate experiment
service.

## Datasets

| Purpose | Tracked input |
| --- | --- |
| repeatable baseline and agent checks | `data/evaluations/demo/demo_eval_dataset.jsonl` |
| BioASQ-derived retrieval evaluation | `data/evaluations/bioasq/` |
| extraction annotations | `data/evaluations/evidence_extraction/*.jsonl` |
| extraction split and run summaries | `data/evaluations/evidence_extraction/*.json` |

Runtime evaluation rows contain:

- `id`
- `query`
- `gold_pmids` or `gold_citations`
- optional `reference_answer`
- optional `top_k`

Large raw downloads and caches are ignored. Curated corpus, annotation, split,
and summary artifacts are tracked when they are required to reproduce a result.

## Baseline and agent evaluation

Run the tracked demo dataset against the local corpus:

```powershell
python scripts/run_eval.py `
  --dataset data/evaluations/demo/demo_eval_dataset.jsonl `
  --data-dir data/corpora/demo `
  --mode baseline `
  --output tmp/demo-baseline-report.json
```

Use `--mode agent` for the agent workflow and `--limit N` for a short smoke
run. The report contains per-question rankings, evidence rows, citations,
quality checks, and aggregate metrics.

The tracked example at `data/evaluations/demo/demo_eval_report.json` shows the
report shape without requiring a live provider.

## Retrieval and answer metrics

Retrieval:

- `hit_at_k`: whether any gold PMID appears in the top results
- `recall_at_k`: share of gold PMIDs recovered
- `mrr`: reciprocal rank of the first gold PMID

Citations:

- `precision`: share of returned citations that are gold PMIDs
- `recall`: share of gold PMIDs cited
- `f1`: harmonic mean of citation precision and recall

Answers:

- `exact_match`: normalized match against an optional reference answer
- `token_overlap`: token-level overlap F1 against that answer

Deterministic quality checks report unsupported or missing citations and
conclusions produced without evidence. These checks verify workflow integrity;
they do not establish that every natural-language claim is scientifically
correct.

## Evidence-extraction comparison

All extraction modes consume the same query-document annotations and target the
same `ModelEvidenceExtraction` schema.

Run the deterministic baseline against the tracked v2 held-out split:

```powershell
python scripts/run_extraction_eval.py `
  --backend rules `
  --dataset artifacts/training/evidence_extraction/training_v2_sft/test.annotations.jsonl `
  --output artifacts/evaluations/extraction_v2_test_rules.json
```

For an OpenAI-compatible prompted model, configure `EXTRACTION_API_KEY`,
`EXTRACTION_BASE_URL`, and `EXTRACTION_MODEL`, then run:

```powershell
python scripts/run_extraction_eval.py `
  --backend prompted `
  --dataset artifacts/training/evidence_extraction/training_v2_sft/test.annotations.jsonl `
  --output artifacts/evaluations/extraction_v2_test_prompted_base.json
```

For the published local adapter, first prepare the pinned release and generate
the tracked v2 split as described in the
[training guide](../training/evidence_extraction/README.md), then run:

```powershell
python scripts/run_extraction_eval.py `
  --backend local `
  --adapter-path artifacts/models/bioevidence-qwen3-4b-extraction-lora-v2 `
  --dataset artifacts/training/evidence_extraction/training_v2_sft/test.annotations.jsonl `
  --output artifacts/evaluations/extraction_v2_test_qlora_adapter_v2.json
```

The evaluator reports:

- strict JSON parse and schema-validity rates
- evidence-status and study-design accuracy
- semantic-field and outcome metrics
- verbatim-span grounding and overlap
- latency and failure categories

The current rules, prompted base, adapter v1, and adapter v2 results are in the
[extraction model report](EXTRACTION_MODEL_REPORT.md). Machine-readable
configuration and aggregates are tracked in
`data/evaluations/evidence_extraction/qlora_training_v2_summary.json`.

## Graph-gain evaluation

Graph augmentation is evaluated against independently supplied gold PMIDs, not
against graph-generated targets. For each question the evaluator:

1. records the baseline PMID ranking;
2. obtains Hetionet-linked entities and related terms;
3. retrieves literature for each graph-derived query;
4. merges rankings with reciprocal rank fusion;
5. reports recall, hit rate, MRR, and newly recovered relevant PMIDs.

After Neo4j has been populated:

```powershell
python scripts/run_graph_eval.py `
  --dataset data/evaluations/demo/demo_eval_dataset.jsonl `
  --data-dir data/corpora/demo `
  --limit 5 `
  --output tmp/graph-gain-report.json
```

Queries without a linked entity retain the baseline ranking and record the
graph status. The report is an ablation comparison, not an assumption that
query expansion always helps.

## Preparing real local data

Seed a small PubMed corpus:

```powershell
python scripts/seed_demo_corpus.py `
  --retmax-per-topic 30 `
  --output-dir data/corpora/demo
```

Convert BioASQ Task B questions and snippets into the existing JSONL contracts:

```powershell
python scripts/convert_bioasq.py `
  --input tmp/BioASQ-training13b/training13b.json `
  --corpus-output-dir data/corpora/bioasq `
  --eval-output-dir data/evaluations/bioasq
```

Then run a bounded retrieval smoke:

```powershell
python scripts/run_eval.py `
  --dataset data/evaluations/bioasq/bioasq13b_eval.jsonl `
  --data-dir data/corpora/bioasq `
  --mode baseline `
  --limit 20
```

## CI smoke

CI runs one baseline item against tracked artifacts:

```powershell
python scripts/run_eval.py `
  --dataset data/evaluations/demo/demo_eval_dataset.jsonl `
  --data-dir data/corpora/demo `
  --mode baseline `
  --limit 1
```

The smoke test checks dataset loading, retrieval, evidence construction,
metrics, and report formatting without requiring provider credentials. It is
not a model-quality benchmark.
