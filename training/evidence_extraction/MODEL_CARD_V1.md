---
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - biology
  - medical
  - information-extraction
  - json
  - lora
  - qlora
  - unsloth
---

# BioEvidence Qwen3-4B Extraction LoRA v1

This is a QLoRA adapter for query-focused structured extraction from one PubMed
title and abstract. It was trained as part of
[BioEvidence Copilot](https://github.com/norths7ar/bioevidence-copilot) and
targets the repository's versioned `ModelEvidenceExtraction` JSON Schema.

The adapter is an engineering experiment in strict structured output and
grounded evidence spans. It is not a biomedical decision model.

## Intended use

Input consists of a biomedical query, a PubMed title, an abstract, and the v1
JSON Schema. Output is exactly one JSON object containing:

- query-specific evidence status;
- study design;
- population, intervention or exposure, and comparator when supported;
- outcome direction and result text;
- verbatim evidence spans copied from the abstract;
- a short evidence summary.

Use the adapter for reproducible extraction experiments, offline extraction
evaluation, and optional enrichment inside BioEvidence Copilot. Validate every
response against the schema and verify evidence spans against the source text.

Do not use it for diagnosis, treatment selection, clinical decision-making, or
unsupported claims beyond the supplied title and abstract.

## Base model

- Base: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Pinned training snapshot: `7744afa8566e264af1a92a806d8d9aae00cc7c78`
- Adapter format: PEFT LoRA
- Base-model license: Apache-2.0

## Training data

The v1 experiment contains 60 query-document annotations built from PubMed
titles and abstracts:

| Label | Rows |
| --- | ---: |
| direct | 8 |
| indirect | 29 |
| none | 23 |

Labels are model-assisted drafts. The 20-row pilot received a second-model
review; 40 expansion rows were produced in Claude Desktop and passed JSON,
Pydantic schema, duplicate-pair, and exact-span validation. Splits are assigned
by PMID to prevent the same document from crossing train, dev, and test.

| Split | Rows | Unique PMIDs |
| --- | ---: | ---: |
| train | 46 | 43 |
| dev | 7 | 5 |
| test | 7 | 5 |

Dataset provenance and split IDs are tracked in the project repository. The
adapter package does not redistribute the training dataset.

## Training procedure

- GPU: NVIDIA RTX 5070 12 GB
- Python: 3.12.13
- Quantization: 4-bit base weights
- Compute: BF16
- LoRA rank / alpha: 16 / 16
- Target modules: attention and MLP projections
- Trainable parameters: 33,030,144 of 4,055,498,240 (0.81%)
- Effective batch size: 4
- Optimizer: 8-bit AdamW
- Learning rate: 2e-4 with linear decay
- Steps / epochs: 36 / 3
- Maximum sequence length: 4,096
- Loss: response-only; the supervised response begins with `{`
- Training time: 238.3 seconds
- Peak PyTorch allocated VRAM: 5.81 GiB
- Dev loss: 0.971 before training, 0.244 after training

## Held-out evaluation

All systems were evaluated on the same seven PMID-held-out draft annotations
with deterministic decoding.

| Metric | Rules | Prompted base | QLoRA adapter |
| --- | ---: | ---: | ---: |
| Strict JSON parse rate | 1.000 | 1.000 | 1.000 |
| Schema validity | 1.000 | 1.000 | 1.000 |
| Grounding rate | n/a | 0.714 | 1.000 |
| Evidence-status accuracy | 0.429 | 0.714 | 0.429 |
| Study-design accuracy | 0.857 | 1.000 | 0.857 |
| Semantic-field token F1 | 0.571 | 0.540 | 0.668 |
| Outcome-name token F1 | 0.429 | 0.414 | 0.631 |
| Outcome-direction accuracy | 0.429 | 0.536 | 0.714 |
| Evidence-span token F1 | 0.513 | 0.461 | 0.511 |
| Evidence-span support rate | 1.000 | 0.857 | 1.000 |
| Mean generation time | n/a | 20.08 s | 7.34 s |

These results demonstrate strict JSON specialization and a working local
training/inference path. Seven draft test rows are too few for a general claim
about biomedical extraction quality, and evidence-status classification did not
improve over the prompted base model.

### Post-integration training-split diagnostic

After product integration, the adapter was also run over all 46 training rows.
This is a memorization and failure-mode diagnostic, not a held-out benchmark.

| Diagnostic | Result |
| --- | ---: |
| Strict JSON parse rate | 1.000 |
| Schema validity | 0.978 |
| Evidence-status accuracy | 0.630 |
| Evidence-span support rate | 0.935 |
| Mean generation time | 6.49 s |

The status confusion is strongly conservative: all 17 `none` rows were
classified correctly, but only 11 of 25 `indirect` rows remained `indirect`
and only one of four `direct` rows remained `direct`. Eleven `indirect` rows
were reduced to `none`; three `direct` rows were reduced to `indirect`. This
points to direct/indirect coverage and boundary examples as the first v2 data
priority rather than simply adding more `none` examples.

## Known limitations

- All 60 labels are draft, model-assisted annotations rather than expert gold.
- The held-out set contains only seven rows.
- Direct labels are absent for the asthma and melanoma query groups.
- Training and evaluation use titles and abstracts, not full text.
- Evidence-span grounding checks copying, not whether a selected span is the
  best scientific interpretation.
- The adapter can misclassify unrelated evidence as indirect.
- The adapter can emit a logically inconsistent `none` result while retaining
  population, intervention, or comparator fields. Product inference rejects
  that output and records a schema fallback to the rules backend.
- Results are specific to the pinned base model, schema, prompt, and split.

## Reproduction and inference

The repository contains the exact environment lock, schema, annotations, split
manifest, training script, evaluator, and aggregate report. After placing the
adapter in a local directory:

```powershell
conda activate bioevidence-training
python scripts/run_extraction_eval.py `
  --backend local `
  --adapter-path path/to/adapter `
  --dataset artifacts/training/evidence_extraction/training_v1_sft/test.annotations.jsonl
```

For optional product inference, set `EXTRACTION_BACKEND=local` and
`EXTRACTION_ADAPTER_PATH=path/to/adapter`. BioEvidence Copilot validates schema
and verbatim spans and falls back to its deterministic structured extractor when
the optional local runtime is unavailable or invalid.
