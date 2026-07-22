---
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
  - qwen3
  - qlora
  - information-extraction
  - biomedical
  - structured-output
language:
  - en
license: apache-2.0
---

# BioEvidence Qwen3-4B Extraction LoRA v2

This is the second QLoRA adapter for query-focused structured extraction from
one PubMed title and abstract. It targets the versioned
`ModelEvidenceExtraction` JSON schema used by BioEvidence Copilot.

The adapter is an engineering experiment in strict structured output and local
GPU deployment. It is not a clinical model and its draft annotations are not an
expert biomedical benchmark.

## Changes from v1

- Expanded the model-assisted draft dataset from 60 to 120 query-PMID pairs.
- Increased direct labels from 8 to 20 and indirect labels from 29 to 65.
- Preserved every v1 PMID split before assigning new PMIDs, keeping the original
  seven test annotations directly comparable.
- Kept the base model, LoRA architecture, learning rate, seed, effective batch
  size, and approximate three-epoch schedule fixed.

## Training

- Base model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Base revision: `7744afa8566e264af1a92a806d8d9aae00cc7c78`
- Training rows: 94
- Development rows: 13
- Test rows: 13
- QLoRA: 4-bit, rank 16, alpha 16
- Maximum sequence length: 4096
- Optimizer steps: 72
- Effective batch size: 4
- Learning rate: `2e-4`
- Training time: 493.4 seconds on an NVIDIA RTX 5070 12 GB
- Peak PyTorch allocated VRAM: 5.81 GiB
- Development loss: 1.104 before training, 0.247 after training

## Held-out comparison

All systems used the same 13-row PMID-held-out draft test set and deterministic
decoding.

| Metric | Rules | Prompted base | Adapter v1 | Adapter v2 |
| --- | ---: | ---: | ---: | ---: |
| Strict JSON parse rate | 1.000 | 1.000 | 1.000 | 1.000 |
| Schema validity | 1.000 | 0.923 | 1.000 | 1.000 |
| Grounding rate | n/a | 0.615 | 0.923 | 0.923 |
| Evidence-status accuracy | 0.308 | 0.769 | 0.462 | 0.615 |
| Study-design accuracy | 0.769 | 0.769 | 0.692 | 0.923 |
| Semantic-field token F1 | 0.487 | 0.512 | 0.595 | 0.681 |
| Outcome-name token F1 | 0.231 | 0.326 | 0.632 | 0.526 |
| Outcome-direction accuracy | 0.308 | 0.410 | 0.692 | 0.692 |
| Evidence-span token F1 | 0.409 | 0.342 | 0.577 | 0.515 |
| Evidence-span support rate | 1.000 | 0.737 | 0.923 | 0.923 |
| Mean generation time | n/a | 19.58 s | 6.81 s | 11.23 s |

Adapter v2 fixed the observed `indirect -> none` collapse on this test: all four
indirect rows remained indirect. It did not solve direct-status calibration;
all three direct rows were classified as indirect. It also traded lower
outcome-name/span metrics and higher latency for improved status, study-design,
and semantic-field metrics over adapter v1.

## Product behavior

BioEvidence Copilot validates every generated object against its Pydantic
schema and requires verbatim evidence spans. Invalid model output falls back to
the deterministic rules backend, with attempted backend, used backend, and
fallback reason exposed per evidence row.

A three-paper product smoke completed with three model successes and no
fallbacks. The v1 schema failure for PMID `41772161` was resolved by adapter v2.

## Limitations

- All 120 labels are model-assisted drafts.
- The held-out set contains only 13 rows over 10 unique PMIDs.
- Direct-status calibration remains weak.
- Results are specific to one schema, prompt, split, base revision, and local
  inference runtime.
- Training and evaluation use PubMed titles and abstracts, not full text.
- Evidence-span grounding verifies copying, not scientific interpretation.
- Do not use the adapter for clinical decision-making.

## Reproduction

The source repository contains the schema, annotations, split manifest,
training script, evaluation code, aggregate result summary, and optional product
backend. The adapter package contains LoRA weights and tokenizer/chat-template
files, but does not redistribute the training dataset or base-model weights.
