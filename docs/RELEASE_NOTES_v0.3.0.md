# BioEvidence Copilot v0.3.0

v0.3.0 adds a complete, inspectable fine-tuned evidence-extraction track to the
existing citation-grounded RAG and graph-augmented agent system.

## Highlights

- Defines a versioned query-focused `ModelEvidenceExtraction` JSON Schema with
  closed enums, status-dependent validation, and verbatim abstract spans.
- Adds a 60-row model-assisted draft dataset with tracked provenance and
  deterministic PMID-level train/dev/test splits.
- Establishes comparable deterministic-rule and prompted Qwen3-4B baselines.
- Trains a 36-step QLoRA adapter on an RTX 5070 12 GB and records the full
  environment, configuration, loss, latency, VRAM, and held-out metrics.
- Fixes assistant-side chat-template scaffolding so the supervised response and
  generated output begin with a strict JSON object.
- Adds an optional lazy local-adapter backend alongside legacy, deterministic,
  and OpenAI-compatible prompted modes.
- Reuses one extraction backend across baseline and agent branches, validates
  schema and grounding, and falls back to deterministic extraction on optional
  model failure.
- Surfaces semantic extraction fields in API, presentation, and evidence-table
  outputs without adding Unsloth or CUDA packages to the default product image.

## Published adapter

The PEFT adapter, tokenizer, chat template, model card, and release manifest are
published at:

[`n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v1`](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v1)

Pinned Hub revision:
`e6a61cd9749f373fc6c4fcdc3563b417ea57b401`.

A clean post-publication download matched every file size and SHA-256 recorded
in the release manifest.

## First held-out result

The seven-row PMID-held-out draft split produced:

| Metric | Prompted base | QLoRA adapter |
| --- | ---: | ---: |
| Strict JSON parse rate | 1.000 | 1.000 |
| Schema validity | 1.000 | 1.000 |
| Grounding rate | 0.714 | 1.000 |
| Evidence-status accuracy | 0.714 | 0.429 |
| Semantic-field token F1 | 0.540 | 0.668 |
| Outcome-name token F1 | 0.414 | 0.631 |
| Outcome-direction accuracy | 0.536 | 0.714 |
| Evidence-span token F1 | 0.461 | 0.511 |
| Mean generation time | 20.08 s | 7.34 s |

This supports the structured-output specialization claim. It does not establish
general biomedical superiority: all labels remain model-assisted drafts, the
test split is small, and evidence-status classification regressed.

## Compatibility

`EXTRACTION_BACKEND=legacy` remains the default, so existing product behavior
and the dependency-light API image are unchanged. Set the backend explicitly to
`rules`, `prompted`, or `local` to expose v1 semantic extraction fields.

Local inference requires the separate training environment and a downloaded
adapter directory. If the optional backend is missing, unavailable, invalid, or
ungrounded, the workflow logs the reason and uses deterministic extraction.

## Verification

- Ruff passed.
- Focused mypy passed across stable schema, evaluation, workflow, and graph
  modules.
- 142 pytest tests passed.
- The standard evaluation smoke test passed.
- The local adapter was exercised through both the shared extraction evaluator
  and the real `run_rag_pipeline` evidence-table path.
