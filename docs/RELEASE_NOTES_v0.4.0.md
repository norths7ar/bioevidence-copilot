# BioEvidence Copilot v0.4.0

v0.4.0 completes the second evidence-extraction experiment and makes the
published adapter reproducible from a fresh repository checkout.

## Expanded extraction experiment

- Expands the model-assisted draft dataset from 60 to 120 query-PMID pairs over
  108 unique PMIDs.
- Preserves every v1 PMID split before assigning unseen PMIDs, preventing the
  original adapter's training documents from leaking into the expanded test.
- Trains a 72-step QLoRA v2 adapter on an NVIDIA RTX 5070 12 GB.
- Compares rules, the prompted base model, adapter v1, and adapter v2 on the
  same 13-row PMID-held-out draft test set.

Adapter v2 improves evidence-status accuracy from 0.462 to 0.615,
study-design accuracy from 0.692 to 0.923, and semantic-field token F1 from
0.595 to 0.681 over v1. The report also records lower outcome-name/span F1,
higher latency, and the unresolved direct-to-indirect calibration error.

## Published adapter

The portable PEFT adapter is published at
[`n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2`](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2)
at revision `20ae7837207fcb697ac99d71961e99d0aebcb4ab`. A fresh download matched
all six manifest-covered release files by size and SHA-256.

## Reproducible setup

The new setup command downloads the pinned public snapshot, verifies its
release manifest, and atomically installs only the runtime files:

```powershell
conda activate bioevidence-training
python scripts/setup_extraction_adapter.py
```

Repeated runs validate and reuse the existing installation. Hugging Face and
GPU dependencies remain confined to the separate training environment, so the
normal API and Docker dependency boundary is unchanged.

## Summary report

`docs/EXTRACTION_MODEL_REPORT.md` presents the experimental boundary, complete
four-system metric table, v1-to-v2 trade-offs, status confusion, product
safeguards, and exact reproduction path in one reviewable document.
