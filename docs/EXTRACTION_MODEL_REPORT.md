# Evidence extraction model report

## Outcome

BioEvidence Copilot now has a complete, inspectable structured-extraction
experiment: one versioned JSON contract, deterministic and prompted baselines,
PMID-safe draft data, two QLoRA adapters, held-out evaluation, guarded product
inference, and externally published weights.

Adapter v2 is the recommended experimental checkpoint. It improves status,
study-design, and semantic-field performance over v1, but it does not dominate
v1 on every metric and does not solve direct-evidence calibration.

## Experiment boundary

- Base model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Base revision: `7744afa8566e264af1a92a806d8d9aae00cc7c78`
- Hardware: NVIDIA RTX 5070 12 GB
- Labels: 120 model-assisted drafts over 108 unique PMIDs
- Label distribution: 20 direct, 65 indirect, 35 none
- Split unit: PMID, with every v1 PMID assignment preserved
- Held-out comparison: 13 rows over 10 unique PMIDs
- Test distribution: 3 direct, 4 indirect, 6 none

The test set is suitable for a reproducible engineering comparison, not a
biomedical benchmark or a broad model-quality claim.

## Held-out results

All four systems were evaluated on the same 13 rows. Model systems used
deterministic decoding.

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

## What changed in v2

The expanded data corrected the clearest v1 failure mode on this test set:
adapter v1 classified only one of four indirect examples as indirect, while v2
classified all four correctly. Study-design accuracy rose from 0.692 to 0.923,
and semantic-field token F1 rose from 0.595 to 0.681.

The trade-off is visible rather than hidden. Outcome-name token F1 fell from
0.632 to 0.526, evidence-span token F1 fell from 0.577 to 0.515, and mean
generation time rose from 6.81 to 11.23 seconds.

## Remaining status error

Adapter v2 status predictions on the held-out set were:

| Reference | Predicted | Rows |
| --- | --- | ---: |
| direct | indirect | 3 |
| indirect | indirect | 4 |
| none | indirect | 2 |
| none | none | 4 |

All direct examples remain under-confident and are classified as indirect. A
future training iteration should therefore target direct-versus-indirect
boundaries rather than simply add more randomly selected abstracts.

## Product safeguards

The optional local backend does not silently trust generated JSON. Product
inference validates the schema, enforces verbatim abstract spans, and falls back
to deterministic extraction when the model is unavailable or invalid. Each
evidence row exposes the attempted backend, backend actually used, and fallback
reason. A three-paper v2 product smoke completed with three model successes and
no fallback.

## Reproduce the public adapter path

From the `bioevidence-training` environment:

```powershell
python scripts/setup_extraction_adapter.py
$env:EXTRACTION_BACKEND="local"
$env:EXTRACTION_ADAPTER_PATH="artifacts/models/bioevidence-qwen3-4b-extraction-lora-v2"
python scripts/run_baseline.py `
  --query "asthma corticosteroids exacerbations randomized trial" `
  --top-k 3 `
  --output artifacts/runs/extraction_demo/local.json
```

The setup command pins Hugging Face revision
`20ae7837207fcb697ac99d71961e99d0aebcb4ab` and verifies the six files covered
by the published release manifest before installing them locally.

## Tracked evidence

- Aggregate configuration and results:
  `data/evaluations/evidence_extraction/qlora_training_v2_summary.json`
- Dataset and provenance: `training/evidence_extraction/DATASET_CARD.md`
- Published model card: `training/evidence_extraction/MODEL_CARD_V2.md`
- Split manifest:
  `data/evaluations/evidence_extraction/training_split_manifest.v2.json`
