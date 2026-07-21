# Evidence extraction training

This directory contains the optional local-model environment and experiment
entry points for the v0.3 evidence-extraction track. The product environment
does not depend on this stack.

## Reproduce the environment

The checked-in lock file is an exact snapshot from Windows 11, Python 3.12.13,
and an NVIDIA RTX 5070 12 GB. From the repository root:

```powershell
conda activate bioevidence-training
pip install -r training/evidence_extraction/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .
pip check
```

The editable install exposes the repository's `src/` package and keeps schema,
prompt, and metric changes immediately visible to the experiment scripts. The
dependency direction stays one-way: the training environment includes the
product package, while the product environment does not include Unsloth.

To keep model and compilation caches off the system drive, configure paths
before loading Unsloth:

```powershell
$env:HF_HOME = "E:/huggingface-cache"
$env:TRITON_CACHE_DIR = "E:/triton-cache"
$env:UNSLOTH_COMPILE_LOCATION = "E:/unsloth-compiled-cache"
```

Download the pinned model snapshot:

```powershell
hf download unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit --revision 7744afa8566e264af1a92a806d8d9aae00cc7c78
```

## Run the prompted baseline

Start with one end-to-end schema and grounding check:

```powershell
python training/evidence_extraction/scripts/smoke_test.py
```

Then evaluate all 20 query-document pairs:

```powershell
python training/evidence_extraction/scripts/run_local_extraction_eval.py `
  --model-label "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit@7744afa"
```

Reports are written under ignored `artifacts/` directories. The full runner
preserves every prediction and raw model response so parse, schema, grounding,
and field-level failures remain inspectable.

## Build the SFT dataset

Convert the validated annotations into Qwen chat-format JSONL:

```powershell
python training/evidence_extraction/scripts/build_sft_dataset.py
```

The builder writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, and
`manifest.json` under `artifacts/training/evidence_extraction/pilot_sft/`.
It also refreshes the tracked aggregate `pilot_split_manifest.json` beside the
source annotations.
Each split also gets a `*.annotations.jsonl` file in the generated directory so
the same held-out labels can be passed directly to the extraction evaluator.
Assignments are deterministic for a fixed seed, and every query variant for the
same PMID stays in one split. Each row contains the exact runtime system/user
prompt, a compact JSON assistant target, and inspectable annotation metadata.

The 20-row pilot is enough to validate the complete data and training pipeline,
but dataset expansion remains the main Milestone 20 task before a meaningful
quality comparison. Because the pilot has only three `direct` labels, its
PMID-safe split leaves one direct example in each split; that is an honest smoke
test constraint, not a training recipe for the final experiment.

Validate the generated records without loading a model:

```powershell
python training/evidence_extraction/scripts/train_qlora_smoke.py --dry-run
```

Run the five-step QLoRA training check:

```powershell
python training/evidence_extraction/scripts/train_qlora_smoke.py
```

The smoke run uses rank-16 LoRA on the attention and MLP projections, 4-bit base
weights, BF16 when supported, effective batch size four, and response-only loss.
Training examples are rendered from the system/user generation prompt plus the
raw assistant JSON target. This deliberately excludes any assistant-side
thinking or tool-call scaffolding that a model chat template may otherwise add,
so the first supervised response character is `{` just as required at runtime.
It evaluates dev loss before and after training, saves the adapter, reloads it,
and writes an inspectable `report.json`. This is a pipeline validation run, not
the final hyperparameter configuration.

The first five-step run completed on the RTX 5070 in 38.85 seconds. Train loss
was 1.801, dev loss moved from 1.980 before training to 1.226 afterward, and
PyTorch reported 5.41 GiB peak allocated VRAM. The 66.1 MB adapter reloaded
successfully. These numbers establish that the training path works; the tiny
pilot makes the loss change unsuitable as a model-quality claim. The tracked
aggregate is `data/evaluations/evidence_extraction/qlora_smoke_summary.json`.

## Recorded baseline

The first local run used deterministic decoding, a 4,096-token context, and a
1,024-token output limit on the 20-row extraction pilot.

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

The local model loaded in 3.75 seconds, generated each item in 22.26 seconds on
average, and peaked at 4.38 GiB of allocated VRAM. It predicted 2.4 outcomes per
item while the pilot labels contain 0.6 on average. This over-extraction explains
most of the weaker outcome and span scores; the only JSON failure reached the
1,024-token output limit before closing the object.

## First expanded QLoRA comparison

The first expanded run combines the 20 pilot rows with 40 model-assisted draft
annotations. PMID-level splitting produced 46 train, 7 dev, and 7 held-out test
rows. The corrected 36-step adapter was compared with the rules baseline and the
prompted base model on exactly the same test split.

| Metric | Rules | Qwen3-4B prompted | Qwen3-4B QLoRA |
| --- | ---: | ---: | ---: |
| JSON parse rate | 1.000 | 1.000 | 1.000 |
| Schema validity | 1.000 | 1.000 | 1.000 |
| Grounding rate | n/a | 0.714 | 1.000 |
| Evidence status accuracy | 0.429 | 0.714 | 0.429 |
| Study design accuracy | 0.857 | 1.000 | 0.857 |
| Semantic-field token F1 | 0.571 | 0.540 | 0.668 |
| Outcome-name token F1 | 0.429 | 0.414 | 0.631 |
| Outcome direction accuracy | 0.429 | 0.536 | 0.714 |
| Evidence-span token F1 | 0.513 | 0.461 | 0.511 |
| Evidence-span support rate | 1.000 | 0.857 | 1.000 |

The adapter cut mean generation time from 20.08 to 7.34 seconds and achieved the
target strict JSON behavior. It did not improve evidence-status classification,
so this result supports structured-output specialization but not a broad claim
that the 60-row draft dataset improves every extraction dimension. The tracked
configuration and aggregate are in
`data/evaluations/evidence_extraction/qlora_training_v1_summary.json`.
