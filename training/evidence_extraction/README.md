# Evidence extraction training

This directory contains the optional local environment and scripts used to
train and evaluate the query-focused evidence-extraction adapters. The normal
product environment does not depend on Unsloth or the GPU training stack.

## Environment

The lock file records the Windows 11, Python 3.12.13, CUDA, and package versions
used on an NVIDIA RTX 5070 12 GB.

```powershell
conda activate bioevidence-training
pip install -r training/evidence_extraction/requirements.lock.txt `
  --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .
pip check
```

Optional cache locations can keep model and compilation artifacts off the
system drive:

```powershell
$env:HF_HOME="<path-to-hugging-face-cache>"
$env:TRITON_CACHE_DIR="<path-to-triton-cache>"
$env:UNSLOTH_COMPILE_LOCATION="<path-to-unsloth-compile-cache>"
```

Download the pinned base snapshot:

```powershell
hf download unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit `
  --revision 7744afa8566e264af1a92a806d8d9aae00cc7c78
```

## Prompted baseline

Run one schema and grounding check:

```powershell
python training/evidence_extraction/scripts/smoke_test.py
```

Run the prompted model over the original pilot:

```powershell
python training/evidence_extraction/scripts/run_local_extraction_eval.py `
  --model-label "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit@7744afa"
```

Predictions and raw responses are written under ignored `artifacts/` paths.

## Build the v2 SFT dataset

The current experiment combines the pilot, v1 expansion, and v2 expansion.
Every recorded PMID assignment is fixed by the tracked v2 split manifest.

```powershell
python training/evidence_extraction/scripts/build_sft_dataset.py `
  --dataset data/evaluations/evidence_extraction/pilot_annotations.jsonl `
  --dataset data/evaluations/evidence_extraction/expansion_annotations.v1.jsonl `
  --dataset data/evaluations/evidence_extraction/expansion_annotations.v2.jsonl `
  --metadata data/evaluations/evidence_extraction/training_dataset_metadata.v2.json `
  --fixed-split-manifest data/evaluations/evidence_extraction/training_split_manifest.v1.json `
  --output-dir artifacts/training/evidence_extraction/training_v2_sft `
  --manifest-output artifacts/training/evidence_extraction/training_v2_sft/split_manifest.json
```

The generated directory contains Qwen chat-format `train.jsonl`, `dev.jsonl`,
and `test.jsonl`, plus matching `*.annotations.jsonl` files for evaluation.
Examples use the runtime system/user prompt and a compact JSON assistant target.

## Train QLoRA v2

Validate data and configuration without loading a model:

```powershell
python training/evidence_extraction/scripts/train_qlora_smoke.py `
  --train-file artifacts/training/evidence_extraction/training_v2_sft/train.jsonl `
  --dev-file artifacts/training/evidence_extraction/training_v2_sft/dev.jsonl `
  --output-dir artifacts/training/evidence_extraction/qwen3_4b_qlora_v2 `
  --max-steps 72 `
  --dry-run
```

Remove `--dry-run` to train. The recorded v2 run used:

- 4-bit base weights with BF16 compute
- rank 16 and alpha 16 LoRA on attention and MLP projections
- effective batch size 4
- learning rate `2e-4`
- response-only loss beginning at the assistant JSON object
- maximum sequence length 4,096
- 72 optimizer steps

It completed in 493.4 seconds, peaked at 5.81 GiB of allocated PyTorch VRAM,
and reduced dev loss from 1.104 to 0.247. The script saves and reloads the
adapter before reporting success.

## Evaluate the adapter

```powershell
python scripts/run_extraction_eval.py `
  --backend local `
  --adapter-path artifacts/training/evidence_extraction/qwen3_4b_qlora_v2/adapter `
  --dataset artifacts/training/evidence_extraction/training_v2_sft/test.annotations.jsonl `
  --output artifacts/evaluations/extraction_local_adapter_v2.json
```

The full four-system comparison and its limitations are in
[`docs/EXTRACTION_MODEL_REPORT.md`](../../docs/EXTRACTION_MODEL_REPORT.md).
Tracked configuration and aggregate results are in
`data/evaluations/evidence_extraction/qlora_training_v2_summary.json`.

## Published adapters

- [v2](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2)
  at revision `20ae7837207fcb697ac99d71961e99d0aebcb4ab`
- [v1](https://huggingface.co/n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v1)
  at revision `e6a61cd9749f373fc6c4fcdc3563b417ea57b401`

`prepare_adapter_release.py` builds a non-destructive release directory,
rewrites the machine-local base-model path to the pinned public model ID, and
records SHA-256 hashes. Model weights remain outside Git; model and dataset
cards stay in this directory.
