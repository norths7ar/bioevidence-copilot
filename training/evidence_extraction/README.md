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
