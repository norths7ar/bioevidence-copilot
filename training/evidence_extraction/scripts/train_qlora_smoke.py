from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"


def main() -> int:
    args = _parse_args()
    dataset_summary = _validate_datasets(args.train_file, args.dev_file)
    if args.dry_run:
        print(json.dumps({"config": _config_payload(args), "dataset": dataset_summary}, indent=2))
        return 0
    return _run_training(args, dataset_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small QLoRA evidence-extraction training check.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/pilot_sft/train.jsonl"),
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/pilot_sft/dev.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/qwen3_4b_qlora_smoke"),
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-reload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    return args


def _run_training(args: argparse.Namespace, dataset_summary: dict[str, Any]) -> int:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from trl import SFTConfig, SFTTrainer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_started = perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    load_seconds = perf_counter() - load_started

    train_dataset = _load_rendered_dataset(args.train_file, tokenizer)
    dev_dataset = _load_rendered_dataset(args.dev_file, tokenizer)
    bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=SFTConfig(
            output_dir=str(args.output_dir / "checkpoints"),
            dataset_text_field="text",
            max_length=args.max_seq_length,
            packing=False,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_steps=1,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            bf16=bf16,
            fp16=not bf16,
            logging_steps=1,
            logging_first_step=True,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
        ),
    )
    trainer = train_on_responses_only(trainer)

    torch.cuda.reset_peak_memory_stats()
    pre_eval = trainer.evaluate(metric_key_prefix="pre")
    train_started = perf_counter()
    train_result = trainer.train()
    train_seconds = perf_counter() - train_started
    post_eval = trainer.evaluate(metric_key_prefix="post")
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_files = sorted(path.name for path in adapter_dir.iterdir() if path.is_file())
    required_files = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_files.issubset(adapter_files):
        raise RuntimeError(f"adapter save is incomplete: expected {sorted(required_files)}")

    reload_verified = False
    if args.verify_reload:
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_model, _ = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            full_finetuning=False,
        )
        reload_verified = bool(getattr(reloaded_model, "peft_config", None))
        del reloaded_model
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": _config_payload(args),
        "dataset": dataset_summary,
        "environment": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "trl": importlib.metadata.version("trl"),
            "unsloth": importlib.metadata.version("unsloth"),
        },
        "load_seconds": round(load_seconds, 3),
        "train_seconds": round(train_seconds, 3),
        "peak_vram_gib": round(peak_vram_gib, 3),
        "pre_eval_loss": pre_eval.get("pre_loss"),
        "post_eval_loss": post_eval.get("post_loss"),
        "train_metrics": train_result.metrics,
        "adapter_dir": adapter_dir.as_posix(),
        "adapter_files": adapter_files,
        "reload_verified": reload_verified,
    }
    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0


def _load_rendered_dataset(path: Path, tokenizer: Any) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_list(_load_jsonl(path))

    def render_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
        return {"text": texts}

    return dataset.map(render_batch, batched=True, desc=f"Render {path.name}")


def _validate_datasets(train_file: Path, dev_file: Path) -> dict[str, Any]:
    train_records = _load_jsonl(train_file)
    dev_records = _load_jsonl(dev_file)
    train_pmids = _validate_records(train_records, expected_split="train")
    dev_pmids = _validate_records(dev_records, expected_split="dev")
    overlap = train_pmids & dev_pmids
    if overlap:
        raise ValueError(f"PMID leakage between train and dev: {sorted(overlap)}")
    return {
        "train_rows": len(train_records),
        "dev_rows": len(dev_records),
        "train_unique_pmids": len(train_pmids),
        "dev_unique_pmids": len(dev_pmids),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(payload)
    if not records:
        raise ValueError(f"{path}: no training records")
    return records


def _validate_records(records: list[dict[str, Any]], *, expected_split: str) -> set[str]:
    pmids: set[str] = set()
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        metadata = record.get("metadata")
        if not isinstance(messages, list) or [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError(f"{expected_split} row {index}: expected system/user/assistant messages")
        if not isinstance(json.loads(messages[-1]["content"]), dict):
            raise ValueError(f"{expected_split} row {index}: assistant target must be a JSON object")
        if not isinstance(metadata, dict) or metadata.get("split") != expected_split:
            raise ValueError(f"{expected_split} row {index}: split metadata mismatch")
        pmid = metadata.get("pmid")
        if not isinstance(pmid, str) or not pmid:
            raise ValueError(f"{expected_split} row {index}: PMID is required")
        pmids.add(pmid)
    return pmids


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "response_only_loss": True,
        "verify_reload": args.verify_reload,
    }


if __name__ == "__main__":
    raise SystemExit(main())
