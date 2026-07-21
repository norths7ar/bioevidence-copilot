from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copy2
from typing import Any, Sequence


DEFAULT_BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
OPTIONAL_RUNTIME_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = prepare_adapter_release(
        args.adapter_dir,
        args.output_dir,
        model_card=args.model_card,
        base_model=args.base_model,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Release directory: {args.output_dir}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a portable PEFT adapter directory for publication.")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/qwen3_4b_qlora_v2/adapter"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/releases/bioevidence-qwen3-4b-extraction-lora-v1"),
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=Path("training/evidence_extraction/MODEL_CARD.md"),
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    return parser.parse_args(argv)


def prepare_adapter_release(
    adapter_dir: Path,
    output_dir: Path,
    *,
    model_card: Path,
    base_model: str = DEFAULT_BASE_MODEL,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"adapter is incomplete; missing: {', '.join(missing)}")
    if not model_card.is_file():
        raise FileNotFoundError(model_card)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"release directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError("adapter_config.json must contain a JSON object")
    source_config["base_model_name_or_path"] = base_model
    _write_json(output_dir / "adapter_config.json", source_config)
    copy2(adapter_dir / "adapter_model.safetensors", output_dir / "adapter_model.safetensors")
    for name in OPTIONAL_RUNTIME_FILES:
        source = adapter_dir / name
        if source.is_file():
            copy2(source, output_dir / name)
    copy2(model_card, output_dir / "README.md")

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "format": "peft_adapter_release_v1",
        "base_model": base_model,
        "source_adapter": adapter_dir.as_posix(),
        "files": files,
    }
    _write_json(output_dir / "release_manifest.json", manifest)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
