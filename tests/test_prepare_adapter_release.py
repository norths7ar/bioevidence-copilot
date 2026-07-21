from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.evidence_extraction.scripts.prepare_adapter_release import (
    DEFAULT_BASE_MODEL,
    prepare_adapter_release,
)


def test_prepare_adapter_release_rewrites_local_base_path_and_adds_card(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "E:/local/snapshot", "peft_type": "LORA"}),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("# Model card\n", encoding="utf-8")
    output_dir = tmp_path / "release"

    manifest = prepare_adapter_release(adapter_dir, output_dir, model_card=model_card)

    config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == DEFAULT_BASE_MODEL
    assert (output_dir / "README.md").read_text(encoding="utf-8") == "# Model card\n"
    assert manifest["files"]["adapter_model.safetensors"]["bytes"] == 7
    assert json.loads((output_dir / "release_manifest.json").read_text(encoding="utf-8")) == manifest


def test_prepare_adapter_release_refuses_nonempty_destination(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("card", encoding="utf-8")
    output_dir = tmp_path / "release"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        prepare_adapter_release(adapter_dir, output_dir, model_card=model_card)

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
