from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.setup_extraction_adapter import INSTALL_METADATA_FILE, install_adapter, verify_adapter_directory


def test_install_adapter_copies_verified_snapshot_and_is_idempotent(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot")
    output_dir = tmp_path / "models" / "adapter"
    calls = 0

    def provide_snapshot(repo_id: str, revision: str) -> Path:
        nonlocal calls
        calls += 1
        assert repo_id == "owner/adapter"
        assert revision == "abc123"
        return snapshot

    installed = install_adapter(
        output_dir,
        repo_id="owner/adapter",
        revision="abc123",
        snapshot_provider=provide_snapshot,
    )
    reused = install_adapter(
        output_dir,
        repo_id="owner/adapter",
        revision="abc123",
        snapshot_provider=provide_snapshot,
    )

    assert installed["status"] == "installed"
    assert reused["status"] == "already_installed"
    assert installed["verified_files"] == 2
    assert calls == 1
    assert json.loads((output_dir / INSTALL_METADATA_FILE).read_text(encoding="utf-8"))["revision"] == "abc123"


def test_verify_adapter_directory_rejects_modified_file(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot")
    (snapshot / "adapter_model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="mismatch"):
        verify_adapter_directory(snapshot)


def test_install_adapter_does_not_keep_failed_staging_directory(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot")
    (snapshot / "adapter_config.json").unlink()
    output_dir = tmp_path / "models" / "adapter"

    with pytest.raises(ValueError, match="missing file"):
        install_adapter(output_dir, snapshot_provider=lambda _repo, _revision: snapshot)

    assert not output_dir.exists()


def _write_snapshot(directory: Path) -> Path:
    directory.mkdir()
    files = {
        "adapter_config.json": b'{"base_model_name_or_path":"base/model"}',
        "adapter_model.safetensors": b"adapter weights",
    }
    for name, content in files.items():
        (directory / name).write_bytes(content)
    manifest = {
        "format": "peft_adapter_release_v1",
        "base_model": "base/model",
        "source_adapter": "ignored/local/path",
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()
        },
    }
    (directory / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory
