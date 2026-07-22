from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Any


DEFAULT_REPO_ID = "n0rths7ar/bioevidence-qwen3-4b-extraction-lora-v2"
DEFAULT_REVISION = "20ae7837207fcb697ac99d71961e99d0aebcb4ab"
DEFAULT_OUTPUT_DIR = Path("artifacts/models/bioevidence-qwen3-4b-extraction-lora-v2")
INSTALL_METADATA_FILE = "bioevidence_install.json"
RELEASE_MANIFEST_FILE = "release_manifest.json"

SnapshotProvider = Callable[[str, str], Path]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = install_adapter(
        args.output_dir,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'$env:EXTRACTION_ADAPTER_PATH="{args.output_dir.as_posix()}"')
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned BioEvidence extraction adapter from Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def install_adapter(
    output_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    snapshot_provider: SnapshotProvider | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        result = verify_adapter_directory(output_dir)
        metadata = _load_json(output_dir / INSTALL_METADATA_FILE)
        if metadata.get("repo_id") != repo_id or metadata.get("revision") != revision:
            raise ValueError(f"existing adapter provenance does not match requested revision: {output_dir}")
        return {"status": "already_installed", "output_dir": output_dir.as_posix(), **result}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    provider = snapshot_provider or _download_snapshot
    snapshot_dir = provider(repo_id, revision)
    verified = verify_adapter_directory(snapshot_dir)

    with TemporaryDirectory(prefix=".bioevidence-adapter-", dir=output_dir.parent) as temporary:
        staging_dir = Path(temporary)
        manifest = _load_release_manifest(snapshot_dir)
        for name in manifest["files"]:
            copy2(snapshot_dir / name, staging_dir / name)
        copy2(snapshot_dir / RELEASE_MANIFEST_FILE, staging_dir / RELEASE_MANIFEST_FILE)
        _write_json(
            staging_dir / INSTALL_METADATA_FILE,
            {
                "format": "bioevidence_adapter_install_v1",
                "repo_id": repo_id,
                "revision": revision,
                "verified_files": verified["verified_files"],
            },
        )
        staging_dir.rename(output_dir)

    return {"status": "installed", "output_dir": output_dir.as_posix(), **verified}


def verify_adapter_directory(directory: Path) -> dict[str, Any]:
    manifest = _load_release_manifest(directory)
    failures: list[str] = []
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            failures.append(f"missing file: {name}")
            continue
        if path.stat().st_size != expected["bytes"]:
            failures.append(f"size mismatch: {name}")
            continue
        if _sha256(path) != expected["sha256"]:
            failures.append(f"SHA-256 mismatch: {name}")
    if failures:
        raise ValueError("adapter verification failed: " + "; ".join(failures))
    return {
        "base_model": manifest["base_model"],
        "verified_files": len(manifest["files"]),
    }


def _download_snapshot(repo_id: str, revision: str) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Adapter setup requires huggingface_hub. Run this command from the bioevidence-training environment."
        ) from exc
    return Path(snapshot_download(repo_id=repo_id, revision=revision))


def _load_release_manifest(directory: Path) -> dict[str, Any]:
    manifest = _load_json(directory / RELEASE_MANIFEST_FILE)
    if manifest.get("format") != "peft_adapter_release_v1":
        raise ValueError(f"unsupported release manifest: {directory / RELEASE_MANIFEST_FILE}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("release manifest must contain files")
    for name, metadata in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe release filename: {name!r}")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("bytes"), int):
            raise ValueError(f"invalid release metadata: {name}")
        digest = metadata.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid release digest: {name}")
    if not isinstance(manifest.get("base_model"), str):
        raise ValueError("release manifest must contain base_model")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
