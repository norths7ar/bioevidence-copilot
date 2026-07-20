from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bioevidence.trace import TraceRecorder


@dataclass(frozen=True, slots=True)
class RunArtifactPaths:
    directory: Path
    log: Path
    report: Path
    trace: Path
    debug: Path


def create_run_artifact_paths(base_dir: Path, recorder: TraceRecorder) -> RunArtifactPaths:
    timestamp = recorder.started_at.strftime("%Y%m%dT%H%M%SZ")
    directory = base_dir / f"{timestamp}-{recorder.run_id[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return RunArtifactPaths(
        directory=directory,
        log=directory / "run.log",
        report=directory / "report.json",
        trace=directory / "trace.jsonl",
        debug=directory / "debug.json",
    )
