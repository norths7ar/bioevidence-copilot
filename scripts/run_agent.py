from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
from typing import Sequence
from urllib.error import URLError

from bioevidence.workflows import run_agent_workflow
from bioevidence.artifacts import create_run_artifact_paths
from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.presentation import (
    build_agent_comparison_payload,
    build_agent_report_payload,
    render_agent_run_summary,
)
from bioevidence.schemas.query import Query
from bioevidence.trace import TraceRecorder
from bioevidence.utils.io import save_json, save_jsonl
from bioevidence.utils.logging_config import close_log_file, configure_logging


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BioEvidence agentic workflow.")
    parser.add_argument(
        "--query",
        type=str,
        default="What evidence exists for a sample biomedical question?",
        help="Biomedical question to run through the agentic workflow.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/corpora/demo"),
        help="Local corpus data directory. The workflow reads processed/*.documents.jsonl under this path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the compact JSON report.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional base directory for a run folder containing log, report, and trace artifacts.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also write the full internal payload; requires --artifacts-dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug and args.artifacts_dir is None:
        parser.error("--debug requires --artifacts-dir")

    settings = load_settings()
    recorder = TraceRecorder()
    artifact_paths = create_run_artifact_paths(args.artifacts_dir, recorder) if args.artifacts_dir is not None else None
    configure_logging(settings.log_level, log_file=artifact_paths.log if artifact_paths else None)
    query = Query(text=args.query)
    try:
        result = run_agent_workflow(
            query,
            data_dir=args.data_dir,
            settings=settings,
            trace_recorder=recorder,
        )
    except (PubMedRequestError, URLError, OSError) as exc:
        LOGGER.warning("agent_workflow_unavailable reason=%s", type(exc).__name__)
        if artifact_paths is not None:
            save_jsonl(recorder.events(), artifact_paths.trace)
            close_log_file(artifact_paths.log)
        return 1

    if result.run_id is None:
        result = replace(result, run_id=recorder.run_id, trace_events=recorder.events())
    report = build_agent_report_payload(result)
    print(render_agent_run_summary(result))

    if args.output is not None:
        save_json(report, args.output)
        print(f"Report written to {args.output}")

    if artifact_paths is not None:
        save_json(report, artifact_paths.report)
        save_jsonl(result.trace_events, artifact_paths.trace)
        if args.debug:
            save_json(build_agent_comparison_payload(result), artifact_paths.debug)
        print(f"Run artifacts written to {artifact_paths.directory}")
        close_log_file(artifact_paths.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
