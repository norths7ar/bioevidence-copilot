from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence
from urllib.error import URLError

from bioevidence.agent.workflow import run_agent_workflow
from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.presentation import build_agent_comparison_payload
from bioevidence.schemas.query import Query


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
        default=None,
        help="Optional local data directory for corpus and cache artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON agent report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text=args.query)
    settings = load_settings()
    try:
        result = run_agent_workflow(query, data_dir=args.data_dir, settings=settings)
    except (PubMedRequestError, URLError, OSError) as exc:
        logging.getLogger(__name__).warning("Agent workflow unavailable in the current environment: %s", exc)
        return 1

    payload = build_agent_comparison_payload(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
