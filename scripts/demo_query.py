from __future__ import annotations

import logging
from urllib.error import URLError
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioevidence.schemas.answer import AnswerBundle
from bioevidence.agent.workflow import run_workflow
from bioevidence.schemas.query import Query


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text="What evidence exists for a sample question?")
    try:
        answer = run_workflow(query)
    except (URLError, OSError) as exc:
        logging.getLogger(__name__).warning("Offline demo mode: %s", exc)
        answer = AnswerBundle(
            answer_text="PubMed fetch is unavailable in the current environment. The scaffold is ready, but live ingestion is disabled.",
            citations=(),
            evidence_records=(),
            rewritten_query=query.text,
        )
    print(answer.answer_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
