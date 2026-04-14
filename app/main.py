from __future__ import annotations

import logging
import json
from urllib.error import URLError

from bioevidence.agent.workflow import run_workflow
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.query import Query


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text="What evidence exists for a sample biomedical question?")
    try:
        answer = run_workflow(query)
    except (URLError, OSError) as exc:
        logging.getLogger(__name__).warning("Falling back to offline demo mode: %s", exc)
        answer = AnswerBundle(
            answer_text="PubMed fetch is unavailable in the current environment. The scaffold is ready, but live ingestion is disabled.",
            citations=(),
            evidence_records=(),
            rewritten_query=query.text,
        )

    payload = {
        "query": query.text,
        "answer": answer.answer_text,
        "citations": list(answer.citations),
        "evidence_count": len(answer.evidence_records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
