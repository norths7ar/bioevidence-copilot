from __future__ import annotations

import json
import logging
from urllib.error import URLError

from bioevidence.agent.workflow import run_rag_pipeline
from bioevidence.config import load_settings
from bioevidence.presentation import build_demo_payload, render_demo_output
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.query import Query


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text="What evidence exists for a sample question?")
    settings = load_settings()
    try:
        result = run_rag_pipeline(query, settings=settings)
    except (PubMedRequestError, URLError, OSError) as exc:
        logging.getLogger(__name__).warning("Offline demo mode: %s", exc)
        answer = AnswerBundle(
            answer_text="PubMed fetch is unavailable in the current environment. The scaffold is ready, but live ingestion is disabled.",
            citations=(),
            evidence_records=(),
            rewritten_query=query.text,
        )
        payload = {
            "query": query.text,
            "rewritten_query": answer.rewritten_query,
            "retrieval_source": "offline_fallback",
            "retrieved_papers": [],
            "evidence_table": [],
            "answer": answer.answer_text,
            "citations": list(answer.citations),
            "evidence_count": len(answer.evidence_records),
        }
        print("Evidence table: (none)")
        print()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = build_demo_payload(query, result)
    print(render_demo_output(result))
    print()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
