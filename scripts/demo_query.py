from __future__ import annotations

import logging
from urllib.error import URLError

from bioevidence.schemas.answer import AnswerBundle
from bioevidence.agent.workflow import run_rag_pipeline
from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import PubMedRequestError
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
        print(answer.answer_text)
        return 0
    print(result.answer.answer_text)
    for candidate in result.retrieved_candidates[: query.top_k]:
        print(f"{candidate.rank}. {candidate.document.pmid} {candidate.document.title} ({candidate.score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
