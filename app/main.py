from __future__ import annotations

import logging
import json
from urllib.error import URLError

from bioevidence.agent.workflow import run_rag_pipeline
from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.query import Query


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text="What evidence exists for a sample biomedical question?")
    settings = load_settings()
    try:
        result = run_rag_pipeline(query, settings=settings)
    except (PubMedRequestError, URLError, OSError) as exc:
        logging.getLogger(__name__).warning("Falling back to offline demo mode: %s", exc)
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
            "answer": answer.answer_text,
            "citations": list(answer.citations),
            "evidence_count": len(answer.evidence_records),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    payload = {
        "query": query.text,
        "rewritten_query": result.answer.rewritten_query,
        "retrieval_source": result.source,
        "retrieved_papers": [
            {
                "pmid": candidate.document.pmid,
                "title": candidate.document.title,
                "journal": candidate.document.journal,
                "year": candidate.document.year,
                "score": round(candidate.score, 4),
                "rank": candidate.rank,
            }
            for candidate in result.retrieved_candidates[: query.top_k]
        ],
        "answer": result.answer.answer_text,
        "citations": list(result.answer.citations),
        "evidence_count": len(result.answer.evidence_records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
