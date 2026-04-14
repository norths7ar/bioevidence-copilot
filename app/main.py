from __future__ import annotations

import logging
import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioevidence.agent.workflow import run_workflow
from bioevidence.schemas.query import Query


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text="What evidence exists for a sample biomedical question?")
    answer = run_workflow(query)

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
