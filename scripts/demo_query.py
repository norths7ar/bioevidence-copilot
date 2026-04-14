from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioevidence.agent.workflow import run_workflow
from bioevidence.schemas.query import Query


def main() -> int:
    answer = run_workflow(Query(text="What evidence exists for a sample question?"))
    print(answer.answer_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
