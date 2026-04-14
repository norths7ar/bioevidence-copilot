from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioevidence.evaluation.runner import run_evaluation


def main() -> int:
    results = run_evaluation(Path("data/eval/dataset.jsonl"))
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
