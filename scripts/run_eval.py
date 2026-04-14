from __future__ import annotations

from pathlib import Path

from bioevidence.evaluation.runner import run_evaluation


def main() -> int:
    results = run_evaluation(Path("data/eval/dataset.jsonl"))
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
