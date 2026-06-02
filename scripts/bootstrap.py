from __future__ import annotations

from pathlib import Path


def main() -> int:
    for relative_path in [
        Path("data/corpora/demo/processed"),
        Path("data/corpora/bioasq/processed"),
        Path("data/evaluations/demo"),
        Path("data/evaluations/bioasq"),
        Path("data/cache"),
        Path("app/pages"),
    ]:
        relative_path.mkdir(parents=True, exist_ok=True)
    print("Repository scaffold directories are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
