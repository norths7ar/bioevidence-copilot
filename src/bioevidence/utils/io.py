from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load JSON content from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(item: Any, path: Path) -> None:
    """Write JSON content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(item, file, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[Any]:
    """Load all records from a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Yield records from a JSONL file one by one."""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def add_to_jsonl(items: Iterable[Any], path: Path) -> None:
    """Append records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_jsonl(items: Iterable[Any], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_text(path: Path) -> str:
    """Load a text or Markdown file."""
    with path.open("r", encoding="utf-8") as file:
        return file.read()


def load_text_lines(path: Path) -> list[str]:
    """Load a text or Markdown file as lines."""
    with path.open("r", encoding="utf-8") as file:
        return file.readlines()


def set_output_dir(path: Path) -> Path:
    """Create and return an output directory path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
