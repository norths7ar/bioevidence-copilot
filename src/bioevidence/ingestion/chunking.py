from __future__ import annotations


def chunk_abstract(text: str, chunk_size: int = 512) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
