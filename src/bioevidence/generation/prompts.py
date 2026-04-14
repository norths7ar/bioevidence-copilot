from __future__ import annotations

from bioevidence.schemas.query import Query


def build_answer_prompt(query: Query) -> str:
    return f"Answer the biomedical question using retrieved evidence: {query.text}"
