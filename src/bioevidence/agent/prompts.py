from __future__ import annotations

import json
from collections.abc import Sequence

from bioevidence.agent.state import AgentState
from bioevidence.schemas.evidence import EvidenceRecord


def build_planning_messages(state: AgentState, branch_count: int = 2) -> list[dict[str, str]]:
    evidence_summary = _summarize_evidence(state.top_evidence_records(limit=5))
    branch_list = "\n".join(f"- {query}" for query in state.branch_queries) or "- (none yet)"
    user_prompt = (
        "You are planning follow-up biomedical literature searches.\n"
        "Return JSON only with keys: branch_queries (array of strings) and rationale (string).\n"
        f"Original query: {state.query.text}\n"
        f"Existing branch queries:\n{branch_list}\n"
        f"Observed evidence summary:\n{evidence_summary}\n"
        f"Generate up to {branch_count} new branch queries that are meaningfully different and focused on evidence gaps."
    )
    return [
        {
            "role": "system",
            "content": "You are a careful biomedical search planner. Output valid JSON only.",
        },
        {"role": "user", "content": user_prompt},
    ]


def build_synthesis_messages(state: AgentState, baseline_answer: str) -> list[dict[str, str]]:
    evidence_payload = json.dumps(
        [
            {
                "pmid": record.pmid,
                "title": record.title,
                "year": record.year,
                "journal": record.journal,
                "entities": list(record.entities),
                "summary": record.summary,
                "relevance_score": record.relevance_score,
            }
            for record in state.top_evidence_records()
        ],
        indent=2,
        sort_keys=True,
    )
    branch_queries = "\n".join(f"- {query}" for query in state.branch_queries) or "- (none)"
    user_prompt = (
        "Write the final biomedical evidence answer using only the evidence provided.\n"
        "Return JSON only with keys: answer_text (string), citations (array of PMIDs), rewritten_query (string).\n"
        f"Original query: {state.query.text}\n"
        f"Branch queries:\n{branch_queries}\n"
        f"Baseline answer for comparison:\n{baseline_answer}\n"
        f"Evidence rows:\n{evidence_payload}\n"
        "Use concise language, mention uncertainty when evidence is sparse, and cite supporting PMIDs in the citations array."
    )
    return [
        {
            "role": "system",
            "content": "You are a biomedical evidence synthesizer. Output valid JSON only.",
        },
        {"role": "user", "content": user_prompt},
    ]


def _summarize_evidence(records: Sequence[EvidenceRecord]) -> str:
    if not records:
        return "(no evidence yet)"
    lines: list[str] = []
    for record in records:
        year_text = str(record.year) if record.year is not None else "n.d."
        lines.append(
            f"{record.pmid}: {record.title} ({year_text}) | score={record.relevance_score:.3f} | {record.summary}"
        )
    return "\n".join(lines)
