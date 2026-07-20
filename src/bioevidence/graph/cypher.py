from __future__ import annotations


def quote_identifier(value: str) -> str:
    return f"`{value.replace('`', '``')}`"
