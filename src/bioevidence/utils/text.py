from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def slugify_text(text: str, default: str = "query") -> str:
    """Turn arbitrary text into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_whitespace(text).lower()).strip("-")
    return slug or default
