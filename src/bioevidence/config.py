from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = Path(os.getenv("BIOEVIDENCE_DATA_DIR", "data"))
    log_level: str = os.getenv("BIOEVIDENCE_LOG_LEVEL", "INFO")
    pubmed_email: str = os.getenv("PUBMED_EMAIL", "")
    pubmed_tool_name: str = os.getenv("PUBMED_TOOL_NAME", "BioEvidence Copilot")


def load_settings() -> Settings:
    return Settings()
