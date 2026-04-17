from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    embedding_cache_dir: Path
    log_level: str
    pubmed_email: str
    pubmed_tool_name: str
    qwen_api_key: str
    qwen_base_url: str
    qwen_embedding_model: str
    qwen_embedding_dimensions: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=_env_path("BIOEVIDENCE_DATA_DIR", "data"),
        embedding_cache_dir=_env_path("BIOEVIDENCE_EMBEDDING_CACHE_DIR", "data/cache"),
        log_level=os.getenv("BIOEVIDENCE_LOG_LEVEL", "INFO"),
        pubmed_email=os.getenv("PUBMED_EMAIL", ""),
        pubmed_tool_name=os.getenv("PUBMED_TOOL_NAME", "BioEvidence Copilot"),
        qwen_api_key=os.getenv("QWEN_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", ""),
        qwen_base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        qwen_embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4"),
        qwen_embedding_dimensions=_env_int("QWEN_EMBEDDING_DIMENSIONS", "1024"),
    )
