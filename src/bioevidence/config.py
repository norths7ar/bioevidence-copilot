from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _env_optional_path(name: str) -> Path | None:
    raw_value = os.getenv(name, "").strip()
    return Path(raw_value) if raw_value else None


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_optional_int(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    return int(raw_value) if raw_value else None


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    embedding_cache_dir: Path
    agent_api_key: str
    agent_base_url: str
    agent_max_iterations: int
    agent_max_output_tokens: int
    agent_min_relevance_score: float
    agent_min_unique_pmids: int
    agent_model: str
    agent_temperature: float
    log_level: str
    pubmed_email: str
    pubmed_tool_name: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dimensions: int | None
    embedding_batch_size: int
    extraction_adapter_path: Path | None = None
    extraction_api_key: str = ""
    extraction_backend: str = "legacy"
    extraction_base_url: str = ""
    extraction_max_output_tokens: int = 1024
    extraction_max_seq_length: int = 4096
    extraction_model: str = ""
    graph_enabled: bool = False
    graph_database: str = "neo4j"
    graph_max_expansion_queries: int = 3
    graph_max_paths: int = 20
    graph_password: str = ""
    graph_uri: str = "bolt://127.0.0.1:7687"
    graph_user: str = "neo4j"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=_env_path("DATA_DIR", "data/corpora/demo"),
        embedding_cache_dir=_env_path("EMBEDDING_CACHE_DIR", "data/cache"),
        agent_api_key=os.getenv("AGENT_API_KEY", ""),
        agent_base_url=os.getenv("AGENT_BASE_URL", ""),
        agent_max_iterations=_env_int("AGENT_MAX_ITERATIONS", "3"),
        agent_max_output_tokens=_env_int("AGENT_MAX_OUTPUT_TOKENS", "8192"),
        agent_min_relevance_score=_env_float("AGENT_MIN_RELEVANCE_SCORE", "0.6"),
        agent_min_unique_pmids=_env_int("AGENT_MIN_UNIQUE_PMIDS", "3"),
        agent_model=os.getenv("AGENT_MODEL", ""),
        agent_temperature=_env_float("AGENT_TEMPERATURE", "0.2"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        pubmed_email=os.getenv("PUBMED_EMAIL", ""),
        pubmed_tool_name=os.getenv("PUBMED_TOOL_NAME", "BioEvidence Copilot"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        embedding_dimensions=_env_optional_int("EMBEDDING_DIMENSIONS"),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", "10"),
        extraction_adapter_path=_env_optional_path("EXTRACTION_ADAPTER_PATH"),
        extraction_api_key=os.getenv("EXTRACTION_API_KEY", ""),
        extraction_backend=os.getenv("EXTRACTION_BACKEND", "legacy"),
        extraction_base_url=os.getenv("EXTRACTION_BASE_URL", ""),
        extraction_max_output_tokens=_env_int("EXTRACTION_MAX_OUTPUT_TOKENS", "1024"),
        extraction_max_seq_length=_env_int("EXTRACTION_MAX_SEQ_LENGTH", "4096"),
        extraction_model=os.getenv("EXTRACTION_MODEL", ""),
        graph_enabled=_env_bool("GRAPH_ENABLED"),
        graph_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        graph_max_expansion_queries=_env_int("GRAPH_MAX_EXPANSION_QUERIES", "3"),
        graph_max_paths=_env_int("GRAPH_MAX_PATHS", "20"),
        graph_password=os.getenv("NEO4J_PASSWORD", ""),
        graph_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        graph_user=os.getenv("NEO4J_USER", "neo4j"),
    )
