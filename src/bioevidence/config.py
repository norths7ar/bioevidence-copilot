from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


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
        data_dir=_env_path("BIOEVIDENCE_DATA_DIR", "data/corpora/demo"),
        embedding_cache_dir=_env_path("BIOEVIDENCE_EMBEDDING_CACHE_DIR", "data/cache"),
        agent_api_key=os.getenv("BIOEVIDENCE_AGENT_API_KEY", ""),
        agent_base_url=os.getenv("BIOEVIDENCE_AGENT_BASE_URL", ""),
        agent_max_iterations=_env_int("BIOEVIDENCE_AGENT_MAX_ITERATIONS", "3"),
        agent_max_output_tokens=_env_int("BIOEVIDENCE_AGENT_MAX_OUTPUT_TOKENS", "800"),
        agent_min_relevance_score=_env_float("BIOEVIDENCE_AGENT_MIN_RELEVANCE_SCORE", "0.6"),
        agent_min_unique_pmids=_env_int("BIOEVIDENCE_AGENT_MIN_UNIQUE_PMIDS", "3"),
        agent_model=os.getenv("BIOEVIDENCE_AGENT_MODEL", ""),
        agent_temperature=_env_float("BIOEVIDENCE_AGENT_TEMPERATURE", "0.2"),
        log_level=os.getenv("BIOEVIDENCE_LOG_LEVEL", "INFO"),
        pubmed_email=os.getenv("PUBMED_EMAIL", ""),
        pubmed_tool_name=os.getenv("PUBMED_TOOL_NAME", "BioEvidence Copilot"),
        embedding_api_key=os.getenv("BIOEVIDENCE_EMBEDDING_API_KEY", ""),
        embedding_base_url=os.getenv("BIOEVIDENCE_EMBEDDING_BASE_URL", ""),
        embedding_model=os.getenv("BIOEVIDENCE_EMBEDDING_MODEL", ""),
        embedding_dimensions=_env_optional_int("BIOEVIDENCE_EMBEDDING_DIMENSIONS"),
        embedding_batch_size=_env_int("BIOEVIDENCE_EMBEDDING_BATCH_SIZE", "10"),
        graph_enabled=_env_bool("BIOEVIDENCE_GRAPH_ENABLED"),
        graph_database=os.getenv("BIOEVIDENCE_GRAPH_DATABASE", "neo4j"),
        graph_max_expansion_queries=_env_int("BIOEVIDENCE_GRAPH_MAX_EXPANSION_QUERIES", "3"),
        graph_max_paths=_env_int("BIOEVIDENCE_GRAPH_MAX_PATHS", "20"),
        graph_password=os.getenv("BIOEVIDENCE_GRAPH_PASSWORD", ""),
        graph_uri=os.getenv("BIOEVIDENCE_GRAPH_URI", "bolt://127.0.0.1:7687"),
        graph_user=os.getenv("BIOEVIDENCE_GRAPH_USER", "neo4j"),
    )
