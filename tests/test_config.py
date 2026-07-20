from __future__ import annotations

from pathlib import Path

from bioevidence import config


def test_load_settings_uses_grouped_environment_names(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    values = {
        "DATA_DIR": "custom/data",
        "LOG_LEVEL": "DEBUG",
        "EMBEDDING_CACHE_DIR": "custom/cache",
        "EMBEDDING_MODEL": "embedding-model",
        "AGENT_MODEL": "agent-model",
        "GRAPH_ENABLED": "true",
        "NEO4J_URI": "bolt://neo4j:7687",
        "NEO4J_USER": "graph-user",
        "NEO4J_PASSWORD": "graph-password",
        "NEO4J_DATABASE": "graph-database",
        "GRAPH_MAX_PATHS": "12",
        "GRAPH_MAX_EXPANSION_QUERIES": "4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = config.load_settings()

    assert settings.data_dir == Path("custom/data")
    assert settings.log_level == "DEBUG"
    assert settings.embedding_cache_dir == Path("custom/cache")
    assert settings.embedding_model == "embedding-model"
    assert settings.agent_model == "agent-model"
    assert settings.graph_enabled is True
    assert settings.graph_uri == "bolt://neo4j:7687"
    assert settings.graph_user == "graph-user"
    assert settings.graph_password == "graph-password"
    assert settings.graph_database == "graph-database"
    assert settings.graph_max_paths == 12
    assert settings.graph_max_expansion_queries == 4


def test_load_settings_does_not_read_legacy_project_prefix(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("BIOEVIDENCE_AGENT_MODEL", "legacy-model")

    settings = config.load_settings()

    assert settings.agent_model == ""
