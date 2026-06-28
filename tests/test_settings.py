"""Tests for Settings — defaults and env var overrides.

The MEM_ARENA_ prefix has been dropped (see memory_arena/settings.py); plain
env var names are now canonical. Some fields keep a MEM_ARENA_-prefixed
alias for backward compatibility, exercised at the bottom of the file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_settings_default_generate_model():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.generate_model == "claude-sonnet-4-6"


def test_settings_default_fast_model():
    from memory_arena.settings import Settings

    s = Settings()
    assert "haiku" in s.fast_model.lower()


def test_settings_default_neo4j_uri():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.neo4j_uri == "bolt://localhost:7687"


def test_settings_default_neo4j_user():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.neo4j_user == "neo4j"


def test_settings_default_chroma_path():
    from memory_arena.settings import Settings

    s = Settings()
    assert "chroma" in s.chroma_path.lower()


def test_settings_default_port():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.port == 8000


def test_settings_default_host(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("MEM_ARENA_HOST", raising=False)
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.host == "127.0.0.1"


def test_settings_default_debug_false():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.debug is False


def test_settings_default_datasets_path():
    from memory_arena.settings import Settings

    s = Settings()
    assert "datasets" in s.datasets_path


def test_settings_default_results_path():
    from memory_arena.settings import Settings

    s = Settings()
    assert "results" in s.results_path


def test_settings_default_benchmark_max_concurrent():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.benchmark_max_concurrent == 5


def test_settings_default_benchmark_max_retries():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.benchmark_max_retries == 2


def test_settings_default_benchmark_temperature():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.benchmark_temperature == 0.0


def test_settings_default_embedding_model():
    from memory_arena.settings import Settings

    s = Settings()
    assert "text-embedding" in s.embedding_model


def test_settings_default_anthropic_api_key_empty(monkeypatch):
    monkeypatch.delenv("MEM_ARENA_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.anthropic_api_key == ""


def test_settings_default_query_timeout():
    from memory_arena.settings import Settings

    s = Settings()
    assert s.benchmark_query_timeout_s == 120


# ---------------------------------------------------------------------------
# Custom env vars via monkeypatch — plain (un-prefixed) names
# ---------------------------------------------------------------------------


def test_settings_anthropic_api_key_via_plain_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.anthropic_api_key == "sk-ant-test"


def test_settings_openai_api_key_via_plain_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.openai_api_key == "sk-openai-test"


def test_settings_neo4j_password_via_plain_env(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j-secret")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.neo4j_password == "neo4j-secret"


def test_settings_port_via_env(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.port == 9999


def test_settings_debug_true_via_env(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.debug is True


def test_settings_neo4j_uri_via_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://remote:7687")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.neo4j_uri == "bolt://remote:7687"


def test_settings_datasets_path_via_env(monkeypatch):
    monkeypatch.setenv("DATASETS_PATH", "/tmp/my-datasets")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.datasets_path == "/tmp/my-datasets"


def test_settings_extra_fields_ignored(monkeypatch):
    monkeypatch.setenv("MEM_ARENA_NONEXISTENT_FIELD", "value")
    from memory_arena.settings import Settings

    s = Settings()
    assert s is not None


def test_settings_host_via_env(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.host == "0.0.0.0"


# ---------------------------------------------------------------------------
# Backward-compat: MEM_ARENA_ aliases for fields with generic names
# ---------------------------------------------------------------------------


def test_settings_legacy_mem_arena_host_alias(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.setenv("MEM_ARENA_HOST", "0.0.0.0")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.host == "0.0.0.0"


def test_settings_legacy_mem_arena_port_alias(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("MEM_ARENA_PORT", "7777")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.port == 7777


def test_settings_legacy_mem_arena_datasets_path_alias(monkeypatch):
    monkeypatch.delenv("DATASETS_PATH", raising=False)
    monkeypatch.setenv("MEM_ARENA_DATASETS_PATH", "/tmp/legacy-datasets")
    from memory_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.datasets_path == "/tmp/legacy-datasets"
