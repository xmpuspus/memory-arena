"""Application settings via pydantic-settings. All config from environment.

Env var names are plain (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) so we
align with vendor SDK conventions. Memory-arena-specific knobs that have
generic names (e.g. `HOST`, `PORT`, `DEBUG`, paths) accept BOTH the plain
name and a `MEM_ARENA_`-prefixed alias for backward compatibility with
existing `.env` files.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _alias(*names: str) -> AliasChoices:
    """Accept any of the given env var names (case-insensitive in env)."""
    return AliasChoices(*names)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — Anthropic (latest models)
    anthropic_api_key: str = Field(
        default="",
        validation_alias=_alias("ANTHROPIC_API_KEY", "MEM_ARENA_ANTHROPIC_API_KEY"),
    )
    generate_model: str = Field(
        default="claude-sonnet-4-6",
        validation_alias=_alias("GENERATE_MODEL", "MEM_ARENA_GENERATE_MODEL"),
    )
    fast_model: str = Field(
        default="claude-haiku-4-5-20251001",
        validation_alias=_alias("FAST_MODEL", "MEM_ARENA_FAST_MODEL"),
    )
    judge_model: str = Field(
        default="claude-opus-4-7",
        validation_alias=_alias("JUDGE_MODEL", "MEM_ARENA_JUDGE_MODEL"),
    )

    # LLM provider selection
    llm_provider: str = Field(
        default="anthropic",
        validation_alias=_alias("LLM_PROVIDER", "MEM_ARENA_LLM_PROVIDER"),
    )
    llm_api_key: str = Field(
        default="",
        validation_alias=_alias("LLM_API_KEY", "MEM_ARENA_LLM_API_KEY"),
    )

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI generation model names (when provider=openai)
    openai_generate_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o"

    # Ollama model names
    ollama_generate_model: str = "llama3.1:8b"
    ollama_fast_model: str = "llama3.1:8b"
    ollama_judge_model: str = "llama3.1:8b"

    # LLM — OpenAI (for embeddings)
    openai_api_key: str = Field(
        default="",
        validation_alias=_alias("OPENAI_API_KEY", "MEM_ARENA_OPENAI_API_KEY"),
    )

    # LLM — OpenRouter (optional; opens up open-model benchmarks: Llama 3.3,
    # Qwen 2.5, DeepSeek Chat, Gemini Flash, etc).
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=_alias("OPENROUTER_API_KEY", "MEM_ARENA_OPENROUTER_API_KEY"),
    )

    # OpenRouter model names (used when llm_provider="openrouter" OR when the
    # model string is a recognised OpenRouter slug like meta-llama/llama-3.3-70b-instruct).
    openrouter_generate_model: str = "meta-llama/llama-3.3-70b-instruct"
    openrouter_fast_model: str = "google/gemini-2.0-flash-001"
    openrouter_judge_model: str = "meta-llama/llama-3.3-70b-instruct"

    # Neo4j (graphiti, mem0g)
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        validation_alias=_alias("NEO4J_URI", "MEM_ARENA_NEO4J_URI"),
    )
    neo4j_user: str = Field(
        default="neo4j",
        validation_alias=_alias("NEO4J_USER", "MEM_ARENA_NEO4J_USER"),
    )
    neo4j_password: str = Field(
        default="",
        validation_alias=_alias("NEO4J_PASSWORD", "MEM_ARENA_NEO4J_PASSWORD"),
    )

    # Postgres + pgvector (Memori)
    postgres_host: str = Field(
        default="localhost",
        validation_alias=_alias("POSTGRES_HOST", "MEM_ARENA_POSTGRES_HOST"),
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias=_alias("POSTGRES_PORT", "MEM_ARENA_POSTGRES_PORT"),
    )
    postgres_user: str = Field(
        default="memarena",
        validation_alias=_alias("POSTGRES_USER", "MEM_ARENA_POSTGRES_USER"),
    )
    postgres_password: str = Field(
        default="memarena",
        validation_alias=_alias("POSTGRES_PASSWORD", "MEM_ARENA_POSTGRES_PASSWORD"),
    )
    postgres_database: str = Field(
        default="memarena",
        validation_alias=_alias("POSTGRES_DATABASE", "MEM_ARENA_POSTGRES_DATABASE"),
    )

    # Mem0
    mem0_api_key: str = Field(
        default="",
        validation_alias=_alias("MEM0_API_KEY", "MEM_ARENA_MEM0_API_KEY"),
    )
    mem0_collection_prefix: str = "mem0"

    # Zep / Graphiti
    zep_api_key: str = Field(
        default="",
        validation_alias=_alias("ZEP_API_KEY", "MEM_ARENA_ZEP_API_KEY"),
    )
    graphiti_group_prefix: str = "ma"

    # FalkorDB (graphiti_falkor) — Redis-based graph engine. Host port 6381 to
    # avoid colliding with other local redis containers commonly bound to
    # 6379/6380.
    falkordb_host: str = Field(
        default="localhost",
        validation_alias=_alias("FALKORDB_HOST", "MEM_ARENA_FALKORDB_HOST"),
    )
    falkordb_port: int = Field(
        default=6381,
        validation_alias=_alias("FALKORDB_PORT", "MEM_ARENA_FALKORDB_PORT"),
    )
    falkordb_password: str = Field(
        default="",
        validation_alias=_alias("FALKORDB_PASSWORD", "MEM_ARENA_FALKORDB_PASSWORD"),
    )

    # ChromaDB
    chroma_path: str = "./chroma_data"

    # Embeddings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Server. `host` is generic enough that we keep a MEM_ARENA_ alias so
    # users with an existing `.env` continue to work.
    host: str = Field(
        default="127.0.0.1",
        validation_alias=_alias("HOST", "MEM_ARENA_HOST"),
        description=(
            "Bind address for the dashboard. 127.0.0.1 is loopback-only; "
            "pass --host 0.0.0.0 for cluster scenarios."
        ),
    )
    port: int = Field(
        default=8000,
        validation_alias=_alias("PORT", "MEM_ARENA_PORT"),
    )
    debug: bool = Field(
        default=False,
        validation_alias=_alias("DEBUG", "MEM_ARENA_DEBUG"),
    )
    cors_origins: list[str] = []
    session_ttl_minutes: int = 30

    # Benchmark
    benchmark_temperature: float = 0.0
    benchmark_max_concurrent: int = 5
    benchmark_query_timeout_s: int = 120
    # Per-strategy wall-clock budget for the whole ingest+recall run. A strategy
    # hung in a non-LLM await (vendor socket, Neo4j/FalkorDB auth hang, Chroma
    # lock) is cancelled past this so one stuck strategy can't block the batch
    # forever. Generous by default; raise it for large corpora.
    benchmark_strategy_timeout_s: int = 3600
    benchmark_max_retries: int = 2
    benchmark_cost_cap_usd: float = 0.0
    benchmark_enable_ragas: bool = False

    # Memory-strategy specific knobs. Names like `full_context_token_budget`
    # are arena-specific; we still keep the legacy MEM_ARENA_ alias so old
    # `.env` files continue to work.
    full_context_token_budget: int = Field(
        default=150000,
        validation_alias=_alias("FULL_CONTEXT_TOKEN_BUDGET", "MEM_ARENA_FULL_CONTEXT_TOKEN_BUDGET"),
    )
    recency_window_n: int = Field(
        default=20,
        validation_alias=_alias("RECENCY_WINDOW_N", "MEM_ARENA_RECENCY_WINDOW_N"),
    )
    recall_default_top_k: int = Field(
        default=10,
        validation_alias=_alias("RECALL_DEFAULT_TOP_K", "MEM_ARENA_RECALL_DEFAULT_TOP_K"),
    )

    # Quantum / quantum-inspired rerankers (qiss, sqr). Both coarse-retrieve
    # top_k * fanout candidates from naive_vector's Chroma index, then rerank.
    qiss_fanout: int = Field(
        default=4,
        validation_alias=_alias("QISS_FANOUT", "MEM_ARENA_QISS_FANOUT"),
    )
    # Multi-query superposition fusion (interference cross-terms). Off by default
    # so QISS's single-query fidelity stays cos^2 of naive_vector's cosine.
    qiss_decompose: bool = Field(
        default=False,
        validation_alias=_alias("QISS_DECOMPOSE", "MEM_ARENA_QISS_DECOMPOSE"),
    )
    sqr_fanout: int = Field(
        default=4,
        validation_alias=_alias("SQR_FANOUT", "MEM_ARENA_SQR_FANOUT"),
    )
    # Qubits for the SWAP test register; 2^n_qubits = amplitude-encoded dims.
    sqr_n_qubits: int = Field(
        default=4,
        validation_alias=_alias("SQR_N_QUBITS", "MEM_ARENA_SQR_N_QUBITS"),
    )
    # Measurement shots. 0 = exact statevector (the benchmark default); >0 picks
    # the noisy sampled estimator for the accuracy-vs-speed curve.
    sqr_shots: int = Field(
        default=0,
        validation_alias=_alias("SQR_SHOTS", "MEM_ARENA_SQR_SHOTS"),
    )

    # Paths. Generic names — keep MEM_ARENA_ alias so existing setups keep
    # working (paths.py also reads MEM_ARENA_DATASETS_PATH / RESULTS_PATH
    # directly via os.environ).
    datasets_path: str = Field(
        default="./datasets",
        validation_alias=_alias("DATASETS_PATH", "MEM_ARENA_DATASETS_PATH"),
    )
    results_path: str = Field(
        default="./results",
        validation_alias=_alias("RESULTS_PATH", "MEM_ARENA_RESULTS_PATH"),
    )


settings = Settings()
