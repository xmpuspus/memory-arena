# Vendor SDK Pins

Memory Arena pins every vendor SDK explicitly. If a pinned version breaks at install time, log the failure here with the breaking error and bump exactly one minor version. Do not silently float.

## Current Pins

| Package         | Pin                | Reason                                                        |
| --------------- | ------------------ | ------------------------------------------------------------- |
| `mem0ai`        | `==0.1.114`        | Vendor default hybrid memory. Verified install on 2026-04-29. |
| `graphiti-core` | `==0.13.0`         | Zep OSS temporal graph memory.                                |
| `cognee`        | `>=1.0,<2.0`       | Knowledge-graph memory (`add` → `cognify` → `search`).        |
| `langmem`       | `>=0.0.10`         | LangChain memory store.                                       |
| `memori`        | `>=3.0.0,<4.0.0`   | SQL-native fact store with augmentation pipeline.             |
| `chromadb`      | `==0.5.23`         | Local vector store used by `naive_vector` and `mem0`.         |
| `neo4j`         | `==5.27.0`         | Graph DB driver for `mem0g` and `graphiti`.                   |

## Replication of Strategy Backends

| Strategy   | Required services                                |
| ---------- | ------------------------------------------------ |
| `mem0`     | Chroma (local) + OpenAI embeddings + Anthropic   |
| `mem0g`    | adds Neo4j (port 7687)                           |
| `graphiti` | Neo4j (shared with mem0g)                        |
| `cognee`   | networkx (default) or Neo4j                      |
| `langmem`  | LangGraph InMemoryStore + OpenAI embeddings      |
| `memori`   | Postgres + pgvector                              |

## Known Breakages and Workarounds

| Date       | SDK / version              | Symptom                                                                                  | Workaround                                                                  |
| ---------- | -------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 2026-04-29 | graphiti-core 0.13.0       | Default OpenAI 8192-token output cap hit during entity extraction on LongMemEval haystack episodes. | Graphiti strategy now (a) chunks each session into 2-turn episodes and (b) instantiates `OpenAIClient(LLMConfig(model="gpt-4o", max_tokens=16384))`. Verified end-to-end on 82 sessions. |
| 2026-04-29 | claude-sonnet-4-6 / opus-4-7| `temperature` parameter rejected by Anthropic API.                                       | `_model_disables_temperature` in `memory_arena/llm/providers.py` skips the field for these model IDs. |
| 2026-04-29 | mem0ai 0.1.114             | `mem0g` setup raised `langchain_neo4j is not installed`.                                  | Listed `langchain-neo4j` in the `mem0` extras (and bundled in dev installs). |
| 2026-04-30 | cognee 1.0.3               | Strict starlette pin (`<0.49`) clashes with FastAPI 0.115. Connection-test loop hangs ~30s. | Run cognee benchmarks from an isolated worktree with a fresher fastapi/starlette pair. Set `ENABLE_BACKEND_ACCESS_CONTROL=false` and `COGNEE_SKIP_CONNECTION_TEST=true`. Use `openai/`-prefixed model names for the embedding endpoint. |
| 2026-04-30 | letta (deferred)           | Single-agent run sends 100k+ tokens of context per turn; 16-question / 82-session bench takes 2+ hours and consumes vendor-internal tokens we can't measure. | Letta is excluded from v0.1.5. Will return when we have a streaming-context config. |

When a vendor SDK ships a breaking minor:

1. Pin to the previous known-good version in `pyproject.toml`.
2. File a row in this table.
3. Open a tracking issue in the repo. Bump the pin in a follow-up PR after verifying both `pip install -e '.[<extra>]'` and the live test (`tests/live/test_<vendor>_live.py`) pass against the new version.
