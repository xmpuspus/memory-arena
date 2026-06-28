# Contributing to Memory Arena

Thanks for your interest. This guide covers setup, conventions, and how to add new corpora and strategies.

## Development Setup

```bash
git clone https://github.com/xmpuspus/memory-arena
cd memory-arena
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Vendor strategies are optional extras
pip install 'memory-arena[mem0]'      # mem0 + mem0g
pip install 'memory-arena[graphiti]'  # graphiti
pip install 'memory-arena[cognee]'    # cognee
pip install 'memory-arena[langmem]'   # langmem
pip install 'memory-arena[memori]'    # memori

# Backends
docker compose up -d neo4j     # graphiti, mem0g
docker compose up -d postgres  # memori
```

## Local State

ChromaDB persistence lives in `chroma_data/` and is gitignored. Each `run_id` is its own collection; older orphans can be safely deleted with `python -c "import shutil; shutil.rmtree('chroma_data', ignore_errors=True)"` between major runs.

## Running Tests

```bash
# Unit tests (fast, mocked, no live API calls)
pytest tests/ -q

# All gates
ruff check . && ruff format --check . && pytest tests/ -q
```

## Code Style

Ruff handles both linting and formatting. Run before every commit:

```bash
ruff check . --fix    # lint + auto-fix
ruff format .         # format
```

CI rejects PRs that don't pass `ruff check . && ruff format --check .`.

## Project Conventions

- **Python 3.11+** - `from __future__ import annotations`, `X | Y` unions, `match`.
- **Pydantic v2** - `BaseModel` with `ConfigDict(extra="forbid")` on response/interchange models.
- **Async first** - every strategy method is async, the runner is a single `asyncio.gather` across strategies.
- **Type hints** - public functions and methods are typed.
- **Settings** - `pydantic-settings` with `MEM_ARENA_` prefix, defined in `memory_arena/settings.py`.
- **No em-dashes** in user-facing copy. Hyphens, commas, periods, parentheses, or colons.
- **No emoji** in code, docs, or output. Use `[PASS]`, `[FAIL]`, `[WARN]` for status.

## Adding a New Memory Strategy

The contract lives in `memory_arena/strategies/base.py`:

```python
class MemoryStrategy(ABC):
    name: str

    async def setup(self, run_id: str) -> None: ...
    async def ingest_session(self, session: Session) -> IngestRecord: ...
    async def recall(self, query: str, top_k: int = 10) -> RecallResult: ...
    async def teardown(self) -> None: ...
```

### 1. Implementation

Create `memory_arena/strategies/your_strategy.py`:

```python
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult


class YourStrategy(MemoryStrategy):
    name = "your_strategy"

    async def setup(self, run_id: str) -> None:
        # Idempotent. Create namespace, schema, etc.
        self.run_id = run_id
        ...

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        # Sequentially ingest the session. Vendor systems often reason about
        # insertion order, so do not parallelize within a strategy.
        ...
        return IngestRecord(session_id=session.id, latency_ms=..., cost_usd=...)

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        # Run retrieval + answer generation. Return supporting_session_ids
        # so the evaluator's source-attribution axis can score the answer.
        ...
        return RecallResult(answer=..., supporting_session_ids=[...], ...)

    async def teardown(self) -> None:
        # Drop the run namespace. Critical for benchmark isolation.
        ...
```

### 2. Registration

Add to `memory_arena/strategies/__init__.py`. If your strategy depends on an optional vendor SDK, register it through `_register_optional()` so the package still imports cleanly when the SDK is missing.

### 3. Tests

Add a class to `tests/test_strategies.py` covering:
- `setup` initializes state correctly.
- `ingest_session` returns a valid `IngestRecord`.
- `recall` returns a `RecallResult` with `supporting_session_ids` populated.
- `teardown` is idempotent.
- If the strategy uses a vendor SDK, add a graceful-failure test that monkey-patches the import to raise `ImportError`.

### 4. Vendor SDK Pin

Add the SDK to `pyproject.toml` `[project.optional-dependencies]` with an exact version pin. Document the pin and any known limits in `docs/vendor-pins.md`.

## Adding a New Corpus

A corpus is a chat-session dataset with ground-truth questions. To add one:

### 1. Loader

Create a class in `memory_arena/sessions/loaders.py` (or a new file in `memory_arena/sessions/`) that produces `(list[Session], list[QuestionRecord])`.

```python
from memory_arena.sessions.schema import Session, QuestionRecord, Turn


class YourLoader:
    def __init__(self, raw_path: str | Path):
        self.raw_path = Path(raw_path)

    def normalize(self) -> tuple[list[Session], list[QuestionRecord]]:
        ...
```

### 2. Question schema

Each question is a `QuestionRecord` with one of five categories:

- `information_extraction` - single-fact recall from one session
- `multi_session_reasoning` - combine facts from 2+ sessions
- `temporal` - time-aware queries
- `knowledge_update` - user changed their mind across sessions; answer must reflect the latest version
- `abstention` - question has no answer in the chat history; system should refuse

Add `must_mention` / `must_not_claim` constraints when the structural axis should fire on a specific phrasing. Set `abstention_expected: true` for abstention questions so the F1 axis scores them correctly.

### 3. Layout

```
datasets/your-corpus/
  raw/                       # downloaded corpus
  processed/
    sessions.jsonl           # one Session per line
    questions.jsonl          # one QuestionRecord per line
  questions/
    smoke/                   # optional: 6-per-category YAML smoke subset
```

The benchmark loader prefers `questions/smoke/*.yaml` if present, otherwise falls back to `processed/questions.jsonl`.

### 4. Tests

- A `tests/test_loaders.py` test class for the new loader.
- A `tests/test_smoke_questions_yaml.py`-style validation if you ship YAML.

## Pull Requests

- One feature per PR.
- Include tests.
- Run `ruff check . && ruff format --check . && pytest tests/ -q` before submitting.
- Simple commit messages, no conventional-commit prefixes.
- No AI co-author trailers.

## Reporting Issues

Open a GitHub issue with:
- What you expected.
- What happened.
- Steps to reproduce.
- Python version, OS, vendor SDK versions.
