"""Memori strategy - SQL-native on-prem agent memory.

Uses Memori 3.x's enqueue/wait augmentation pipeline for ingest and
``recall(query, limit=k)`` for retrieval. Stores facts in the docker compose
Postgres container.

Caveat: even in BYODB mode with a local conn factory, Memori 3.x routes its
augmentation runtime through a cloud quota service that 429s anonymous IPs
after a few requests. Set ``MEMORI_API_KEY`` in your environment for full
throughput; without it the benchmark will run but augmentation will be
disabled mid-run and accuracy will drop close to zero.
"""

from __future__ import annotations

import logging
import time

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

logger = logging.getLogger(__name__)

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)


class MemoriStrategy(MemoryStrategy):
    name = "memori"
    # Memori's recall returns content rows that don't carry the original
    # chat-session_id (the augmentation pipeline collapses turns into facts).
    # Recall@k against session-level ground truth isn't a fair metric for this
    # data model — the runner reads this class attribute and renders "—".
    recall_at_k_measurable = False

    def __init__(self) -> None:
        super().__init__()
        self._memori = None
        self._llm: LLMClient | None = None
        self._errors: list[dict] = []

    def _conn_factory(self):
        import psycopg

        host = settings.postgres_host or "localhost"
        port_env = 5433  # docker-compose maps host port 5433 -> container 5432
        # autocommit=True prevents InFailedSqlTransaction when the augmentation
        # pipeline aborts mid-operation. Memori's writer issues many small
        # statements that don't need to be in one transaction.
        return psycopg.connect(
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{host}:{port_env}/{settings.postgres_database}",
            autocommit=True,
        )

    async def setup(self, run_id: str) -> None:
        # Memori 3.x's BYODB API does not expose an embedder kwarg; the
        # augmentation runtime owns embedding internally. We can't hold this
        # strategy's embedder to settings.embedding_model the way we do for
        # mem0/langmem/cognee. Document and accept.
        try:
            from memori import Memori
        except ImportError as exc:
            raise RuntimeError("memori not installed. pip install memori") from exc
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            # Don't subprocess-pip-install at runtime: surface a clean error and
            # let the user pick the install command. The 'memori' extras pin
            # psycopg[binary] so `pip install 'memory-arena[memori]'` is the fix.
            raise ImportError(
                "memori strategy requires psycopg. pip install 'memory-arena[memori]'"
            ) from exc
        self.run_id = run_id
        self._errors = []
        self._memori = Memori(conn=self._conn_factory)
        self._memori.new_session()
        # BYO LLM: register an Anthropic client so Memori does fact extraction
        # locally instead of phoning home to the cloud API (which 429s on free
        # tier). Without this, augmentation is disabled and recall returns nothing.
        if settings.anthropic_api_key:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                self._memori.anthropic.register(client=client)
            except Exception as exc:
                logger.warning(
                    "strategy=%s setup anthropic register failed: %s",
                    self.name,
                    exc,
                )
                self._errors.append(
                    {
                        "phase": "setup",
                        "step": "anthropic_register",
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._memori is None:
            raise RuntimeError("setup() not called")
        from memori.memory.augmentation._message import ConversationMessage
        from memori.memory.augmentation.input import AugmentationInput

        start = self._start_timer()
        messages = [
            ConversationMessage(
                role=t.role if t.role in ("user", "assistant") else "user",
                content=t.content,
            )
            for t in session.turns
            if t.role in ("user", "assistant")
        ]
        err = ""
        try:
            self._memori.set_session(session_id=f"{self.run_id}_{session.id}")
            input_data = AugmentationInput(
                conversation_id=session.id,
                entity_id=self.run_id,
                process_id=None,
                conversation_messages=messages,
            )
            self._memori.augmentation.enqueue(input_data)
        except Exception as exc:
            err = f"ingest: {exc}"
            logger.warning(
                "strategy=%s ingest session=%s failed: %s",
                self.name,
                session.id,
                exc,
            )
            self._errors.append(
                {
                    "phase": "ingest",
                    "session_id": session.id,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            facts_extracted=len(messages),
            error=err,
        )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._memori is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        # Memori 3.x runs ingest async via AugmentationManager; wait briefly for
        # any queued enqueues to finish before recalling.
        try:
            self._memori.augmentation.wait(timeout=120.0)
        except Exception as exc:
            logger.warning("strategy=%s recall augmentation.wait failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "augmentation_wait",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        retrieval_start = time.perf_counter()
        memories: list[dict] = []
        try:
            recalls = self._memori.recall(query, limit=top_k)
        except Exception as exc:
            logger.warning("strategy=%s recall failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "recall",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
            return RecallResult(
                answer=f"[ERROR] memori recall: {exc}",
                strategy=self.name,
                latency_ms=(self._start_timer() - start) * 1000,
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        for r in recalls or []:
            memories.append({"content": getattr(r, "content", str(r))})

        context = "\n\n---\n\n".join(m["content"] for m in memories)
        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        # recall_at_k_measurable=False is declared on the class above; the
        # runner consults the class attribute when aggregating.
        return RecallResult(
            answer=resp.text,
            supporting_session_ids=[],
            retrieved_memories=memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        try:
            if self._memori is not None:
                self._memori.close()
        except Exception as exc:
            logger.warning("strategy=%s teardown close failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "teardown",
                    "step": "close",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        self._memori = None
        self._llm = None
