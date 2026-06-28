"""LangMem strategy - LangGraph-native long-term memory manager.

Uses LangMem's `create_memory_store_manager` over an in-memory LangGraph store.
Each ingested session is fed through the manager which produces extracted
memories. Recall uses the search tool against the same store.

Pure Python: no Docker, no vendor cloud key beyond an LLM API key.
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


class LangMemStrategy(MemoryStrategy):
    name = "langmem"
    # LangMem stores extracted memories indexed by content, not by chat-session
    # id. Recall@k against LongMemEval session-level ground truth isn't a fair
    # metric for this data model — the runner consults the class attribute and
    # renders "—" instead of attributing a structural-zero to the vendor.
    recall_at_k_measurable = False

    def __init__(self) -> None:
        super().__init__()
        self._store = None
        self._manager = None
        self._namespace: tuple[str, ...] = ()
        self._llm: LLMClient | None = None
        self._errors: list[dict] = []

    async def setup(self, run_id: str) -> None:
        try:
            import os

            from langgraph.store.memory import InMemoryStore
            from langmem import create_memory_store_manager
        except ImportError as exc:
            raise RuntimeError("langmem / langgraph not installed. pip install langmem") from exc
        if not settings.openai_api_key:
            raise RuntimeError("MEM_ARENA_OPENAI_API_KEY required for langmem")
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        # Level the fact-extraction LLM to the harness model (Claude Sonnet) so
        # the only variable vs the baselines is the memory architecture, not the
        # model. Embeddings stay on OpenAI text-embedding-3-large (pinned across
        # the table). init_chat_model reads ANTHROPIC_API_KEY from the env.
        if settings.anthropic_api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        self.run_id = run_id
        self._errors = []
        self._namespace = ("memory_arena", run_id)
        # Hold embedder constant with the rest of the leaderboard by reading
        # settings.embedding_model. text-embedding-3-large -> 3072 dims;
        # -3-small -> 1536. Pin both ends so InMemoryStore's index matches.
        self._store = InMemoryStore(
            index={
                "dims": settings.embedding_dimensions,
                "embed": f"openai:{settings.embedding_model}",
            }
        )
        self._manager = create_memory_store_manager(
            f"anthropic:{settings.generate_model}",
            namespace=self._namespace,
            store=self._store,
            enable_inserts=True,
        )
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._manager is None or self._store is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        messages = [
            {"role": t.role if t.role in ("user", "assistant") else "user", "content": t.content}
            for t in session.turns
            if t.role in ("user", "assistant")
        ]
        err = ""
        try:
            await self._manager.ainvoke(
                {"messages": messages},
                config={"configurable": {"langgraph_user_id": self.run_id}},
            )
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
        if self._store is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        retrieval_start = time.perf_counter()
        memories: list[dict] = []
        try:
            results = await self._store.asearch(self._namespace, query=query, limit=top_k)
        except Exception as exc:
            logger.warning("strategy=%s recall search failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "search",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
            return RecallResult(
                answer=f"[ERROR] langmem search: {exc}",
                strategy=self.name,
                latency_ms=(self._start_timer() - start) * 1000,
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        for item in results or []:
            value = getattr(item, "value", None) or {}
            memories.append({"content": str(value.get("content") or value.get("memory") or value)})

        context = "\n\n---\n\n".join(m["content"] for m in memories)
        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        # recall_at_k_measurable=False is declared on the class above; the
        # runner aggregates from the class attribute.
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
        self._store = None
        self._manager = None
        self._llm = None
