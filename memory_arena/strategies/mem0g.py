"""Mem0 with graph memory enabled (Neo4j-backed) — pinned to mem0ai v1.

DEPRECATED for v0.2: mem0ai v2.0.0 removed the OSS graph_store entirely
(~4000 LOC of graph driver code deleted). Until graphiti covers the graph-memory
comparison axis on its own, we keep mem0g on v1 to preserve apples-to-apples
benchmarking against the v0.1.x leaderboard.

Install via ``pip install 'memory-arena[mem0g]'``. Cannot coexist with the
``[mem0]`` extra (which pins v2) in the same environment — install in separate
virtualenvs if you need both strategies.

This module deliberately re-implements the v1 client surface instead of
subclassing :class:`Mem0Strategy` because v1 and v2 disagree on the
``search()`` signature (v1: ``user_id`` + ``limit``; v2: ``filters`` + ``top_k``)
and on whether ``graph_store`` is a valid config key.
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
    "You are an assistant answering a question using retrieved memories. "
    "Use only information from the memories. Cite sessions by their id in square brackets. "
    'If the memories do not contain the answer, say "I do not have that information."'
)


def _build_config(run_id: str) -> dict:
    # v1 still ships the Anthropic-adapter bug where temperature + top_p are both
    # sent on every call, which Claude 4.x rejects. v1's default LLM is
    # openai/gpt-4o-mini, so we use that to keep the strategy operational; the
    # honest fix lives in v2 which mem0g cannot adopt yet (no graph_store there).
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"{settings.mem0_collection_prefix}g_{run_id}",
                "path": settings.chroma_path,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": settings.embedding_model},
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": settings.neo4j_uri,
                "username": settings.neo4j_user,
                "password": settings.neo4j_password,
            },
        },
    }


class Mem0GraphStrategy(MemoryStrategy):
    name = "mem0g"
    _graph_enabled = True  # retained for back-compat with existing tests

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._llm: LLMClient | None = None
        self._user_ids: set[str] = set()
        self._errors: list[dict] = []

    async def setup(self, run_id: str) -> None:
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError(
                "mem0ai not installed. pip install 'memory-arena[mem0g]' "
                "(pins mem0ai==0.1.114; cannot coexist with [mem0] v2)"
            ) from exc
        self.run_id = run_id
        config = _build_config(run_id)
        self._client = Memory.from_config(config)
        self._llm = LLMClient()
        self._user_ids = set()
        self._errors = []

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._client is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        messages = [
            {"role": t.role, "content": t.content}
            for t in session.turns
            if t.role in ("user", "assistant")
        ]
        try:
            self._client.add(
                messages,
                user_id=session.user_id,
                metadata={
                    "run_id": self.run_id,
                    "session_id": session.id,
                    "timestamp": session.timestamp,
                },
            )
        except Exception as exc:
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
                error=str(exc),
            )
        self._user_ids.add(session.user_id)
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            facts_extracted=len(messages),
        )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._client is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        retrieval_start = time.perf_counter()
        all_memories: list[dict] = []
        for uid in self._user_ids or {"default"}:
            try:
                # v1 API: user_id + limit (top-level kwargs).
                results = self._client.search(query, user_id=uid, limit=top_k)
            except Exception as exc:
                logger.warning(
                    "strategy=%s recall search user=%s failed: %s",
                    self.name,
                    uid,
                    exc,
                )
                self._errors.append(
                    {
                        "phase": "recall",
                        "step": "search",
                        "user_id": uid,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )
                results = {"results": []}
            mems = results.get("results", []) if isinstance(results, dict) else results
            for m in mems[:top_k]:
                all_memories.append(m if isinstance(m, dict) else {"memory": str(m)})
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        session_ids: list[str] = []
        turn_ids: list[str] = []
        for m in all_memories:
            meta = m.get("metadata", {}) or {}
            sid = meta.get("session_id", "")
            if sid and sid not in session_ids:
                session_ids.append(sid)

        def _fmt(m: dict) -> str:
            sid = (m.get("metadata") or {}).get("session_id", "?")
            mid = m.get("id", "?")
            body = m.get("memory", m.get("text", ""))
            return f"[mem id={mid} session={sid}] {body}"

        context = "\n\n".join(_fmt(m) for m in all_memories)

        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=resp.text,
            supporting_session_ids=session_ids,
            supporting_turn_ids=turn_ids,
            retrieved_memories=all_memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        if self._client is None:
            return
        try:
            for uid in self._user_ids:
                try:
                    self._client.delete_all(user_id=uid)
                except Exception as exc:
                    logger.warning(
                        "strategy=%s teardown delete_all user=%s failed: %s",
                        self.name,
                        uid,
                        exc,
                    )
                    self._errors.append(
                        {
                            "phase": "teardown",
                            "step": "delete_all",
                            "user_id": uid,
                            "error": str(exc),
                            "type": type(exc).__name__,
                        }
                    )
        finally:
            self._client = None
            self._llm = None
            self._user_ids = set()
