"""Mem0 strategy — vendor default hybrid memory.

Pinned to mem0ai==2.0.2. Vector store: Chroma. Embedder: OpenAI. LLM: Anthropic Sonnet.

v2 notes:
- v2.0.0 removed the OSS graph_store (~4000 LOC of graph driver code). The
  graph-memory comparison stays in mem0g, which pins v1.
- The Anthropic adapter bug that forced v0.1.4 onto gpt-4o-mini is fixed in v2:
  mem0/llms/anthropic.py now drops top_p when both temperature and top_p are
  set, so Claude Sonnet 4.6+ stops 400ing. We're back on Anthropic to match the
  rest of the harness.
- search() signature changed: `limit` -> `top_k`, top-level entity ids -> `filters`.
"""

from __future__ import annotations

import logging
import os
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


def _llm_config() -> dict:
    """mem0's internal extraction LLM.

    Defaults to Anthropic Sonnet (``settings.generate_model``) so mem0 extracts
    with the same model as the rest of the harness, the leveled apples-to-apples
    config. ``MEM0_EXTRACT_PROVIDER=openai`` (with optional ``MEM0_EXTRACT_MODEL``)
    is a diagnostic knob to A/B the extraction model and separate the model effect
    from the mem0 v1->v2 version change. Keys are passed explicitly because mem0's
    adapters read from their own config, not our settings, and ANTHROPIC_API_KEY is
    loaded into settings from .env but never exported to os.environ.
    """
    if os.environ.get("MEM0_EXTRACT_PROVIDER") == "openai":
        return {
            "provider": "openai",
            "config": {
                "model": os.environ.get("MEM0_EXTRACT_MODEL", "gpt-4o-mini"),
                "temperature": 0.0,
                "api_key": settings.openai_api_key,
            },
        }
    return {
        "provider": "anthropic",
        "config": {
            "model": settings.generate_model,
            "temperature": 0.0,
            "api_key": settings.anthropic_api_key,
        },
    }


def _build_config(run_id: str) -> dict:
    # Hold embedder + LLM constant with the rest of the harness so the leaderboard
    # is apples-to-apples. v2's Anthropic adapter no longer 400s on Claude 4.x.
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"{settings.mem0_collection_prefix}_{run_id}",
                "path": settings.chroma_path,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.embedding_model,
                "api_key": settings.openai_api_key,
            },
        },
        "llm": _llm_config(),
    }


class Mem0Strategy(MemoryStrategy):
    name = "mem0"

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
            raise RuntimeError("mem0ai not installed. pip install 'memory-arena[mem0]'") from exc
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
                # v2: filters dict replaces top-level user_id; top_k replaces limit.
                results = self._client.search(
                    query,
                    top_k=top_k,
                    filters={"user_id": uid},
                )
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
