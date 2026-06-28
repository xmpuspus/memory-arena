"""Hybrid retrieval: vector + BM25 fused via Reciprocal Rank Fusion.

The 2024-26 production default for general-purpose RAG. Independently runs
naive_vector and bm25, then combines their rankings with RRF (k=60). Neither
ranking dominates; outliers from either side are smoothed out.
"""

from __future__ import annotations

import time

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.bm25 import BM25Strategy
from memory_arena.strategies.naive_vector import NaiveVectorStrategy

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)

_RRF_K = 60


class HybridRRFStrategy(MemoryStrategy):
    name = "hybrid_rrf"
    # Each branch (vector + BM25) retrieves
    # `max(top_k * CANDIDATE_POOL_MULTIPLIER, 20)` candidates; RRF fuses them
    # and we return exactly top_k. This is a fusion candidate pool, not the
    # number of items handed to the generator.
    CANDIDATE_POOL_MULTIPLIER = 4

    def __init__(self) -> None:
        super().__init__()
        self._vector = NaiveVectorStrategy()
        self._bm25 = BM25Strategy()
        self._llm: LLMClient | None = None

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        await self._vector.setup(run_id + "_vec")
        await self._bm25.setup(run_id + "_bm25")
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        v_rec = await self._vector.ingest_session(session)
        b_rec = await self._bm25.ingest_session(session)
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            cost_usd=v_rec.cost_usd + b_rec.cost_usd,
            facts_extracted=v_rec.facts_extracted,
        )

    async def recall(self, query: str, top_k: int = 5) -> RecallResult:
        start = self._start_timer()
        retrieval_start = time.perf_counter()
        # Each branch retrieves `max(top_k * CANDIDATE_POOL_MULTIPLIER, 20)`
        # candidates; RRF fuses them and we return top_k. The 4x multiplier
        # widens the fusion window without changing what the generator sees.
        candidate_pool_size = max(top_k * self.CANDIDATE_POOL_MULTIPLIER, 20)
        vector_result = await self._vector.recall(query, top_k=candidate_pool_size)
        bm25_result = await self._bm25.recall(query, top_k=candidate_pool_size)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # RRF score: sum of 1/(k+rank) across rankings
        rrf_scores: dict[str, float] = {}
        turn_payload: dict[str, dict] = {}
        for rank, m in enumerate(vector_result.retrieved_memories):
            tid = m.get("turn_id", "")
            if not tid:
                continue
            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            turn_payload.setdefault(tid, m)
        for rank, m in enumerate(bm25_result.retrieved_memories):
            tid = m.get("turn_id", "")
            if not tid:
                continue
            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            turn_payload.setdefault(tid, m)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        memories: list[dict] = []
        session_ids: list[str] = []
        turn_ids: list[str] = []
        for tid, score in ranked:
            m = dict(turn_payload[tid])
            m["rrf"] = score
            memories.append(m)
            sid = m.get("session_id", "")
            if sid and sid not in session_ids:
                session_ids.append(sid)
            if tid and tid not in turn_ids:
                turn_ids.append(tid)

        context = "\n\n---\n\n".join(
            f"[session={m.get('session_id', '?')} rrf={m.get('rrf', 0):.4f}] "
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in memories
        )

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
            retrieved_memories=memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens
            + (vector_result.tokens_used or 0)
            + (bm25_result.tokens_used or 0),
            cost_usd=resp.cost_usd + (vector_result.cost_usd or 0) + (bm25_result.cost_usd or 0),
        )

    async def teardown(self) -> None:
        await self._vector.teardown()
        await self._bm25.teardown()
        self._llm = None
