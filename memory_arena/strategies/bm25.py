"""BM25 lexical baseline.

The honest non-neural floor. Often beats dense retrieval on out-of-distribution
queries (proper nouns, exact phrases, rare terms). No vendor SDK, no embedding
calls during ingest.
"""

from __future__ import annotations

import re
import time

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session, Turn
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class BM25Strategy(MemoryStrategy):
    name = "bm25"

    def __init__(self) -> None:
        super().__init__()
        self._turns: list[Turn] = []
        self._tokenized: list[list[str]] = []
        self._llm: LLMClient | None = None
        self._bm25 = None

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._turns = []
        self._tokenized = []
        self._bm25 = None
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        for turn in session.turns:
            self._turns.append(turn)
            self._tokenized.append(_tokenize(f"{turn.role} {turn.content}"))
        self._bm25 = None  # invalidate; rebuild lazily on next recall
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            facts_extracted=len(session.turns),
        )

    def _build_index(self):
        from rank_bm25 import BM25Okapi

        if self._bm25 is None and self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        return self._bm25

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        start = self._start_timer()
        retrieval_start = time.perf_counter()
        bm25 = self._build_index()
        if bm25 is None or not self._turns:
            return RecallResult(answer="I do not have that information.", strategy=self.name)
        q_tokens = _tokenize(query)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        memories: list[dict] = []
        session_ids: list[str] = []
        turn_ids: list[str] = []
        for i in ranked:
            t = self._turns[i]
            memories.append(
                {
                    "session_id": t.session_id,
                    "turn_id": t.id,
                    "score": float(scores[i]),
                    "content": t.content,
                    "role": t.role,
                }
            )
            if t.session_id not in session_ids:
                session_ids.append(t.session_id)
            if t.id not in turn_ids:
                turn_ids.append(t.id)

        context = "\n\n---\n\n".join(
            f"[session={m['session_id']} score={m['score']:.2f}] {m['role']}: {m['content']}"
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
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        self._turns = []
        self._tokenized = []
        self._bm25 = None
        self._llm = None
