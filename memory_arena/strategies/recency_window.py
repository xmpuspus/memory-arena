"""Recency-window baseline: last N turns across all sessions.

Cheap baseline. No retrieval, just whatever was said most recently.
"""

from __future__ import annotations

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session, Turn
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

_RECALL_SYSTEM = (
    "You are an assistant answering a question using only the most recent N turns of "
    "conversation. Use only information that appears in those turns. Cite turns by their "
    "session id in square brackets. If the recent window does not contain the answer, say "
    '"I do not have that information."'
)


class RecencyWindowStrategy(MemoryStrategy):
    name = "recency_window"

    def __init__(self) -> None:
        super().__init__()
        self._turns: list[Turn] = []
        self._llm: LLMClient | None = None

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._turns = []
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        self._turns.extend(session.turns)
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            tokens_used=0,
            cost_usd=0.0,
            facts_extracted=len(session.turns),
        )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        start = self._start_timer()
        n = settings.recency_window_n
        recent = self._turns[-n:] if len(self._turns) > n else self._turns

        lines: list[str] = []
        used_session_ids: list[str] = []
        seen: set[str] = set()
        for turn in recent:
            lines.append(f"[{turn.session_id}] {turn.role}: {turn.content}")
            if turn.session_id not in seen:
                seen.add(turn.session_id)
                used_session_ids.append(turn.session_id)
        context = "\n".join(lines)

        if self._llm is None:
            self._llm = LLMClient()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        latency = (self._start_timer() - start) * 1000

        return RecallResult(
            answer=resp.text,
            supporting_session_ids=used_session_ids,
            supporting_turn_ids=[t.id for t in recent],
            retrieved_memories=[{"window_size": n, "actual_turns": len(recent)}],
            strategy=self.name,
            latency_ms=latency,
            generation_latency_ms=latency,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        self._turns = []
        self._llm = None
