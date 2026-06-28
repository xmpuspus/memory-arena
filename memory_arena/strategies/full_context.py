"""Full-context baseline: stuff every ingested turn into the prompt up to a token budget.

Ceiling baseline. No retrieval; just dump everything. Useful as a topline for what
the LLM could answer if it had perfect recall.
"""

from __future__ import annotations

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

_RECALL_SYSTEM = (
    "You are an assistant answering a question about a user based on a complete chat history. "
    "Use only information that is explicitly stated in the history. Cite sessions by their id "
    "in square brackets, e.g. [session_03]. If the history does not contain the answer, say "
    '"I do not have that information."'
)


class FullContextStrategy(MemoryStrategy):
    name = "full_context"

    def __init__(self) -> None:
        super().__init__()
        self._sessions: list[Session] = []
        self._llm: LLMClient | None = None

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._sessions = []
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        self._sessions.append(session)
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
        budget = settings.full_context_token_budget
        # Rough character budget (~4 chars/token).
        char_budget = budget * 4

        chunks: list[str] = []
        used_session_ids: list[str] = []
        used_chars = 0
        for session in self._sessions:
            block_lines = [f"### Session {session.id} ({session.timestamp or 'unknown'})"]
            for turn in session.turns:
                block_lines.append(f"{turn.role}: {turn.content}")
            block = "\n".join(block_lines)
            if used_chars + len(block) > char_budget and chunks:
                break
            chunks.append(block)
            used_session_ids.append(session.id)
            used_chars += len(block)

        context = "\n\n".join(chunks)
        if self._llm is None:
            self._llm = LLMClient()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        latency = (self._start_timer() - start) * 1000

        return RecallResult(
            answer=resp.text,
            supporting_session_ids=used_session_ids,
            supporting_turn_ids=[],
            retrieved_memories=[{"sessions": used_session_ids, "char_count": used_chars}],
            strategy=self.name,
            latency_ms=latency,
            generation_latency_ms=latency,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        self._sessions = []
        self._llm = None
