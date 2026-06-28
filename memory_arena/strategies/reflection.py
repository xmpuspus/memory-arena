"""Reflection-augmented memory (Generative-Agents pattern, Park et al. 2023).

Stores raw turns in a vector DB AND, periodically, asks the LLM to write
higher-level reflections on the user. Both reflections and raw turns are
retrieved at query time. Tests whether interpretive memory beats literal
extraction.
"""

from __future__ import annotations

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.naive_vector import NaiveVectorStrategy

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Some memories are raw turns, others are reflections written by an analyst. "
    "Use only the retrieved content. Cite sessions by their id in square brackets. "
    'If the memories do not contain the answer, say "I do not have that information."'
)

_REFLECTION_SYSTEM = (
    "You are an analyst summarizing a user's recent chat sessions into 5-8 short reflections. "
    "Each reflection states one durable fact, preference, or pattern about the user. "
    "Format as a numbered list. Be concise and concrete."
)


class ReflectionStrategy(MemoryStrategy):
    name = "reflection"

    def __init__(self) -> None:
        super().__init__()
        self._vector = NaiveVectorStrategy()
        self._buffer: list[Session] = []
        self._reflection_every_n: int = 4
        self._llm: LLMClient | None = None
        self._reflection_count = 0

    def _collection_name(self) -> str:
        return f"reflection_{self.run_id}" if self.run_id else "reflection_default"

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        # Prefix the inner Chroma collection with strategy name so concurrent
        # strategies that wrap NaiveVectorStrategy don't share a collection.
        await self._vector.setup(f"reflection_{run_id}")
        self._buffer = []
        self._reflection_count = 0
        self._llm = LLMClient()

    async def _flush_reflections(self) -> tuple[int, float]:
        if not self._buffer or self._llm is None:
            return 0, 0.0
        # Compose recent window text
        lines: list[str] = []
        for sess in self._buffer:
            lines.append(f"### Session {sess.id} ({sess.timestamp or 'unknown'})")
            for turn in sess.turns:
                lines.append(f"{turn.role}: {turn.content}")
        recent = "\n".join(lines)[:12000]
        resp = await self._llm.generate(query="", context=recent, system_prompt=_REFLECTION_SYSTEM)

        # Each reflection becomes a synthetic "turn" pinned to the most recent session
        anchor = self._buffer[-1]
        from memory_arena.sessions.schema import Session as ReflectSession
        from memory_arena.sessions.schema import Turn as ReflectTurn

        synth_turns = []
        for i, line in enumerate(resp.text.splitlines()):
            line = line.strip()
            if not line:
                continue
            synth_turns.append(
                ReflectTurn(
                    id=f"reflection_{self.run_id}_{self._reflection_count:03d}_{i:02d}",
                    session_id=f"reflection_{anchor.id}",
                    role="analyst",
                    content=line,
                    timestamp=anchor.timestamp,
                    metadata={"kind": "reflection", "anchor_session": anchor.id},
                )
            )
        if synth_turns:
            synth_session = ReflectSession(
                id=f"reflection_{anchor.id}_{self._reflection_count:03d}",
                user_id=anchor.user_id,
                timestamp=anchor.timestamp,
                turns=synth_turns,
                metadata={"kind": "reflection_session"},
            )
            await self._vector.ingest_session(synth_session)
        self._reflection_count += 1
        self._buffer = []
        return len(synth_turns), resp.cost_usd

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        v_rec = await self._vector.ingest_session(session)
        self._buffer.append(session)
        cost = v_rec.cost_usd
        facts = v_rec.facts_extracted

        if len(self._buffer) >= self._reflection_every_n:
            extra_facts, extra_cost = await self._flush_reflections()
            cost += extra_cost
            facts += extra_facts

        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            cost_usd=cost,
            facts_extracted=facts,
        )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        # Flush any remaining buffered sessions into reflections before recall.
        if self._buffer:
            await self._flush_reflections()
        start = self._start_timer()
        result = await self._vector.recall(query, top_k=top_k)
        # Drop the synthetic reflection_ session ids from supporting; keep the
        # anchor sessions they pointed at.
        cleaned_session_ids: list[str] = []
        for m in result.retrieved_memories:
            sid = m.get("session_id", "")
            if sid.startswith("reflection_"):
                anchor = sid.replace("reflection_", "", 1).rsplit("_", 1)[0]
                if anchor and anchor not in cleaned_session_ids:
                    cleaned_session_ids.append(anchor)
            elif sid and sid not in cleaned_session_ids:
                cleaned_session_ids.append(sid)
        elapsed = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=result.answer,
            supporting_session_ids=cleaned_session_ids or result.supporting_session_ids,
            supporting_turn_ids=result.supporting_turn_ids,
            retrieved_memories=result.retrieved_memories,
            strategy=self.name,
            latency_ms=result.latency_ms + elapsed,
            retrieval_latency_ms=result.retrieval_latency_ms,
            generation_latency_ms=result.generation_latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
        )

    async def teardown(self) -> None:
        await self._vector.teardown()
        self._buffer = []
        self._llm = None
