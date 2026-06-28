"""Persona-profile compression baseline.

At ingest time, accumulates raw sessions into an in-memory rolling buffer.
Just before the first recall (or whenever the buffer fills), distills the
buffer into a single user profile JSON via Haiku, then prepends that profile
to every recall prompt alongside top-k vector retrievals.

Tests whether one well-distilled profile beats a flat vector store of turns.
"""

from __future__ import annotations

import json
import time

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.naive_vector import NaiveVectorStrategy

_PROFILE_SYSTEM = (
    "You are profiling a user from their chat history. Output a JSON object with exactly these "
    "keys: identity (name, role, employer), location, hobbies, preferences, recent_changes "
    "(list of {what, when}), open_questions. Use null for unknown fields. Keep each value short. "
    "Reply with only the JSON object."
)

_RECALL_SYSTEM = (
    "You are an assistant answering a question using a user profile and retrieved memories. "
    "Prefer profile facts; fall back to retrieved turns when needed. Cite sessions by their id "
    'in square brackets. If neither contains the answer, say "I do not have that information."'
)


class PersonaProfileStrategy(MemoryStrategy):
    name = "persona_profile"

    def __init__(self) -> None:
        super().__init__()
        self._vector = NaiveVectorStrategy()
        self._sessions_buffer: list[Session] = []
        self._profile: str = ""
        self._profile_built = False
        self._llm: LLMClient | None = None

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        await self._vector.setup(f"persona_{run_id}")
        self._sessions_buffer = []
        self._profile = ""
        self._profile_built = False
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        v_rec = await self._vector.ingest_session(session)
        self._sessions_buffer.append(session)
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            cost_usd=v_rec.cost_usd,
            facts_extracted=v_rec.facts_extracted,
        )

    async def _build_profile(self) -> tuple[str, float, int]:
        if not self._sessions_buffer or self._llm is None:
            return "", 0.0, 0
        lines: list[str] = []
        # Up to ~15K chars from the buffer (profile fits one Haiku call)
        char_budget = 15000
        used = 0
        for sess in self._sessions_buffer:
            block = [f"### Session {sess.id} ({sess.timestamp or 'unknown'})"]
            for turn in sess.turns:
                block.append(f"{turn.role}: {turn.content}")
            block_text = "\n".join(block)
            if used + len(block_text) > char_budget and lines:
                break
            lines.append(block_text)
            used += len(block_text)
        history = "\n\n".join(lines)
        resp = await self._llm.generate(
            query="Profile this user.", context=history, system_prompt=_PROFILE_SYSTEM
        )
        return resp.text.strip(), resp.cost_usd, resp.total_tokens

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if not self._profile_built:
            self._profile, build_cost, build_toks = await self._build_profile()
            self._profile_built = True
        else:
            build_cost = 0.0
            build_toks = 0

        start = self._start_timer()
        retrieval_start = time.perf_counter()
        v_result = await self._vector.recall(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        try:
            profile_pretty = (
                json.dumps(json.loads(self._profile), indent=2) if self._profile else "(no profile)"
            )
        except Exception:
            profile_pretty = self._profile or "(no profile)"

        retrieved_block = "\n\n".join(
            f"[session={m.get('session_id', '?')}] {m.get('role', '')}: {m.get('content', '')}"
            for m in v_result.retrieved_memories
        )
        context = f"USER PROFILE:\n{profile_pretty}\n\nRETRIEVED TURNS:\n{retrieved_block}"

        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=resp.text,
            supporting_session_ids=v_result.supporting_session_ids,
            supporting_turn_ids=v_result.supporting_turn_ids,
            retrieved_memories=v_result.retrieved_memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens + build_toks + (v_result.tokens_used or 0),
            cost_usd=resp.cost_usd + build_cost + (v_result.cost_usd or 0),
        )

    async def teardown(self) -> None:
        await self._vector.teardown()
        self._sessions_buffer = []
        self._profile = ""
        self._profile_built = False
        self._llm = None
