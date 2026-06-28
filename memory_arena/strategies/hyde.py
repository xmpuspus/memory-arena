"""HyDE: Hypothetical Document Embeddings.

Before retrieval, ask the LLM to write a *hypothetical* answer to the query.
Embed THAT, retrieve the closest real turns by cosine, then answer normally.
The hypothetical answer often matches the embedding distribution of the
correct memory better than the bare query does.
"""

from __future__ import annotations

import time

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.naive_vector import NaiveVectorStrategy

_HYDE_SYSTEM = (
    "Write a brief plausible answer to the question as if you knew the user well. "
    "Do not say you do not know - just guess a reasonable answer. "
    "One short paragraph. This text will be used as a vector retrieval probe."
)

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)


class HydeStrategy(MemoryStrategy):
    name = "hyde"

    def __init__(self) -> None:
        super().__init__()
        self._vector = NaiveVectorStrategy()
        self._llm: LLMClient | None = None

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        await self._vector.setup(f"hyde_{run_id}")
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        return await self._vector.ingest_session(session)

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._llm is None:
            self._llm = LLMClient()
        start = self._start_timer()
        # Step 1: hypothetical answer
        hyde_resp = await self._llm.generate(query=query, context="", system_prompt=_HYDE_SYSTEM)
        hypothetical = hyde_resp.text.strip()
        # Step 2: retrieve using the hypothetical answer as the query
        retrieval_start = time.perf_counter()
        v_result = await self._vector.recall(hypothetical, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 3: answer the *actual* question using the retrievals
        retrieved_block = "\n\n".join(
            f"[session={m.get('session_id', '?')}] {m.get('role', '')}: {m.get('content', '')}"
            for m in v_result.retrieved_memories
        )
        gen_start = time.perf_counter()
        final = await self._llm.generate(query, retrieved_block, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=final.text,
            supporting_session_ids=v_result.supporting_session_ids,
            supporting_turn_ids=v_result.supporting_turn_ids,
            retrieved_memories=v_result.retrieved_memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=hyde_resp.total_tokens + final.total_tokens + (v_result.tokens_used or 0),
            cost_usd=hyde_resp.cost_usd + final.cost_usd + (v_result.cost_usd or 0),
        )

    async def teardown(self) -> None:
        await self._vector.teardown()
        self._llm = None
