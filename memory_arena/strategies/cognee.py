"""Cognee strategy - open-source agent memory with multi-graph backend.

Uses cognee's `add` + `cognify` (extracts entities/relationships into a graph)
and `search` with SearchType.INSIGHTS for retrieval. Air-gapped friendly:
defaults to local NetworkX storage when no graph DB is configured.

Pinned to cognee 1.x. Uses OPENAI_API_KEY for the underlying LLM.
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
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)


class CogneeStrategy(MemoryStrategy):
    name = "cognee"
    # Cognee's INSIGHTS / GRAPH_COMPLETION return entity triples and graph
    # paths that don't carry the original chat-session_id. Recall@k against
    # session-level ground truth isn't a fair metric here — the runner emits
    # null instead of attributing a structural-zero to the vendor.
    recall_at_k_measurable = False

    def __init__(self) -> None:
        super().__init__()
        self._llm: LLMClient | None = None
        self._dataset_name: str = ""
        self._cognee = None
        self._errors: list[dict] = []

    async def setup(self, run_id: str) -> None:
        try:
            import cognee
        except ImportError as exc:
            raise RuntimeError("cognee not installed. pip install 'memory-arena[cognee]'") from exc
        # Force-set env first (some cognee paths read directly from os.environ),
        # then push the same values through cognee's config setters which
        # override any pydantic-settings cache that loaded the .env file.
        if settings.openai_api_key:
            os.environ["LLM_API_KEY"] = settings.openai_api_key
            os.environ["EMBEDDING_API_KEY"] = settings.openai_api_key
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        # cognee cannot be leveled to Claude in the shared env: cognee 1.0.3
        # pulls a newer starlette than the project's pinned fastapi allows, so it
        # only runs in its own venv (the graphiti_falkor pattern). Kept at its
        # benchmarked vendor default (gpt-4o-mini); see the leveling note.
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-4o-mini"
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        # Hold the embedder constant across strategies via settings instead of
        # a hardcoded -small. Pinning both the model name and the matching
        # dimensions so cognee's vector index aligns with the rest of the
        # leaderboard (naive_vector / hyde / persona_profile / reflection).
        os.environ["EMBEDDING_MODEL"] = settings.embedding_model
        if settings.openai_api_key:
            try:
                cognee.config.set_llm_provider("openai")
                cognee.config.set_llm_model("gpt-4o-mini")
                cognee.config.set_llm_api_key(settings.openai_api_key)
                cognee.config.set_embedding_dimensions(settings.embedding_dimensions)
            except Exception as exc:
                logger.warning("strategy=%s setup config-set failed: %s", self.name, exc)
                self._errors.append(
                    {
                        "phase": "setup",
                        "step": "config_set",
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )

        self.run_id = run_id
        self._dataset_name = f"memory_arena_{run_id}"
        self._cognee = cognee
        self._llm = LLMClient()
        self._errors = []
        # Reset any prior state for this dataset (best effort).
        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        except Exception as exc:
            logger.warning("strategy=%s setup prune failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "setup",
                    "step": "prune",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._cognee is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        # Format the session as a single document so cognify can extract from it.
        body_lines = [f"Session {session.id} ({session.timestamp or 'unknown'}):"]
        for turn in session.turns:
            body_lines.append(f"{turn.role}: {turn.content}")
        body = "\n".join(body_lines)
        err = ""
        try:
            await self._cognee.add(body, dataset_name=self._dataset_name)
        except Exception as exc:
            err = f"add: {exc}"
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
            facts_extracted=len(session.turns),
            error=err,
        )

    async def _cognify_once(self) -> None:
        if self._cognee is None:
            return
        try:
            await self._cognee.cognify([self._dataset_name])
        except Exception as exc:
            logger.warning("strategy=%s cognify failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "cognify",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._cognee is None:
            raise RuntimeError("setup() not called")
        # Cognify all collected sessions into the graph on the first recall.
        if not getattr(self, "_cognified", False):
            await self._cognify_once()
            self._cognified = True

        start = self._start_timer()
        retrieval_start = time.perf_counter()
        memories: list[dict] = []
        try:
            from cognee.modules.search.types import SearchType

            # Cognee 1.x renamed INSIGHTS; GRAPH_COMPLETION returns an LLM
            # answer grounded in the knowledge graph. CHUNKS gives raw triples.
            search_type = getattr(SearchType, "GRAPH_COMPLETION", None) or SearchType.CHUNKS
            results = await self._cognee.search(
                query_type=search_type,
                query_text=query,
                datasets=[self._dataset_name],
            )
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
                answer=f"[ERROR] cognee search failed: {exc}",
                strategy=self.name,
                latency_ms=(self._start_timer() - start) * 1000,
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        for r in results or []:
            memories.append({"content": str(r)})

        # Cognee's INSIGHTS return triples and don't carry session_ids; we leave
        # supporting_session_ids empty rather than fabricate them.
        context = "\n\n---\n\n".join(m["content"] for m in memories[: top_k * 2])
        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        # recall_at_k_measurable is declared False on the class; the runner
        # consults the class attribute and renders "—" for this strategy.
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
        if self._cognee is None:
            return
        try:
            await self._cognee.prune.prune_data()
            await self._cognee.prune.prune_system(metadata=True)
        except Exception as exc:
            logger.warning("strategy=%s teardown prune failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "teardown",
                    "step": "prune",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        self._cognee = None
        self._llm = None
