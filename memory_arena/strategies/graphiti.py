"""Graphiti strategy — Zep's graph memory with temporal edges.

Pinned to graphiti-core==0.13.0. Reuses the docker-compose Neo4j instance.
Each session ingests as one episode (source=message). group_id namespaces by run.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

logger = logging.getLogger(__name__)

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a temporal "
    "knowledge graph. Each fact has a valid_at timestamp. Use only the retrieved facts. "
    "Cite sessions in square brackets. If the facts do not contain the answer, say "
    '"I do not have that information."'
)


class GraphitiStrategy(MemoryStrategy):
    name = "graphiti"

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._llm: LLMClient | None = None
        self._errors: list[dict] = []

    def _group_id(self) -> str:
        return f"{settings.graphiti_group_prefix}_{self.run_id}"

    async def setup(self, run_id: str) -> None:
        try:
            from graphiti_core import Graphiti
            from graphiti_core.llm_client import OpenAIClient
            from graphiti_core.llm_client.config import LLMConfig
        except ImportError as exc:
            raise RuntimeError(
                "graphiti-core not installed. pip install 'memory-arena[graphiti]'"
            ) from exc
        self.run_id = run_id
        # Graphiti default LLM config has max_tokens=8192; large LongMemEval
        # episodes overflow this during entity extraction. Use gpt-4o with 16k
        # output cap. Combined with per-turn-pair episode chunking below this
        # keeps each extraction call comfortably under the limit.
        llm_config = LLMConfig(
            api_key=settings.openai_api_key or None,
            model="gpt-4o",
            max_tokens=16384,
        )
        llm_client = OpenAIClient(config=llm_config, max_tokens=16384)
        self._client = Graphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            llm_client=llm_client,
        )
        self._errors = []
        try:
            await self._client.build_indices_and_constraints()
        except Exception as exc:
            logger.warning("strategy=%s setup build_indices failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "setup",
                    "step": "build_indices_and_constraints",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._client is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        try:
            from datetime import timedelta

            from graphiti_core.nodes import EpisodeType
        except Exception as exc:
            elapsed = (self._start_timer() - start) * 1000
            return IngestRecord(session_id=session.id, latency_ms=elapsed, error=str(exc))

        # Chunk the session into 2-turn (user+assistant) episodes so each
        # entity-extraction call stays well under the LLM output cap.
        ref_time = self._parse_ts(session.timestamp)
        chunks: list[str] = []
        buf: list[str] = []
        for turn in session.turns:
            buf.append(f"{turn.role}: {turn.content}")
            if len(buf) >= 2:
                chunks.append("\n".join(buf))
                buf = []
        if buf:
            chunks.append("\n".join(buf))

        successes = 0
        first_err = ""
        for i, body in enumerate(chunks):
            episode_time = ref_time + timedelta(seconds=i)
            try:
                await self._client.add_episode(
                    name=f"{session.id}_part_{i:03d}",
                    episode_body=body,
                    source=EpisodeType.message,
                    source_description=f"chat session {session.id} part {i}",
                    reference_time=episode_time,
                    group_id=self._group_id(),
                )
                successes += 1
            except Exception as exc:
                if not first_err:
                    first_err = str(exc)
                logger.warning(
                    "strategy=%s ingest session=%s chunk=%d failed: %s",
                    self.name,
                    session.id,
                    i,
                    exc,
                )
                self._errors.append(
                    {
                        "phase": "ingest",
                        "session_id": session.id,
                        "chunk": i,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )
                # Continue: a single overflow shouldn't kill the whole session.
                continue

        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            facts_extracted=successes,
            error=first_err if successes == 0 else "",
        )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._client is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        retrieval_start = time.perf_counter()
        try:
            facts = await self._client.search(
                query=query,
                group_ids=[self._group_id()],
                num_results=top_k,
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
                answer=f"[ERROR] graphiti search failed: {exc}",
                strategy=self.name,
                latency_ms=(self._start_timer() - start) * 1000,
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        memories: list[dict] = []
        session_ids: list[str] = []
        for f in facts or []:
            fact_text = getattr(f, "fact", str(f))
            valid_at = getattr(f, "valid_at", None)
            invalid_at = getattr(f, "invalid_at", None)
            episode_uuids = getattr(f, "episodes", []) or []
            memories.append(
                {
                    "fact": fact_text,
                    "valid_at": str(valid_at) if valid_at else None,
                    "invalid_at": str(invalid_at) if invalid_at else None,
                    "episodes": list(episode_uuids),
                }
            )
        # Graphiti episode names embed the source session_id when we ingest
        # ("{session.id}_part_{i:03d}"). Pull session ids out of the episode UUIDs
        # if Graphiti exposes them. Otherwise leave session_ids empty + mark as
        # not-measurable so we don't structurally zero Recall@k.
        try:
            episode_node_uuids = []
            for f in facts or []:
                episode_node_uuids.extend(getattr(f, "episodes", []) or [])
            if episode_node_uuids and self._client is not None:
                # graphiti-core 0.13 exposes get_episode by uuid
                from graphiti_core.nodes import EpisodicNode

                for uuid in set(episode_node_uuids):
                    try:
                        node = await EpisodicNode.get_by_uuid(self._client.driver, uuid)
                        ep_name = getattr(node, "name", "")
                        # episode names are "{session.id}_part_{i:03d}"
                        if "_part_" in ep_name:
                            sid = ep_name.split("_part_")[0]
                            if sid and sid not in session_ids:
                                session_ids.append(sid)
                    except Exception as exc:
                        logger.warning(
                            "strategy=%s recall episode lookup uuid=%s failed: %s",
                            self.name,
                            uuid,
                            exc,
                        )
                        self._errors.append(
                            {
                                "phase": "recall",
                                "step": "episode_lookup",
                                "uuid": str(uuid),
                                "error": str(exc),
                                "type": type(exc).__name__,
                            }
                        )
                        continue
        except Exception as exc:
            logger.warning(
                "strategy=%s recall session-id resolution failed: %s",
                self.name,
                exc,
            )
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "session_id_resolution",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        recall_measurable = len(session_ids) > 0

        context = "\n".join(f"- {m['fact']} (valid_at={m.get('valid_at')})" for m in memories)

        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=resp.text,
            supporting_session_ids=session_ids,
            retrieved_memories=memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
            recall_at_k_measurable=recall_measurable,
        )

    async def teardown(self) -> None:
        if self._client is None:
            return
        try:
            driver = getattr(self._client, "driver", None)
            if driver is not None:
                async with driver.session() as session:
                    await session.run(
                        "MATCH (n) WHERE n.group_id = $g DETACH DELETE n",
                        g=self._group_id(),
                    )
        except Exception as exc:
            logger.warning("strategy=%s teardown DETACH DELETE failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "teardown",
                    "step": "detach_delete",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        try:
            close = getattr(self._client, "close", None)
            if close is not None:
                await close()
        except Exception as exc:
            logger.warning("strategy=%s teardown driver close failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "teardown",
                    "step": "close_driver",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        self._client = None
        self._llm = None

    @staticmethod
    def _parse_ts(ts: str | None) -> datetime:
        if not ts:
            return datetime.now(UTC)
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except ValueError:
            return datetime.now(UTC)
