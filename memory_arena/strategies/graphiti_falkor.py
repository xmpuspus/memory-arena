"""Graphiti on FalkorDB — same temporal-graph algorithm, Redis-based engine.

graphiti_falkor reuses GraphitiStrategy's ingest/recall verbatim and swaps only
the graph engine: Neo4j (bolt) -> FalkorDB (Redis protocol, GraphBLAS under the
hood). Isolating the database as the single variable lets the arena measure
FalkorDB's latency and cost against Neo4j for an identical memory architecture,
rather than trusting the vendor's headline latency claims.

Requires graphiti-core>=0.17 (first release with the FalkorDB driver) plus the
falkordb client. Install in its OWN environment, NOT alongside [graphiti], which
is pinned to 0.13 for baseline reproducibility:

    pip install 'memory-arena[falkordb]'

Bring up the engine on host port 6381 (6379/6380 are usually taken by other
local redis containers):

    docker compose up -d falkordb
"""

from __future__ import annotations

import logging

from memory_arena.llm.client import LLMClient
from memory_arena.settings import settings
from memory_arena.strategies.graphiti import GraphitiStrategy

logger = logging.getLogger(__name__)


def _sanitize_db_name(name: str) -> str:
    """FalkorDB graph names are Redis keys; keep them alphanumeric + underscore."""
    cleaned = "".join(c if c.isalnum() else "_" for c in name)
    return cleaned or "ma_default"


class GraphitiFalkorStrategy(GraphitiStrategy):
    name = "graphiti_falkor"

    def _falkor_db_name(self) -> str:
        return _sanitize_db_name(f"{settings.graphiti_group_prefix}_{self.run_id}")

    async def setup(self, run_id: str) -> None:
        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from graphiti_core.llm_client import OpenAIClient
            from graphiti_core.llm_client.config import LLMConfig
        except ImportError as exc:
            raise RuntimeError(
                "graphiti-core[falkordb]>=0.17 not installed. "
                "pip install 'memory-arena[falkordb]'"
            ) from exc
        self.run_id = run_id
        # Identical extraction config to GraphitiStrategy so the only variable
        # versus `graphiti` (Neo4j) is the graph engine. gpt-4o, 16k output cap;
        # the inherited ingest chunks each session into 2-turn episodes to stay
        # under the cap.
        llm_config = LLMConfig(
            api_key=settings.openai_api_key or None,
            model="gpt-4o",
            max_tokens=16384,
        )
        llm_client = OpenAIClient(config=llm_config, max_tokens=16384)
        driver = FalkorDriver(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password or None,
            database=self._falkor_db_name(),
        )
        self._client = Graphiti(graph_driver=driver, llm_client=llm_client)
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

    async def teardown(self) -> None:
        if self._client is None:
            return
        # Drop the per-run FalkorDB graph via the native client. This is robust
        # against graphiti driver-API churn between releases and is the natural
        # FalkorDB cleanup (one graph per run, deleted whole).
        try:
            from falkordb import FalkorDB

            db = FalkorDB(
                host=settings.falkordb_host,
                port=settings.falkordb_port,
                password=settings.falkordb_password or None,
            )
            db.select_graph(self._falkor_db_name()).delete()
        except Exception as exc:
            logger.warning("strategy=%s teardown graph delete failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "teardown",
                    "step": "graph_delete",
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


__strategy__ = GraphitiFalkorStrategy
