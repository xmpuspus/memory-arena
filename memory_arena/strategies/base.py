"""Abstract base for the 7 memory strategies.

Lifecycle:
    setup(run_id) -> ingest_session(...) x N -> recall(query) x N -> teardown()

The benchmark runner only interacts via this interface. Strategies own their
own client/connection state and namespace by run_id so concurrent runs do not
contaminate each other.

A legacy `Strategy` alias is exported for code paths that still expect the
kb-arena ABC (chatbot API, arena engine). Memory-arena's runner uses
MemoryStrategy directly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class IngestRecord(BaseModel):
    """Per-session ingest measurement.

    cost_usd is Optional. None = the strategy can't measure cost (vendor SDK
    that hides its internal LLM usage). 0.0 = a real measurement of zero cost.
    """

    session_id: str
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float | None = 0.0
    facts_extracted: int = 0
    error: str = ""


class RecallResult(BaseModel):
    """Unified recall result from any memory strategy.

    recall_at_k_measurable=False means the strategy's data model doesn't carry
    chat-session pointers (LangMem stores extracted facts, Cognee stores triples,
    etc.), so Recall@k against session-level ground truth is not a fair metric.
    The runner records this so the dashboard can render "—" instead of 0%.

    cost_usd=None means the strategy can't measure its own cost — vendors like
    LangMem and Cognee run internal LLM calls (gpt-4o-mini) that don't go
    through memory-arena's accounting. Renders as "—".
    """

    answer: str
    supporting_session_ids: list[str] = Field(default_factory=list)
    supporting_turn_ids: list[str] = Field(default_factory=list)
    retrieved_memories: list[dict[str, Any]] = Field(default_factory=list)
    strategy: str = ""
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float | None = 0.0
    recall_at_k_measurable: bool = True
    mock: bool = False


class AnswerResult(BaseModel):
    """Legacy answer result kept for chatbot/arena compatibility."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    graph_context: Any | None = None
    retrieval: Any | None = None
    strategy: str = ""
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    mock: bool = False


class MemoryStrategy(ABC):
    """Abstract base for the 7 memory strategies."""

    name: str = "base"
    # Class-level signal: does this strategy's data model carry chat-session
    # pointers? Most retrievers do (they index session_ids), so the default is
    # True. Vendors that store extracted facts (LangMem, Cognee, Memori) or
    # graph triples without session_ids override this to False on the class.
    # The runner aggregates per-strategy and emits null for non-measurable
    # cases instead of attributing a structural-zero Recall@k to the vendor.
    recall_at_k_measurable: bool = True

    def __init__(self) -> None:
        self.run_id: str = ""
        self.last_supporting_session_ids: list[str] = []
        self.last_latency_ms: float = 0.0
        self.last_tokens_used: int = 0
        self.last_cost_usd: float = 0.0

    @abstractmethod
    async def setup(self, run_id: str) -> None:
        """Idempotent setup: create namespace, schema, etc."""

    @abstractmethod
    async def ingest_session(self, session: Any) -> IngestRecord:
        """Sequentially ingest one session of turns. Returns ingest cost+latency+tokens."""

    @abstractmethod
    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        """Recall + answer. Returns answer with supporting session ids and metrics."""

    async def update(self, fact: dict) -> None:
        """Optional: explicit fact update. Default no-op."""

    async def forget(self, scope: str) -> None:
        """Optional: scoped forget. Default no-op."""

    @abstractmethod
    async def teardown(self) -> None:
        """Drop run namespace. Critical for benchmark isolation."""

    async def stream_answer(
        self, question: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Stream answer tokens. Default: call recall() and yield full answer."""
        result = await self.recall(question)
        yield result.answer

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _record_metrics(
        self,
        start: float,
        tokens: int = 0,
        cost: float = 0.0,
        supporting_session_ids: list[str] | None = None,
    ) -> float:
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_latency_ms = elapsed_ms
        self.last_tokens_used = tokens
        self.last_cost_usd = cost
        self.last_supporting_session_ids = supporting_session_ids or []
        return elapsed_ms


# Legacy alias used by the chatbot/arena code paths still expecting the
# kb-arena interface. Memory-arena production code should import MemoryStrategy.
Strategy = MemoryStrategy
