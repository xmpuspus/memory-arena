"""Unit tests for HippoRAG2Strategy.

Mocks LLMClient + OpenAI embeddings. Covers:
- setup creates a fresh ChromaDB collection scoped to run_id
- ingest_session adds passages + extracts triples + builds graph
- synonym edges added between near-duplicate nodes
- recall runs personalized PageRank and returns top_k passages
- LLM_RERANK toggle produces a different order when enabled
- teardown cleans up
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_arena.llm.client import LLMResponse
from memory_arena.sessions.schema import Session, Turn
from memory_arena.strategies.base import IngestRecord, RecallResult
from memory_arena.strategies.hipporag2 import (
    HippoRAG2Strategy,
    _cosine,
    _normalize_entity,
)


def _make_session(sid: str, content: str) -> Session:
    return Session(
        id=sid,
        user_id="user_42",
        timestamp="2026-03-12T10:00:00Z",
        turns=[
            Turn(
                id=f"{sid}_t1",
                session_id=sid,
                role="user",
                content=content,
                timestamp="2026-03-12T10:00:00Z",
            )
        ],
    )


def _embed_for(text: str) -> list[float]:
    """Deterministic embed mock. Each token gets a fixed slot in a tiny vector."""
    tokens = text.lower().split()
    # 12-dim "lexical" embedding: 1.0 where the token appears, else 0.
    keys = [
        "user",
        "software",
        "engineer",
        "fintech",
        "developer",
        "manila",
        "tokyo",
        "city",
        "fraud",
        "detection",
        "pipeline",
        "favourite",
    ]
    return [1.0 if k in tokens else 0.0 for k in keys]


class FakeEmbeddingFunction:
    """Stand-in for OpenAIEmbedding that's deterministic and cheap."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, inputs):  # noqa: D401
        return [_embed_for(t) for t in inputs]


def _fake_collection_pair():
    """Build a (client, collection) pair that records upserts and supports get."""
    upserts: dict[str, str] = {}

    collection = MagicMock()

    def _upsert(*, ids, documents, metadatas=None):
        for i, doc in zip(ids, documents, strict=False):
            upserts[i] = doc

    def _get(*, ids, include=None):
        return {"documents": [upserts.get(i, "") for i in ids]}

    def _query(*, query_texts, n_results, include=None):
        ids = list(upserts.keys())[:n_results]
        return {
            "ids": [ids],
            "documents": [[upserts.get(i, "") for i in ids]],
            "metadatas": [[{"session_id": i.removeprefix("p_")} for i in ids]],
            "distances": [[0.1 for _ in ids]],
        }

    collection.upsert.side_effect = _upsert
    collection.get.side_effect = _get
    collection.query.side_effect = _query

    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.delete_collection.return_value = None
    return client, collection, upserts


@pytest.fixture
def patched_strategy(monkeypatch):
    """HippoRAG2 with Chroma + embeddings patched to in-memory fakes."""
    client, collection, upserts = _fake_collection_pair()

    class _Persistent:
        def __init__(self, *args, **kwargs):
            pass

        def __new__(cls, *args, **kwargs):
            return client

    import chromadb

    monkeypatch.setattr(chromadb, "PersistentClient", _Persistent)
    monkeypatch.setattr(
        "memory_arena.strategies.hipporag2.HippoRAG2Strategy._get_embedder",
        lambda self: FakeEmbeddingFunction(),
    )
    # Also patch the OpenAIEmbedding referenced inside _get_collection.
    monkeypatch.setattr(
        "memory_arena.strategies.embeddings.OpenAIEmbedding",
        FakeEmbeddingFunction,
    )
    return collection, upserts


@pytest.mark.asyncio
async def test_setup_creates_collection(monkeypatch, patched_strategy):
    collection, _ = patched_strategy
    s = HippoRAG2Strategy()
    # Need a no-op LLMClient for setup.
    monkeypatch.setattr("memory_arena.strategies.hipporag2.LLMClient", lambda: AsyncMock())
    await s.setup("run-hipporag-test")
    assert s.run_id == "run-hipporag-test"
    assert s._collection_name() == "hipporag2_run-hipporag-test"
    assert s._graph.number_of_nodes() == 0


@pytest.mark.asyncio
async def test_ingest_extracts_triples(monkeypatch, patched_strategy):
    collection, upserts = patched_strategy

    # Two ingest sessions: first extracts (user, works_at, fintech); second extracts
    # (user, lives_in, manila).
    openie_resp_a = LLMResponse(
        text='[{"s": "user", "p": "works_at", "o": "fintech"}, '
        '{"s": "user", "p": "is_a", "o": "software engineer"}]',
        input_tokens=80,
        output_tokens=30,
        cost_usd=0.0002,
    )
    openie_resp_b = LLMResponse(
        text='[{"s": "user", "p": "lives_in", "o": "manila"}]',
        input_tokens=50,
        output_tokens=15,
        cost_usd=0.0001,
    )

    fake_llm = AsyncMock()
    fake_llm._call = AsyncMock(side_effect=[openie_resp_a, openie_resp_b])
    monkeypatch.setattr("memory_arena.strategies.hipporag2.LLMClient", lambda: fake_llm)

    s = HippoRAG2Strategy()
    await s.setup("run-ingest")

    rec_a = await s.ingest_session(_make_session("sess_a", "user is software engineer at fintech"))
    rec_b = await s.ingest_session(_make_session("sess_b", "user lives in manila"))

    assert isinstance(rec_a, IngestRecord)
    assert rec_a.facts_extracted == 2
    assert rec_b.facts_extracted == 1
    assert "p_sess_a" in upserts and "p_sess_b" in upserts
    # Graph picks up entities
    assert s._graph.has_node("user")
    assert s._graph.has_node("fintech")
    assert s._graph.has_node("manila")
    # Edges carry predicates
    assert s._graph.has_edge("user", "manila")
    assert s._graph["user"]["manila"]["predicate"] == "lives_in"


@pytest.mark.asyncio
async def test_synonym_edges_between_near_duplicates(monkeypatch, patched_strategy):
    """Two entity phrases with shared tokens should get a synonym edge."""
    openie_a = LLMResponse(
        text='[{"s": "software engineer", "p": "at", "o": "fintech"}]',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )
    openie_b = LLMResponse(
        text='[{"s": "software developer", "p": "at", "o": "fintech"}]',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )

    fake_llm = AsyncMock()
    fake_llm._call = AsyncMock(side_effect=[openie_a, openie_b])
    fake_llm.generate = AsyncMock(
        return_value=LLMResponse(text="ans", input_tokens=1, output_tokens=1, cost_usd=0.0)
    )
    monkeypatch.setattr("memory_arena.strategies.hipporag2.LLMClient", lambda: fake_llm)

    s = HippoRAG2Strategy()
    # Lower threshold so the fake embedding "software engineer"/"software developer"
    # (sharing one token slot out of two) clears it.
    s.SYNONYM_THRESHOLD = 0.4
    await s.setup("run-syn")
    await s.ingest_session(_make_session("sa", "engineer text"))
    await s.ingest_session(_make_session("sb", "developer text"))

    # Calling recall will trigger _ensure_node_embeddings + synonym refresh.
    await s.recall("anything", top_k=1)
    has_syn = any(d.get("predicate") == "synonym" for _, _, d in s._graph.edges(data=True))
    assert has_syn, "expected at least one synonym edge between similar entities"


@pytest.mark.asyncio
async def test_recall_runs_pagerank_and_returns_top_k(monkeypatch, patched_strategy):
    openie_a = LLMResponse(
        text='[{"s": "user", "p": "works_at", "o": "fintech"}]',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )
    openie_b = LLMResponse(
        text='[{"s": "user", "p": "lives_in", "o": "manila"}]',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )
    gen_resp = LLMResponse(
        text="The user works at a fintech [sess_a].",
        input_tokens=50,
        output_tokens=10,
        cost_usd=0.0003,
    )

    fake_llm = AsyncMock()
    fake_llm._call = AsyncMock(side_effect=[openie_a, openie_b])
    fake_llm.generate = AsyncMock(return_value=gen_resp)
    monkeypatch.setattr("memory_arena.strategies.hipporag2.LLMClient", lambda: fake_llm)

    s = HippoRAG2Strategy()
    await s.setup("run-recall")
    await s.ingest_session(_make_session("sess_a", "user fintech engineer"))
    await s.ingest_session(_make_session("sess_b", "user manila city"))

    result = await s.recall("Where does the user work?", top_k=2)
    assert isinstance(result, RecallResult)
    assert result.strategy == "hipporag2"
    # Query embedding overlaps with the fintech passage's entity "fintech"; the
    # personalized PageRank should pull sess_a first.
    assert "sess_a" in result.supporting_session_ids
    assert len(result.retrieved_memories) <= 2


@pytest.mark.asyncio
async def test_llm_rerank_changes_order(monkeypatch, patched_strategy):
    """With LLM_RERANK enabled, the rerank prompt should drive a new order."""
    openie_a = LLMResponse(
        text='[{"s": "user", "p": "works_at", "o": "fintech"}]',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )
    openie_b = LLMResponse(
        text='[{"s": "user", "p": "lives_in", "o": "manila"}]',
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )
    # Rerank response forces sess_b first.
    rerank_resp = LLMResponse(
        text='["p_sess_b", "p_sess_a"]', input_tokens=20, output_tokens=10, cost_usd=0.0
    )
    answer_resp = LLMResponse(text="answer", input_tokens=10, output_tokens=5, cost_usd=0.0)

    fake_llm = AsyncMock()
    fake_llm._call = AsyncMock(side_effect=[openie_a, openie_b])
    fake_llm.generate = AsyncMock(side_effect=[rerank_resp, answer_resp])
    monkeypatch.setattr("memory_arena.strategies.hipporag2.LLMClient", lambda: fake_llm)

    s = HippoRAG2Strategy()
    s.LLM_RERANK = True
    await s.setup("run-rerank")
    await s.ingest_session(_make_session("sess_a", "user fintech"))
    await s.ingest_session(_make_session("sess_b", "user manila"))

    result = await s.recall("any question", top_k=2)
    # Reranked order has sess_b before sess_a.
    assert result.supporting_session_ids[0] == "sess_b"


@pytest.mark.asyncio
async def test_teardown_clears_state(monkeypatch, patched_strategy):
    collection, _ = patched_strategy
    fake_llm = AsyncMock()
    fake_llm._call = AsyncMock(
        return_value=LLMResponse(
            text='[{"s": "user", "p": "is", "o": "engineer"}]',
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
        )
    )
    monkeypatch.setattr("memory_arena.strategies.hipporag2.LLMClient", lambda: fake_llm)
    s = HippoRAG2Strategy()
    await s.setup("run-teardown")
    await s.ingest_session(_make_session("sa", "user is engineer"))
    assert s._graph.number_of_nodes() > 0
    assert s._passages
    await s.teardown()
    assert s._graph.number_of_nodes() == 0
    assert s._passages == {}
    assert s._node_embeddings == {}
    assert s._llm is None


def test_cosine_handles_zero_vectors():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_accepts_numpy_arrays():
    # The embedder returns numpy arrays for the query vector, while node
    # embeddings are stored as lists. ``not array`` raises "truth value of an
    # array is ambiguous", which previously killed every hipporag2 recall.
    import numpy as np

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(_cosine(a, b) - 1.0) < 1e-6
    assert _cosine(np.array([]), b) == 0.0
    assert _cosine(a, np.array([1.0, 2.0, 3.0])) == 0.0


def test_normalize_entity_lowercases_and_caps_length():
    assert _normalize_entity("  Hello   World  ") == "hello world"
    assert _normalize_entity(None) == ""
    long = "x" * 200
    assert len(_normalize_entity(long)) == 80
