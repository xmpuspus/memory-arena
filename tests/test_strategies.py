"""Tests for memory_arena.strategies (registry, base, baselines)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_arena.llm.client import LLMResponse
from memory_arena.strategies import STRATEGY_REGISTRY, get_strategy
from memory_arena.strategies.base import (
    AnswerResult,
    IngestRecord,
    MemoryStrategy,
    RecallResult,
)
from memory_arena.strategies.full_context import FullContextStrategy
from memory_arena.strategies.recency_window import RecencyWindowStrategy


def _patch_llm(monkeypatch, response_text: str = "Software engineer at fintech [session_03]."):
    fake = AsyncMock()
    fake.generate = AsyncMock(
        return_value=LLMResponse(
            text=response_text, input_tokens=80, output_tokens=20, cost_usd=0.0003
        )
    )

    def factory(*args, **kwargs):
        return fake

    monkeypatch.setattr("memory_arena.strategies.full_context.LLMClient", factory)
    monkeypatch.setattr("memory_arena.strategies.recency_window.LLMClient", factory)
    monkeypatch.setattr("memory_arena.strategies.naive_vector.LLMClient", factory)
    return fake


class TestRegistry:
    def test_seven_strategies_present(self):
        assert "full_context" in STRATEGY_REGISTRY
        assert "recency_window" in STRATEGY_REGISTRY
        assert "naive_vector" in STRATEGY_REGISTRY
        assert "mem0" in STRATEGY_REGISTRY
        assert "mem0g" in STRATEGY_REGISTRY
        assert "graphiti" in STRATEGY_REGISTRY

    def test_get_strategy_by_name(self):
        s = get_strategy("full_context")
        assert isinstance(s, FullContextStrategy)
        assert s.name == "full_context"

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            get_strategy("nonexistent_strategy")


class TestMemoryStrategyABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MemoryStrategy()

    def test_record_metrics(self):
        s = FullContextStrategy()
        start = s._start_timer()
        elapsed = s._record_metrics(start, tokens=10, cost=0.01, supporting_session_ids=["s1"])
        assert elapsed >= 0
        assert s.last_tokens_used == 10
        assert s.last_cost_usd == 0.01
        assert s.last_supporting_session_ids == ["s1"]


class TestFullContext:
    @pytest.mark.asyncio
    async def test_setup_initializes(self, monkeypatch):
        _patch_llm(monkeypatch)
        s = FullContextStrategy()
        await s.setup("run-123")
        assert s.run_id == "run-123"
        assert s._sessions == []

    @pytest.mark.asyncio
    async def test_ingest_appends_session(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch)
        s = FullContextStrategy()
        await s.setup("run-1")
        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id
        assert s._sessions == [sample_session]

    @pytest.mark.asyncio
    async def test_recall_returns_answer(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch, "Engineer at fintech [session_03].")
        s = FullContextStrategy()
        await s.setup("run-1")
        await s.ingest_session(sample_session)
        result = await s.recall("What does the user do?")
        assert isinstance(result, RecallResult)
        assert "fintech" in result.answer
        assert "session_03" in result.supporting_session_ids
        assert result.strategy == "full_context"

    @pytest.mark.asyncio
    async def test_teardown_clears(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch)
        s = FullContextStrategy()
        await s.setup("run-1")
        await s.ingest_session(sample_session)
        await s.teardown()
        assert s._sessions == []
        assert s._llm is None

    @pytest.mark.asyncio
    async def test_recall_respects_token_budget(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch)
        s = FullContextStrategy()
        await s.setup("run-1")
        await s.ingest_session(sample_session)
        # Override budget to a tiny value to confirm clamping works
        from memory_arena.settings import settings as _s

        old = _s.full_context_token_budget
        _s.full_context_token_budget = 5
        try:
            result = await s.recall("?")
            assert "session_03" in result.supporting_session_ids
        finally:
            _s.full_context_token_budget = old


class TestRecencyWindow:
    @pytest.mark.asyncio
    async def test_setup(self, monkeypatch):
        _patch_llm(monkeypatch)
        s = RecencyWindowStrategy()
        await s.setup("r1")
        assert s.run_id == "r1"
        assert s._turns == []

    @pytest.mark.asyncio
    async def test_ingest_adds_turns(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch)
        s = RecencyWindowStrategy()
        await s.setup("r1")
        await s.ingest_session(sample_session)
        assert len(s._turns) == 3

    @pytest.mark.asyncio
    async def test_recall_uses_window(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch, "Software engineer [session_03]")
        s = RecencyWindowStrategy()
        await s.setup("r1")
        await s.ingest_session(sample_session)
        result = await s.recall("?")
        assert isinstance(result, RecallResult)
        assert result.supporting_session_ids == ["session_03"]
        assert len(result.supporting_turn_ids) <= 3

    @pytest.mark.asyncio
    async def test_window_caps_total_turns(self, monkeypatch, sample_sessions):
        from memory_arena.settings import settings as _s

        _patch_llm(monkeypatch)
        s = RecencyWindowStrategy()
        await s.setup("r1")
        for sess in sample_sessions:
            await s.ingest_session(sess)
        old = _s.recency_window_n
        _s.recency_window_n = 2
        try:
            result = await s.recall("?")
            assert len(result.supporting_turn_ids) == 2
        finally:
            _s.recency_window_n = old

    @pytest.mark.asyncio
    async def test_teardown(self, monkeypatch, sample_session):
        _patch_llm(monkeypatch)
        s = RecencyWindowStrategy()
        await s.setup("r1")
        await s.ingest_session(sample_session)
        await s.teardown()
        assert s._turns == []


class TestNaiveVectorStrategyShape:
    """We don't hit a real ChromaDB — just verify lifecycle calls."""

    @pytest.mark.asyncio
    async def test_setup_and_teardown_run(self, monkeypatch, mock_chroma_client):
        _patch_llm(monkeypatch)
        from memory_arena.strategies.naive_vector import NaiveVectorStrategy

        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = NaiveVectorStrategy()
                await s.setup("run-naive")
                assert s.run_id == "run-naive"
                await s.teardown()

    @pytest.mark.asyncio
    async def test_ingest_session_calls_upsert(
        self, monkeypatch, mock_chroma_client, sample_session
    ):
        _patch_llm(monkeypatch)
        from memory_arena.strategies.naive_vector import NaiveVectorStrategy

        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = NaiveVectorStrategy()
                await s.setup("run-naive")
                rec = await s.ingest_session(sample_session)
                assert rec.session_id == sample_session.id
                # upsert was called
                collection = mock_chroma_client.get_or_create_collection.return_value
                collection.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_recall_returns_supporting_ids(
        self, monkeypatch, mock_chroma_client, sample_session
    ):
        _patch_llm(monkeypatch)
        from memory_arena.strategies.naive_vector import NaiveVectorStrategy

        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = NaiveVectorStrategy()
                await s.setup("run-naive")
                await s.ingest_session(sample_session)
                result = await s.recall("What kind of work?")
                assert "session_03" in result.supporting_session_ids
                assert "session_03_turn_001" in result.supporting_turn_ids


class TestVendorStrategiesGracefulFailure:
    """Vendor strategies should raise informative errors when their SDKs are absent."""

    @pytest.mark.asyncio
    async def test_mem0_missing_sdk(self, monkeypatch):
        # Force ImportError
        import sys

        from memory_arena.strategies.mem0 import Mem0Strategy

        monkeypatch.setitem(sys.modules, "mem0", None)
        s = Mem0Strategy()
        # The fast path imports inside setup; on systems without mem0, this raises RuntimeError.
        # On systems WITH mem0 installed, this would succeed - so we test the import-error code path
        # by patching the import directly.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mem0" or name.startswith("mem0."):
                raise ImportError("mem0 not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(RuntimeError) as ei:
            await s.setup("r")
        assert "mem0ai not installed" in str(ei.value)

    @pytest.mark.asyncio
    async def test_graphiti_missing_sdk(self, monkeypatch):
        import builtins

        from memory_arena.strategies.graphiti import GraphitiStrategy

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "graphiti_core" or name.startswith("graphiti_core."):
                raise ImportError("graphiti not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        s = GraphitiStrategy()
        with pytest.raises(RuntimeError) as ei:
            await s.setup("r")
        assert "graphiti-core" in str(ei.value)


class TestAnswerResultLegacyAlias:
    def test_legacy_class_exists(self):
        ans = AnswerResult(answer="hello")
        assert ans.answer == "hello"
        assert ans.cost_usd == 0.0
