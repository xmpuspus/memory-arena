"""Tests for memory_arena.exceptions hierarchy."""

from __future__ import annotations

import pytest

from memory_arena.exceptions import (
    EvaluationError,
    GraphError,
    IngestError,
    KBArenaError,
    LLMError,
    MemoryArenaError,
    MemorySystemError,
    StrategyError,
)


class TestExceptionHierarchy:
    def test_base_exception_inheritance(self):
        assert issubclass(MemoryArenaError, Exception)

    def test_ingest_error_subclass(self):
        assert issubclass(IngestError, MemoryArenaError)

    def test_graph_error_subclass(self):
        assert issubclass(GraphError, MemoryArenaError)

    def test_strategy_error_subclass(self):
        assert issubclass(StrategyError, MemoryArenaError)

    def test_memory_system_error_subclass(self):
        assert issubclass(MemorySystemError, MemoryArenaError)

    def test_evaluation_error_subclass(self):
        assert issubclass(EvaluationError, MemoryArenaError)

    def test_llm_error_subclass(self):
        assert issubclass(LLMError, MemoryArenaError)

    def test_kbarena_error_alias(self):
        assert KBArenaError is MemoryArenaError


class TestRaiseAndCatch:
    def test_raise_strategy_error(self):
        with pytest.raises(StrategyError):
            raise StrategyError("setup failed")

    def test_raise_ingest_error(self):
        with pytest.raises(IngestError):
            raise IngestError("bad jsonl")

    def test_caught_by_base(self):
        with pytest.raises(MemoryArenaError):
            raise StrategyError("x")

    def test_message_preserved(self):
        try:
            raise EvaluationError("judge timed out")
        except EvaluationError as exc:
            assert "judge timed out" in str(exc)

    def test_chaining(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise LLMError("outer") from inner
        except LLMError as outer:
            assert outer.__cause__ is not None
            assert isinstance(outer.__cause__, ValueError)
