"""Verify vendor strategies surface SDK errors instead of swallowing them.

The v0.1.5 audit found ~22 silent ``except: pass`` blocks across the vendor
strategies that could silently turn benchmark numbers into "swallowed SDK
errors disguised as accuracy". These tests assert the new ``self._errors``
accumulator + the runner's ``out['errors']`` wiring catches at least one
representative case per concern.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_arena.llm.client import LLMResponse


def _mock_llm():
    fake = AsyncMock()
    fake.generate = AsyncMock(
        return_value=LLMResponse(text="ok", input_tokens=1, output_tokens=1, cost_usd=0.0)
    )
    fake.classify = AsyncMock(return_value="answer")
    return fake


class TestStrategyErrorAccumulator:
    """Strategies must record vendor exceptions on self._errors."""

    @pytest.mark.asyncio
    async def test_mem0_records_search_error(self, monkeypatch, sample_session):
        from memory_arena.strategies.mem0 import Mem0Strategy

        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": []}
        mock_memory.search.side_effect = RuntimeError("vendor 500")
        mock_memory.delete_all.return_value = None
        mock_memory_cls = MagicMock()
        mock_memory_cls.from_config.return_value = mock_memory
        fake_mem0 = SimpleNamespace(Memory=mock_memory_cls)
        monkeypatch.setitem(sys.modules, "mem0", fake_mem0)
        monkeypatch.setattr("memory_arena.strategies.mem0.LLMClient", lambda: _mock_llm())

        s = Mem0Strategy()
        await s.setup("run_test")
        await s.ingest_session(sample_session)
        await s.recall("Q?", top_k=3)

        assert len(s._errors) >= 1
        rec = s._errors[0]
        assert rec["phase"] == "recall"
        assert rec["type"] == "RuntimeError"
        assert "vendor 500" in rec["error"]


class TestRunnerSurfacesErrors:
    """The runner must copy strategy._errors into the result JSON's errors[]."""

    @pytest.mark.asyncio
    async def test_runner_drains_swallowed_errors(self, caplog):
        import logging

        from memory_arena.benchmark.runner import _run_strategy
        from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

        class FlakyStrategy(MemoryStrategy):
            name = "flaky"

            async def setup(self, run_id: str) -> None:
                self.run_id = run_id
                self._errors = [
                    {"phase": "setup", "step": "config", "error": "boom", "type": "RuntimeError"}
                ]

            async def ingest_session(self, session) -> IngestRecord:
                return IngestRecord(session_id=session.id, latency_ms=1.0)

            async def recall(self, query: str, top_k: int = 5) -> RecallResult:
                return RecallResult(answer="ok", strategy=self.name, latency_ms=1.0)

            async def teardown(self) -> None:
                self._errors.append(
                    {"phase": "teardown", "step": "close", "error": "x", "type": "RuntimeError"}
                )

        progress = MagicMock()
        progress.advance = MagicMock()
        with caplog.at_level(logging.WARNING, logger="memory_arena.benchmark.runner"):
            out = await _run_strategy(
                strategy=FlakyStrategy(),
                sessions=[],
                questions=[],
                run_id="r",
                top_k=3,
                llm=_mock_llm(),
                cost_cap=10.0,
                cumulative_cost={"flaky": 0.0},
                progress=progress,
                task_id=None,
            )

        assert out["swallowed_error_count"] == 2
        assert any(isinstance(e, dict) and e.get("phase") == "setup" for e in out["errors"])
        assert any(isinstance(e, dict) and e.get("phase") == "teardown" for e in out["errors"])
        assert any("swallowed errors" in r.message for r in caplog.records)
