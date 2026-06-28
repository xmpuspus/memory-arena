"""Tests for memory_arena.benchmark.recall_lab."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_arena.benchmark.recall_lab import _run_one, run_recall_lab
from memory_arena.llm.client import LLMResponse


def _make_strategy(retrieved_session_ids):
    s = AsyncMock()
    s.name = "fake_strat"
    s.setup = AsyncMock()
    s.ingest_session = AsyncMock()
    s.recall = AsyncMock(
        return_value=MagicMock(
            supporting_session_ids=retrieved_session_ids,
            supporting_turn_ids=[],
            answer="x",
            cost_usd=0.0,
            tokens_used=0,
            latency_ms=10.0,
            retrieved_memories=[],
        )
    )
    s.teardown = AsyncMock()
    return s


class TestRunOne:
    @pytest.mark.asyncio
    async def test_runs_lifecycle(self, sample_session, sample_question):
        s = _make_strategy(["session_03"])
        out = await _run_one(s, [sample_session], [sample_question], run_id="r", top_k=3)
        assert out["strategy"] == "fake_strat"
        assert len(out["rows"]) == 1
        s.setup.assert_called_once()
        s.ingest_session.assert_called_once()
        s.recall.assert_called_once()
        s.teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_failure_short_circuits(self, sample_session, sample_question):
        s = _make_strategy(["session_03"])
        s.setup = AsyncMock(side_effect=RuntimeError("boom"))
        out = await _run_one(s, [sample_session], [sample_question], run_id="r", top_k=3)
        assert "error" in out
        s.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_failure_records_error_row(self, sample_session, sample_question):
        s = _make_strategy(["session_03"])
        s.recall = AsyncMock(side_effect=RuntimeError("recall boom"))
        out = await _run_one(s, [sample_session], [sample_question], run_id="r", top_k=3)
        assert any(r.get("error") for r in out["rows"])


class TestRunRecallLab:
    @pytest.mark.asyncio
    async def test_no_data_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = await run_recall_lab(corpus="missing", strategies="full_context")
        assert code == 1

    @pytest.mark.asyncio
    async def test_runs_smoke(self, tmp_path, monkeypatch, sample_session):
        # Build dataset
        base = tmp_path / "datasets" / "fake"
        (base / "processed").mkdir(parents=True)
        (base / "processed" / "sessions.jsonl").write_text(sample_session.model_dump_json() + "\n")
        smoke = base / "questions" / "smoke"
        smoke.mkdir(parents=True)
        (smoke / "smoke.yaml").write_text(
            "- id: q1\n"
            "  category: information_extraction\n"
            "  hops: 1\n"
            '  question: "Where does the user work?"\n'
            "  ground_truth:\n"
            "    answer: fintech\n"
            "    supporting_session_ids: [session_03]\n"
            "  constraints:\n"
            "    must_mention: []\n"
        )
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(
            return_value=LLMResponse(
                text="x [session_03]", input_tokens=1, output_tokens=1, cost_usd=0.0
            )
        )
        with patch("memory_arena.strategies.full_context.LLMClient", return_value=fake_llm):
            code = await run_recall_lab(corpus="fake", strategies="full_context", min_recall=0.0)
        assert code in (0, 1)
        # A result file was written
        produced = list(results_dir.glob("recall_lab_*.json"))
        assert len(produced) == 1
