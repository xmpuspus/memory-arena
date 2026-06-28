"""Tests for memory_arena.benchmark.runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from memory_arena.benchmark.runner import (
    _abstention_f1,
    _ratio,
    _resolve_strategies,
    run_memory_benchmark,
)
from memory_arena.llm.client import LLMResponse


class TestAggregations:
    def test_abstention_f1_perfect(self):
        calls = [
            {"expected": True, "actual": True},
            {"expected": True, "actual": True},
        ]
        assert _abstention_f1(calls) == 1.0

    def test_abstention_f1_all_wrong(self):
        calls = [
            {"expected": True, "actual": False},
            {"expected": True, "actual": False},
        ]
        assert _abstention_f1(calls) == 0.0

    def test_abstention_f1_empty(self):
        assert _abstention_f1([]) == 0.0

    def test_abstention_f1_partial(self):
        calls = [
            {"expected": True, "actual": True},  # tp
            {"expected": True, "actual": False},  # fn
            {"expected": False, "actual": True},  # fp
            {"expected": False, "actual": False},  # tn
        ]
        # precision = 1/2 = 0.5, recall = 1/2 = 0.5, f1 = 0.5
        assert abs(_abstention_f1(calls) - 0.5) < 1e-9

    def test_ratio_empty(self):
        assert _ratio([]) == 0.0

    def test_ratio_all_true(self):
        assert _ratio([True, True, True]) == 1.0

    def test_ratio_half(self):
        assert _ratio([True, False]) == 0.5


class TestResolveStrategies:
    def test_all_keyword(self):
        instances = _resolve_strategies("all")
        # baselines must be present (vendor strategies may fail to instantiate but they're stubs)
        names = [s.name for s in instances]
        assert "full_context" in names
        assert "recency_window" in names
        assert "naive_vector" in names

    def test_comma_list(self):
        instances = _resolve_strategies("full_context,recency_window")
        names = [s.name for s in instances]
        assert names == ["full_context", "recency_window"]

    def test_unknown_skipped(self):
        instances = _resolve_strategies("full_context,unknown_strategy")
        names = [s.name for s in instances]
        assert "full_context" in names
        assert "unknown_strategy" not in names


class TestRunMemoryBenchmarkSmoke:
    """End-to-end (mocked LLM) smoke test of run_memory_benchmark."""

    @pytest.mark.asyncio
    async def test_runs_with_baselines(
        self, tmp_path, monkeypatch, sample_session, sample_question
    ):
        # Build a fake dataset
        base = tmp_path / "datasets" / "fake"
        (base / "processed").mkdir(parents=True)
        (base / "processed" / "sessions.jsonl").write_text(sample_session.model_dump_json() + "\n")
        smoke = base / "questions" / "smoke"
        smoke.mkdir(parents=True)
        (smoke / "smoke.yaml").write_text(
            "- id: q1\n"
            "  category: information_extraction\n"
            "  hops: 1\n"
            "  question: What does the user do?\n"
            "  ground_truth:\n"
            "    answer: Software engineer\n"
            "    supporting_session_ids: [session_03]\n"
            "  constraints:\n"
            "    must_mention: [engineer]\n"
        )
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        # Patch LLMClient everywhere
        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(
            return_value=LLMResponse(
                text="The user is a software engineer at a fintech startup [session_03].",
                input_tokens=80,
                output_tokens=20,
                cost_usd=0.0003,
            )
        )
        fake_llm.classify = AsyncMock(return_value="NO")
        fake_llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 80, "completeness": 70, "rationale": "ok"}',
                input_tokens=50,
                output_tokens=10,
                cost_usd=0.0001,
            )
        )

        with patch("memory_arena.benchmark.runner.LLMClient", return_value=fake_llm):
            with patch("memory_arena.strategies.full_context.LLMClient", return_value=fake_llm):
                with patch(
                    "memory_arena.strategies.recency_window.LLMClient", return_value=fake_llm
                ):
                    await run_memory_benchmark(
                        corpus="fake",
                        strategy="full_context,recency_window",
                        questions="smoke",
                        cost_cap=10.0,
                        top_k=3,
                    )

        # Result files should exist
        produced = list(results_dir.glob("fake_*.json"))
        assert len(produced) >= 2
        # Each should have a strategy key
        for p in produced:
            data = json.loads(p.read_text())
            assert "strategy" in data
            assert "accuracy" in data
            assert "abstention_f1" in data
