"""Tests for the 7-axis evaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from memory_arena.benchmark.evaluator import (
    MemoryScore,
    _classify_abstention,
    _evaluate_sources,
    _evaluate_structural,
    _quick_abstain_match,
    _temporal_overlap,
    clear_eval_cache,
    evaluate_memory_answer,
)
from memory_arena.llm.client import LLMResponse
from memory_arena.sessions.schema import Constraints, GroundTruth


class TestStructural:
    def test_must_mention_present(self):
        c = Constraints(must_mention=["fintech", "engineer"])
        ok, fails = _evaluate_structural("I am a software engineer at a fintech startup.", c)
        assert ok is True
        assert fails == []

    def test_must_mention_missing(self):
        c = Constraints(must_mention=["fintech"])
        ok, fails = _evaluate_structural("I am a teacher.", c)
        assert ok is False
        assert any("fintech" in f for f in fails)

    def test_must_not_claim_present(self):
        c = Constraints(must_not_claim=["currently unemployed"])
        ok, fails = _evaluate_structural("The user is currently unemployed.", c)
        assert ok is False
        assert any("currently unemployed" in f for f in fails)

    def test_max_tokens_within(self):
        c = Constraints(max_tokens=100)
        ok, fails = _evaluate_structural("short answer", c)
        assert ok is True

    def test_max_tokens_exceeded(self):
        c = Constraints(max_tokens=10)
        ok, fails = _evaluate_structural("word " * 200, c)
        assert ok is False
        assert any("max_tokens" in f for f in fails)

    def test_case_insensitive(self):
        c = Constraints(must_mention=["FINTECH"])
        ok, _ = _evaluate_structural("a fintech startup", c)
        assert ok is True


class TestSources:
    def test_intersect(self):
        gt = GroundTruth(answer="x", supporting_session_ids=["s1", "s2"])
        assert _evaluate_sources("a", ["s1"], gt) is True

    def test_no_overlap(self):
        gt = GroundTruth(answer="x", supporting_session_ids=["s1"])
        assert _evaluate_sources("a", ["s2"], gt) is False

    def test_empty_expected_passes(self):
        gt = GroundTruth(answer="x", supporting_session_ids=[])
        assert _evaluate_sources("a", [], gt) is True


class TestAbstention:
    def test_quick_match_positive(self):
        assert _quick_abstain_match("I do not have that information.")
        assert _quick_abstain_match("I don't know")
        assert _quick_abstain_match("Cannot answer")

    def test_quick_match_negative(self):
        assert not _quick_abstain_match("The user is a software engineer.")

    @pytest.mark.asyncio
    async def test_classify_uses_quick_path(self):
        llm = AsyncMock()
        result, _, _ = await _classify_abstention("I don't know.", llm)
        assert result is True
        llm.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_falls_through_to_llm(self):
        llm = AsyncMock()
        llm.classify = AsyncMock(return_value="YES")
        result, _, _ = await _classify_abstention("ambiguous reply", llm)
        assert result is True


class TestTemporalOverlap:
    def test_year_match(self):
        assert _temporal_overlap("in 2026", "2026-03-15") is True

    def test_no_match(self):
        assert _temporal_overlap("last summer", "2026-03-15") is False

    def test_empty_expected(self):
        assert _temporal_overlap("2026", "") is False


class TestEvaluateMemoryAnswer:
    @pytest.mark.asyncio
    async def test_information_extraction_perfect(self, sample_question, mock_llm_client):
        clear_eval_cache()
        score = await evaluate_memory_answer(
            answer="The user is a software engineer at a fintech startup.",
            ground_truth=sample_question.ground_truth,
            constraints=sample_question.constraints,
            question=sample_question,
            llm=mock_llm_client,
            supporting_session_ids=["session_03"],
        )
        assert isinstance(score, MemoryScore)
        assert score.structural_pass is True
        assert score.sources_pass is True
        assert score.judge_score == 90.0
        assert score.accuracy >= 0.5

    @pytest.mark.asyncio
    async def test_abstention_correct(self, abstention_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(return_value="YES")
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 100, "completeness": 100, "rationale": "good"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        score = await evaluate_memory_answer(
            answer="I do not have that information.",
            ground_truth=abstention_question.ground_truth,
            constraints=abstention_question.constraints,
            question=abstention_question,
            llm=llm,
        )
        assert score.abstained is True
        assert score.abstention_correct is True
        assert score.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_abstention_hallucinated(self, abstention_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(return_value="NO")
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 20, "completeness": 0, "rationale": "made it up"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        score = await evaluate_memory_answer(
            answer="The SSN is 555-12-3456.",
            ground_truth=abstention_question.ground_truth,
            constraints=abstention_question.constraints,
            question=abstention_question,
            llm=llm,
        )
        assert score.abstained is False
        assert score.abstention_correct is False
        assert score.accuracy == 0.0

    @pytest.mark.asyncio
    async def test_update_question_reflects_latest(self, update_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(return_value="NO")
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 95, "completeness": 90, "rationale": "matches latest"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        score = await evaluate_memory_answer(
            answer="The user lives in Manila.",
            ground_truth=update_question.ground_truth,
            constraints=update_question.constraints,
            question=update_question,
            llm=llm,
        )
        assert score.update_precision_correct is True

    @pytest.mark.asyncio
    async def test_update_question_uses_stale_value(self, update_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(return_value="NO")
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 30, "completeness": 20, "rationale": "out of date"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        score = await evaluate_memory_answer(
            answer="The user lives in Tokyo.",
            ground_truth=update_question.ground_truth,
            constraints=update_question.constraints,
            question=update_question,
            llm=llm,
        )
        assert score.update_precision_correct is False
        assert score.accuracy < 0.5

    @pytest.mark.asyncio
    async def test_temporal_question_year_match(self, temporal_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(side_effect=["2026-03-15", "NO"])
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 80, "completeness": 80, "rationale": "ok"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        score = await evaluate_memory_answer(
            answer="The user mentioned mountain biking in 2026 [session_04].",
            ground_truth=temporal_question.ground_truth,
            constraints=temporal_question.constraints,
            question=temporal_question,
            llm=llm,
            supporting_session_ids=["session_04"],
        )
        assert score.temporal_correct is True

    @pytest.mark.asyncio
    async def test_temporal_question_no_marker(self, temporal_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(side_effect=["NONE", "NO"])
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 50, "completeness": 50, "rationale": "vague"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        score = await evaluate_memory_answer(
            answer="The user mentioned mountain biking once.",
            ground_truth=temporal_question.ground_truth,
            constraints=temporal_question.constraints,
            question=temporal_question,
            llm=llm,
        )
        assert score.temporal_correct is False

    @pytest.mark.asyncio
    async def test_judge_cache_hit(self, sample_question, mock_llm_client):
        clear_eval_cache()
        # First call populates cache
        await evaluate_memory_answer(
            answer="Software engineer at fintech",
            ground_truth=sample_question.ground_truth,
            constraints=sample_question.constraints,
            question=sample_question,
            llm=mock_llm_client,
            supporting_session_ids=["session_03"],
        )
        first_calls = mock_llm_client.judge.call_count
        # Second identical call hits cache
        await evaluate_memory_answer(
            answer="Software engineer at fintech",
            ground_truth=sample_question.ground_truth,
            constraints=sample_question.constraints,
            question=sample_question,
            llm=mock_llm_client,
            supporting_session_ids=["session_03"],
        )
        assert mock_llm_client.judge.call_count == first_calls

    @pytest.mark.asyncio
    async def test_structural_failure_dampens_accuracy(self, sample_question):
        clear_eval_cache()
        llm = AsyncMock()
        llm.classify = AsyncMock(return_value="NO")
        llm.judge = AsyncMock(
            return_value=LLMResponse(
                text='{"accuracy": 80, "completeness": 80, "rationale": "ok"}',
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        # Answer omits the required "fintech" mention
        score = await evaluate_memory_answer(
            answer="The user is a software engineer.",
            ground_truth=sample_question.ground_truth,
            constraints=sample_question.constraints,
            question=sample_question,
            llm=llm,
            supporting_session_ids=["session_03"],
        )
        assert score.structural_pass is False
        assert score.accuracy < 0.8


class TestMemoryScoreModel:
    def test_default_values(self):
        s = MemoryScore()
        assert s.accuracy == 0.0
        assert s.structural_pass is True
        assert s.judge_score == 0.0

    def test_extra_field_forbidden(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MemoryScore(unknown=5)
