"""Tests for the cost-cap predicate and runner halt behavior.

The earlier implementation only checked the cap BEFORE each
ingest/recall call. If we were sitting just under the cap, ONE expensive
call would sneak past and the loop wouldn't halt until the NEXT iteration.
With the post-call check the breaching call still counts (we paid for it)
but no further calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from memory_arena.benchmark.runner import _check_cost_cap, _run_strategy
from memory_arena.llm.client import LLMResponse
from memory_arena.sessions.schema import (
    Constraints,
    GroundTruth,
    QuestionRecord,
    Session,
    Turn,
)
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult


class _FakeStrategy(MemoryStrategy):
    """Strategy whose ingest and recall consume a deterministic cost.

    `ingest_cost` and `recall_cost` are spent per call. Both methods append
    a record so the runner aggregates remain valid.
    """

    name = "fake"
    recall_at_k_measurable = True

    def __init__(self, ingest_cost: float = 0.0, recall_cost: float = 0.0):
        super().__init__()
        self._ingest_cost = ingest_cost
        self._recall_cost = recall_cost
        self.ingest_calls = 0
        self.recall_calls = 0

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id

    async def ingest_session(self, session: Session) -> IngestRecord:
        self.ingest_calls += 1
        return IngestRecord(
            session_id=session.id,
            latency_ms=1.0,
            cost_usd=self._ingest_cost,
            facts_extracted=len(session.turns),
        )

    async def recall(self, query: str, top_k: int = 5) -> RecallResult:
        self.recall_calls += 1
        return RecallResult(
            answer="ok [session_03]",
            supporting_session_ids=["session_03"],
            supporting_turn_ids=["session_03_turn_001"],
            strategy=self.name,
            latency_ms=1.0,
            cost_usd=self._recall_cost,
        )

    async def teardown(self) -> None:
        return None


def _mk_session(idx: int) -> Session:
    return Session(
        id=f"session_{idx:02d}",
        user_id="user_42",
        timestamp="2026-03-12T10:00:00Z",
        turns=[
            Turn(
                id=f"session_{idx:02d}_turn_001",
                session_id=f"session_{idx:02d}",
                role="user",
                content="dummy",
                timestamp="2026-03-12T10:00:00Z",
            ),
        ],
    )


def _mk_question(idx: int) -> QuestionRecord:
    return QuestionRecord(
        id=f"q{idx}",
        category="information_extraction",
        hops=1,
        question="What did the user say?",
        ground_truth=GroundTruth(
            answer="dummy",
            supporting_session_ids=["session_03"],
            supporting_turn_ids=["session_03_turn_001"],
        ),
        constraints=Constraints(must_mention=[]),
    )


class TestCheckCostCapPredicate:
    def test_no_cap_returns_false(self):
        errors: list[str] = []
        result = _check_cost_cap(
            "fake", {"fake": 100.0}, cost_cap=0.0, phase="ingest", errors_list=errors
        )
        assert result is False
        assert errors == []

    def test_under_cap_returns_false(self):
        errors: list[str] = []
        assert (
            _check_cost_cap(
                "fake", {"fake": 4.99}, cost_cap=5.0, phase="ingest", errors_list=errors
            )
            is False
        )
        assert errors == []

    def test_at_cap_returns_true_with_overshoot_zero(self):
        errors: list[str] = []
        assert (
            _check_cost_cap("fake", {"fake": 5.0}, cost_cap=5.0, phase="ingest", errors_list=errors)
            is True
        )
        assert len(errors) == 1
        assert "cost cap reached" in errors[0]
        assert "overshoot=$0.00" in errors[0]
        assert "during ingest" in errors[0]

    def test_overshoot_reported(self):
        errors: list[str] = []
        _check_cost_cap(
            "fake",
            {"fake": 5.23},
            cost_cap=5.0,
            phase="ingest",
            errors_list=errors,
            context="session=xxx",
        )
        assert "overshoot=$0.23" in errors[0]
        assert "of session=xxx" in errors[0]

    def test_dedup_per_phase(self):
        # Second call in the same phase shouldn't append another line.
        errors: list[str] = []
        _check_cost_cap("fake", {"fake": 5.5}, 5.0, phase="ingest", errors_list=errors)
        _check_cost_cap("fake", {"fake": 5.5}, 5.0, phase="ingest", errors_list=errors)
        assert len(errors) == 1


@pytest.mark.asyncio
async def test_cost_cap_halts_mid_ingest():
    """An ingest that pushes cumulative cost over the cap should record
    the cap-reached error and stop calling further ingest_session."""
    sessions = [_mk_session(i) for i in range(5)]
    questions = [_mk_question(0)]
    strategy = _FakeStrategy(ingest_cost=2.0, recall_cost=0.0)
    cumulative: dict[str, float] = {strategy.name: 0.0}

    fake_llm = AsyncMock()
    fake_llm.judge = AsyncMock(
        return_value=LLMResponse(
            text='{"accuracy": 50, "completeness": 50, "rationale": "ok"}',
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )
    )
    fake_llm.classify = AsyncMock(return_value="NO")

    class _DummyProgress:
        def advance(self, _task_id):
            return None

    with patch("memory_arena.benchmark.runner.LLMClient", return_value=fake_llm):
        out = await _run_strategy(
            strategy,
            sessions,
            questions,
            run_id="run-1",
            top_k=5,
            llm=fake_llm,
            cost_cap=5.0,
            cumulative_cost=cumulative,
            progress=_DummyProgress(),
            task_id=None,
        )

    # 0 -> 2 -> 4 -> 6: third ingest breaches; no fourth ingest.
    assert strategy.ingest_calls == 3
    cap_errors = [e for e in out["errors"] if "cost cap reached" in e and "during ingest" in e]
    assert len(cap_errors) == 1
    assert "overshoot=$1.00" in cap_errors[0]
    assert "session=session_02" in cap_errors[0]


@pytest.mark.asyncio
async def test_cost_cap_halts_during_recall():
    """A recall that pushes cumulative cost over the cap should record
    the cap-reached error, score the breaching call, and stop recalling."""
    sessions = [_mk_session(0)]
    questions = [_mk_question(i) for i in range(5)]
    strategy = _FakeStrategy(ingest_cost=0.0, recall_cost=2.0)
    cumulative: dict[str, float] = {strategy.name: 0.0}

    fake_llm = AsyncMock()
    fake_llm.judge = AsyncMock(
        return_value=LLMResponse(
            text='{"accuracy": 50, "completeness": 50, "rationale": "ok"}',
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )
    )
    fake_llm.classify = AsyncMock(return_value="NO")

    class _DummyProgress:
        def advance(self, _task_id):
            return None

    with patch("memory_arena.benchmark.runner.LLMClient", return_value=fake_llm):
        out = await _run_strategy(
            strategy,
            sessions,
            questions,
            run_id="run-2",
            top_k=5,
            llm=fake_llm,
            cost_cap=5.0,
            cumulative_cost=cumulative,
            progress=_DummyProgress(),
            task_id=None,
        )

    # 0 -> 2 -> 4 -> 6: third recall breaches and is scored, no fourth.
    assert strategy.recall_calls == 3
    assert len(out["recall_records"]) == 3
    cap_errors = [e for e in out["errors"] if "cost cap reached" in e and "during recall" in e]
    assert len(cap_errors) == 1
    assert "overshoot=$1.00" in cap_errors[0]
    assert "question=q2" in cap_errors[0]


@pytest.mark.asyncio
async def test_cost_cap_zero_disables_check():
    """cost_cap <= 0 must NOT halt regardless of cumulative spend."""
    sessions = [_mk_session(i) for i in range(3)]
    questions = [_mk_question(0)]
    strategy = _FakeStrategy(ingest_cost=10.0, recall_cost=0.0)
    cumulative: dict[str, float] = {strategy.name: 0.0}

    fake_llm = AsyncMock()
    fake_llm.judge = AsyncMock(
        return_value=LLMResponse(
            text='{"accuracy": 50, "completeness": 50, "rationale": "ok"}',
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )
    )
    fake_llm.classify = AsyncMock(return_value="NO")

    class _DummyProgress:
        def advance(self, _task_id):
            return None

    with patch("memory_arena.benchmark.runner.LLMClient", return_value=fake_llm):
        out = await _run_strategy(
            strategy,
            sessions,
            questions,
            run_id="run-3",
            top_k=5,
            llm=fake_llm,
            cost_cap=0.0,
            cumulative_cost=cumulative,
            progress=_DummyProgress(),
            task_id=None,
        )

    assert strategy.ingest_calls == 3
    cap_errors = [e for e in out["errors"] if "cost cap reached" in e]
    assert cap_errors == []
