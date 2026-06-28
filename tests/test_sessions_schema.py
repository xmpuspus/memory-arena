"""Tests for memory_arena.sessions.schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory_arena.sessions.schema import (
    CATEGORIES,
    Constraints,
    FactAssertion,
    GroundTruth,
    QuestionRecord,
    Session,
    Turn,
    category_to_tier,
)


class TestTurn:
    def test_construct(self):
        t = Turn(id="t1", session_id="s1", role="user", content="hi")
        assert t.id == "t1"
        assert t.session_id == "s1"
        assert t.role == "user"
        assert t.content == "hi"
        assert t.timestamp is None
        assert t.metadata == {}

    def test_with_timestamp(self):
        t = Turn(id="t1", session_id="s1", role="assistant", content="hi", timestamp="2026-04-01")
        assert t.timestamp == "2026-04-01"

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Turn(id="t1", session_id="s1", role="user", content="x", extra_field="boom")

    def test_metadata_dict(self):
        t = Turn(id="t", session_id="s", role="user", content="x", metadata={"foo": 1})
        assert t.metadata == {"foo": 1}

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            Turn(role="user", content="x")


class TestSession:
    def test_construct_empty(self):
        s = Session(id="s1")
        assert s.id == "s1"
        assert s.user_id == "default"
        assert s.turns == []
        assert s.turn_count == 0

    def test_with_turns(self, sample_session):
        assert sample_session.turn_count == 3
        assert sample_session.turns[0].role == "user"

    def test_to_messages(self, sample_session):
        messages = sample_session.to_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "content" in messages[0]

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Session(id="s1", bogus="x")

    def test_user_id_default(self):
        s = Session(id="s1")
        assert s.user_id == "default"


class TestFactAssertion:
    def test_basic(self):
        f = FactAssertion(value="Manila", source_session_id="s1")
        assert f.value == "Manila"
        assert f.confidence == 1.0
        assert f.valid_at is None

    def test_with_temporal(self):
        f = FactAssertion(
            value="Tokyo",
            valid_at="2026-01-01",
            invalid_at="2026-03-01",
            source_session_id="s1",
        )
        assert f.valid_at == "2026-01-01"
        assert f.invalid_at == "2026-03-01"

    def test_confidence_float(self):
        f = FactAssertion(value="x", source_session_id="s1", confidence=0.7)
        assert abs(f.confidence - 0.7) < 1e-9


class TestGroundTruth:
    def test_basic(self):
        g = GroundTruth(answer="x")
        assert g.answer == "x"
        assert g.supporting_session_ids == []
        assert g.fact_versions == []

    def test_with_sources(self):
        g = GroundTruth(answer="x", supporting_session_ids=["s1", "s2"])
        assert g.supporting_session_ids == ["s1", "s2"]

    def test_with_fact_versions(self):
        g = GroundTruth(
            answer="latest",
            fact_versions=[
                FactAssertion(value="old", source_session_id="s1"),
                FactAssertion(value="latest", source_session_id="s2"),
            ],
        )
        assert len(g.fact_versions) == 2


class TestConstraints:
    def test_defaults(self):
        c = Constraints()
        assert c.must_mention == []
        assert c.must_not_claim == []
        assert c.abstention_expected is False
        assert c.max_tokens == 500

    def test_overrides(self):
        c = Constraints(must_mention=["A"], abstention_expected=True, max_tokens=100)
        assert c.must_mention == ["A"]
        assert c.abstention_expected is True
        assert c.max_tokens == 100


class TestQuestionRecord:
    def test_construct(self, sample_question):
        assert sample_question.id == "longmemeval-extract-001"
        assert sample_question.category == "information_extraction"
        assert sample_question.hops == 1
        assert sample_question.constraints.must_mention == ["software engineer", "fintech"]

    def test_categories_match_canonical(self):
        for cat in CATEGORIES:
            q = QuestionRecord(
                id=f"q-{cat}",
                category=cat,
                question="?",
                ground_truth=GroundTruth(answer="x"),
            )
            assert q.category == cat


class TestCategoryToTier:
    def test_known_categories(self):
        assert category_to_tier("information_extraction") == 1
        assert category_to_tier("multi_session_reasoning") == 2
        assert category_to_tier("temporal") == 3
        assert category_to_tier("knowledge_update") == 4
        assert category_to_tier("abstention") == 5

    def test_unknown_returns_zero(self):
        assert category_to_tier("nonsense") == 0
