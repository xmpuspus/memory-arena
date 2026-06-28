"""Tests for memory_arena.sessions.loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arena.sessions.loaders import (
    LongMemEvalLoader,
    load_questions_jsonl,
    load_sessions,
)
from memory_arena.sessions.schema import QuestionRecord, Session


def _write_raw(tmp_path: Path, payload: list[dict]) -> Path:
    p = tmp_path / "longmemeval_s.json"
    p.write_text(json.dumps(payload))
    return p


class TestLongMemEvalLoader:
    def test_missing_raw(self, tmp_path):
        loader = LongMemEvalLoader(tmp_path / "nope.json")
        with pytest.raises(FileNotFoundError):
            loader.load_raw()

    def test_loads_minimal_record(self, tmp_path, longmemeval_raw_record):
        path = _write_raw(tmp_path, [longmemeval_raw_record])
        loader = LongMemEvalLoader(path)
        sessions, questions = loader.normalize()
        assert len(sessions) == 2
        assert len(questions) == 1
        assert isinstance(sessions[0], Session)
        assert isinstance(questions[0], QuestionRecord)

    def test_question_category_mapped(self, tmp_path, longmemeval_raw_record):
        path = _write_raw(tmp_path, [longmemeval_raw_record])
        loader = LongMemEvalLoader(path)
        _, questions = loader.normalize()
        assert questions[0].category == "information_extraction"

    def test_temporal_category(self, tmp_path):
        record = {
            "question_id": "qa1",
            "question_type": "temporal-reasoning",
            "question": "When?",
            "answer": "Last week",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2026-03-12"],
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            "answer_session_ids": ["s1"],
            "user_id": "u1",
        }
        path = _write_raw(tmp_path, [record])
        loader = LongMemEvalLoader(path)
        _, questions = loader.normalize()
        assert questions[0].category == "temporal"

    def test_abstention_category_marks_constraint(self, tmp_path):
        record = {
            "question_id": "qa1",
            "question_type": "abstention",
            "question": "What is the SSN?",
            "answer": "I don't have that information.",
            "haystack_session_ids": ["s1"],
            "haystack_dates": [None],
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            "answer_session_ids": [],
            "user_id": "u1",
        }
        path = _write_raw(tmp_path, [record])
        loader = LongMemEvalLoader(path)
        _, questions = loader.normalize()
        assert questions[0].category == "abstention"
        assert questions[0].constraints.abstention_expected is True

    def test_session_dedup_across_records(self, tmp_path, longmemeval_raw_record):
        # Two records share the same session_03
        rec2 = {**longmemeval_raw_record, "question_id": "qa2"}
        path = _write_raw(tmp_path, [longmemeval_raw_record, rec2])
        loader = LongMemEvalLoader(path)
        sessions, questions = loader.normalize()
        # session_03 + session_05, even though referenced twice
        assert len(sessions) == 2
        assert len(questions) == 2

    def test_turns_get_sequential_ids(self, tmp_path, longmemeval_raw_record):
        path = _write_raw(tmp_path, [longmemeval_raw_record])
        loader = LongMemEvalLoader(path)
        sessions, _ = loader.normalize()
        s = next(x for x in sessions if x.id == "session_03")
        ids = [t.id for t in s.turns]
        assert ids == ["session_03_turn_000", "session_03_turn_001"]

    def test_write_processed_creates_jsonl(self, tmp_path, longmemeval_raw_record):
        path = _write_raw(tmp_path, [longmemeval_raw_record])
        loader = LongMemEvalLoader(path)
        sessions, questions = loader.normalize()
        out_dir = tmp_path / "processed"
        s_path, q_path = loader.write_processed(sessions, questions, out_dir)
        assert s_path.exists()
        assert q_path.exists()
        s_lines = s_path.read_text().splitlines()
        q_lines = q_path.read_text().splitlines()
        assert len(s_lines) == len(sessions)
        assert len(q_lines) == len(questions)

    def test_write_processed_roundtrip(self, tmp_path, longmemeval_raw_record):
        path = _write_raw(tmp_path, [longmemeval_raw_record])
        loader = LongMemEvalLoader(path)
        sessions, questions = loader.normalize()
        out_dir = tmp_path / "processed"
        loader.write_processed(sessions, questions, out_dir)
        # Roundtrip via Session.model_validate_json
        for line in (out_dir / "sessions.jsonl").read_text().splitlines():
            s = Session.model_validate_json(line)
            assert s.id

    def test_missing_optional_fields_default_safely(self, tmp_path):
        minimal = {
            "question": "?",
            "answer": "x",
            "haystack_sessions": [],
        }
        path = _write_raw(tmp_path, [minimal])
        loader = LongMemEvalLoader(path)
        sessions, questions = loader.normalize()
        assert len(questions) == 1
        assert questions[0].id == "longmemeval-0000"


class TestLoadHelpers:
    def test_load_sessions_missing_file(self):
        # Different cwd: no datasets/<corpus>/processed/sessions.jsonl
        result = load_sessions("does-not-exist")
        assert result == []

    def test_load_questions_jsonl_missing(self):
        result = load_questions_jsonl("does-not-exist")
        assert result == []

    def test_load_sessions_handles_blank_lines(self, tmp_path, monkeypatch, sample_session):
        # Build a fake datasets dir
        dataset = tmp_path / "datasets" / "fake" / "processed"
        dataset.mkdir(parents=True)
        path = dataset / "sessions.jsonl"
        path.write_text(sample_session.model_dump_json() + "\n\n")
        monkeypatch.chdir(tmp_path)
        result = load_sessions("fake")
        assert len(result) == 1
        assert result[0].id == "session_03"

    def test_load_questions_jsonl_handles_blanks(self, tmp_path, monkeypatch, sample_question):
        dataset = tmp_path / "datasets" / "fake" / "processed"
        dataset.mkdir(parents=True)
        path = dataset / "questions.jsonl"
        path.write_text("\n" + sample_question.model_dump_json() + "\n")
        monkeypatch.chdir(tmp_path)
        result = load_questions_jsonl("fake")
        assert len(result) == 1
