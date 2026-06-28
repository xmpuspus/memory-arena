"""Tests for memory_arena.benchmark.questions YAML/JSONL loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from memory_arena.benchmark.questions import (
    discover_corpora,
    load_memory_questions,
)


def _write_yaml(path: Path, items):
    path.write_text(yaml.safe_dump(items))


def _build_dataset(tmp_path: Path, corpus: str = "longmemeval-s") -> Path:
    base = tmp_path / "datasets" / corpus / "questions" / "smoke"
    base.mkdir(parents=True)
    items = [
        {
            "id": "q1",
            "category": "information_extraction",
            "hops": 1,
            "question": "What does the user do?",
            "ground_truth": {
                "answer": "Software engineer",
                "supporting_session_ids": ["s1"],
            },
            "constraints": {
                "must_mention": ["engineer"],
                "max_tokens": 200,
            },
        },
        {
            "id": "q2",
            "category": "abstention",
            "question": "Unknown?",
            "ground_truth": {"answer": "I do not know"},
            "constraints": {"abstention_expected": True},
        },
    ]
    _write_yaml(base / "smoke.yaml", items)
    return tmp_path


class TestLoadMemoryQuestions:
    def test_smoke_yaml_loads(self, tmp_path, monkeypatch):
        _build_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        records = load_memory_questions("longmemeval-s", subset="smoke")
        assert len(records) == 2
        assert records[0].id == "q1"
        assert records[0].constraints.must_mention == ["engineer"]

    def test_abstention_constraint_propagates(self, tmp_path, monkeypatch):
        _build_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        records = load_memory_questions("longmemeval-s", subset="smoke")
        abst = next(r for r in records if r.id == "q2")
        assert abst.constraints.abstention_expected is True
        assert abst.category == "abstention"

    def test_missing_corpus_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_memory_questions("does-not-exist", subset="smoke")
        assert result == []

    def test_custom_yaml_file_path(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom.yaml"
        _write_yaml(
            custom,
            [
                {
                    "id": "x1",
                    "category": "temporal",
                    "question": "When?",
                    "ground_truth": {"answer": "2026"},
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        records = load_memory_questions("any", subset=str(custom))
        assert len(records) == 1
        assert records[0].id == "x1"

    def test_custom_yaml_dir(self, tmp_path, monkeypatch):
        d = tmp_path / "qdir"
        d.mkdir()
        for i, cat in enumerate(["information_extraction", "temporal"]):
            _write_yaml(
                d / f"{cat}.yaml",
                [
                    {
                        "id": f"q{i}",
                        "category": cat,
                        "question": "?",
                        "ground_truth": {"answer": "x"},
                    }
                ],
            )
        monkeypatch.chdir(tmp_path)
        records = load_memory_questions("any", subset=str(d))
        assert len(records) == 2

    def test_full_falls_back_to_yaml_dir(self, tmp_path, monkeypatch):
        base = tmp_path / "datasets" / "c" / "questions"
        base.mkdir(parents=True)
        _write_yaml(
            base / "main.yaml",
            [
                {
                    "id": "f1",
                    "category": "information_extraction",
                    "question": "?",
                    "ground_truth": {"answer": "x"},
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        records = load_memory_questions("c", subset="full")
        assert len(records) == 1


class TestDiscoverCorpora:
    def test_no_datasets_dir(self, tmp_path, monkeypatch):
        # MEM_ARENA_DATASETS_PATH override to a non-existent path so the new
        # bundled-fallback resolver in memory_arena.paths doesn't pick up the
        # in-package memory_arena/data/ that ships in the wheel.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MEM_ARENA_DATASETS_PATH", str(tmp_path / "nope"))
        assert discover_corpora() == []

    def test_finds_corpus_with_yaml(self, tmp_path, monkeypatch):
        _build_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MEM_ARENA_DATASETS_PATH", str(tmp_path / "datasets"))
        out = discover_corpora()
        assert "longmemeval-s" in out

    def test_finds_corpus_with_jsonl(self, tmp_path, monkeypatch):
        base = tmp_path / "datasets" / "j" / "processed"
        base.mkdir(parents=True)
        (base / "questions.jsonl").write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MEM_ARENA_DATASETS_PATH", str(tmp_path / "datasets"))
        assert "j" in discover_corpora()
