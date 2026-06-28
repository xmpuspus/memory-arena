"""Validate the bundled smoke YAML questions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from memory_arena.benchmark.questions import _yaml_to_record
from memory_arena.sessions.schema import CATEGORIES

SMOKE_DIR = (
    Path(__file__).parent.parent / "datasets" / "longmemeval-s" / "questions" / "smoke_synthetic"
)


def _load_smoke_files() -> list[Path]:
    return sorted(SMOKE_DIR.glob("*.yaml"))


class TestSmokeYamlStructure:
    def test_smoke_dir_exists(self):
        assert SMOKE_DIR.exists()

    def test_five_category_files(self):
        files = _load_smoke_files()
        assert len(files) == 5

    def test_files_named_per_category(self):
        names = [p.stem for p in _load_smoke_files()]
        # cat1_extraction, cat2_multi_session, cat3_temporal, cat4_update, cat5_abstention
        assert any("extraction" in n for n in names)
        assert any("multi_session" in n for n in names)
        assert any("temporal" in n for n in names)
        assert any("update" in n for n in names)
        assert any("abstention" in n for n in names)


class TestSmokeQuestionShape:
    @pytest.mark.parametrize("path", _load_smoke_files(), ids=lambda p: p.name)
    def test_yaml_loads(self, path):
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 6

    @pytest.mark.parametrize("path", _load_smoke_files(), ids=lambda p: p.name)
    def test_each_question_validates(self, path):
        data = yaml.safe_load(path.read_text())
        for item in data:
            record = _yaml_to_record(item)
            assert record.id
            assert record.question
            assert record.category in CATEGORIES

    def test_total_30_questions(self):
        total = 0
        for path in _load_smoke_files():
            data = yaml.safe_load(path.read_text())
            total += len(data)
        assert total >= 30

    def test_abstention_questions_marked_as_expected(self):
        path = SMOKE_DIR / "cat5_abstention.yaml"
        data = yaml.safe_load(path.read_text())
        for item in data:
            assert item["constraints"]["abstention_expected"] is True

    def test_update_questions_have_fact_versions(self):
        path = SMOKE_DIR / "cat4_update.yaml"
        data = yaml.safe_load(path.read_text())
        for item in data:
            versions = item["ground_truth"].get("fact_versions") or []
            assert len(versions) >= 2

    def test_temporal_questions_have_valid_as_of(self):
        path = SMOKE_DIR / "cat3_temporal.yaml"
        data = yaml.safe_load(path.read_text())
        for item in data:
            assert item["ground_truth"].get("valid_as_of") is not None

    def test_unique_ids_across_files(self):
        seen: set[str] = set()
        for path in _load_smoke_files():
            data = yaml.safe_load(path.read_text())
            for item in data:
                assert item["id"] not in seen, f"Duplicate id: {item['id']}"
                seen.add(item["id"])
        assert len(seen) >= 30
