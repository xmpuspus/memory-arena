"""Tests for memory_arena.cli."""

from __future__ import annotations

from typer.testing import CliRunner

from memory_arena.cli import app

runner = CliRunner()


class TestCLIHelp:
    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "memory-arena" in result.output

    def test_init_corpus_help(self):
        result = runner.invoke(app, ["init-corpus", "--help"])
        assert result.exit_code == 0
        assert "corpus" in result.output.lower()

    def test_benchmark_help(self):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0

    def test_recall_lab_help(self):
        result = runner.invoke(app, ["recall-lab", "--help"])
        assert result.exit_code == 0

    def test_health_help(self):
        result = runner.invoke(app, ["health", "--help"])
        assert result.exit_code == 0


class TestInitCorpus:
    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init-corpus", "test-corpus"])
        assert result.exit_code == 0
        assert (tmp_path / "datasets" / "test-corpus" / "raw").exists()
        assert (tmp_path / "datasets" / "test-corpus" / "processed").exists()
        assert (tmp_path / "datasets" / "test-corpus" / "questions").exists()
        assert (tmp_path / "datasets" / "test-corpus" / "questions" / "smoke").exists()

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init-corpus", "test-corpus"])
        result = runner.invoke(app, ["init-corpus", "test-corpus"])
        assert result.exit_code == 0
        assert "already exists" in result.output


class TestHealth:
    def test_health_runs(self):
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        assert "Memory Arena" in result.output

    def test_health_json_format(self):
        result = runner.invoke(app, ["health", "--format", "json"])
        assert result.exit_code == 0
        import json

        body = json.loads(result.output)
        assert "api_keys" in body
        assert "strategies_registered" in body


class TestBenchmarkDryRun:
    def test_dry_run_no_keys_required(self, monkeypatch):
        # Provide a fake key so the preflight passes
        from memory_arena.settings import settings

        old = settings.anthropic_api_key
        settings.anthropic_api_key = "fake-key-for-dry-run"
        try:
            result = runner.invoke(app, ["benchmark", "--dry-run"])
            assert result.exit_code == 0
            assert "Dry run" in result.output
        finally:
            settings.anthropic_api_key = old


class TestIngestSessionsErrors:
    def test_missing_raw_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["ingest-sessions", "--corpus", "longmemeval-s"])
        assert result.exit_code == 1
        assert "Raw file not found" in result.output


class TestReportErrors:
    def test_no_results(self, tmp_path, monkeypatch):
        # RESULTS_PATH override so the bundled-fallback resolver doesn't pick
        # up memory_arena/data/results_snapshot/ that ships in the wheel.
        # MEM_ARENA_RESULTS_PATH still works as a legacy alias.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MEM_ARENA_RESULTS_PATH", str(tmp_path / "nope"))
        result = runner.invoke(app, ["report", "--corpus", "longmemeval-s"])
        assert result.exit_code == 1


class TestCorpusSlugValidation:
    """Path-traversal protection — corpus name flows into Path("datasets") / X."""

    def test_path_traversal_corpus_rejected_ingest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["ingest-sessions", "--corpus", "../../etc/passwd"])
        assert result.exit_code != 0
        assert "invalid corpus name" in result.output.lower()

    def test_path_traversal_corpus_rejected_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init-corpus", "../escape"])
        assert result.exit_code != 0
        assert "invalid corpus name" in result.output.lower()

    def test_uppercase_corpus_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["report", "--corpus", "LongMemEval"])
        assert result.exit_code != 0
        assert "invalid corpus name" in result.output.lower()

    def test_slash_in_corpus_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["benchmark", "--corpus", "foo/bar", "--dry-run"])
        assert result.exit_code != 0
        assert "invalid corpus name" in result.output.lower()

    def test_valid_corpus_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # init-corpus with a valid slug should succeed (creates dirs).
        result = runner.invoke(app, ["init-corpus", "longmemeval-s"])
        assert result.exit_code == 0
