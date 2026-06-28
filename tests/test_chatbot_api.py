"""Tests for memory_arena.chatbot.api FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memory_arena.chatbot.api import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["strategies"], list)
        assert "full_context" in body["strategies"]

    def test_health_includes_results_flag(self, client):
        r = client.get("/api/health")
        assert "has_results" in r.json()


class TestCorpora:
    def test_corpora_returns_list(self, client):
        r = client.get("/api/corpora")
        assert r.status_code == 200
        body = r.json()
        assert "corpora" in body
        assert isinstance(body["corpora"], list)

    def test_corpora_has_label_and_count(self, client):
        r = client.get("/api/corpora")
        body = r.json()
        if body["corpora"]:
            entry = body["corpora"][0]
            assert "name" in entry
            assert "label" in entry
            assert "count" in entry


class TestStrategies:
    def test_strategies_endpoint(self, client):
        r = client.get("/api/strategies")
        assert r.status_code == 200
        body = r.json()
        names = [s["name"] for s in body["strategies"]]
        assert "full_context" in names
        assert "naive_vector" in names


class TestResultsLookup:
    def test_results_404_when_missing(self, client):
        r = client.get("/api/results/nonexistent-corpus")
        assert r.status_code == 404

    def test_benchmark_alias_404(self, client):
        r = client.get("/api/benchmark/nonexistent-corpus")
        assert r.status_code == 404


class TestCORS:
    def test_cors_preflight_allowed(self, client):
        r = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORSMiddleware responds 200 to preflight requests
        assert r.status_code in (200, 204)


class TestRecallRecords:
    def test_recall_records_falls_back_to_seed0(self, client):
        # bm25 ships only `_seed{N}.json` / `_summary.json` with no bare
        # `{corpus}_{strategy}.json`, the same shape as the bundled wheel
        # snapshot. The endpoint must fall back to seed 0 (which carries
        # recall_records) instead of 404ing, otherwise the Recall Lab page is
        # dead for every pip-installed user.
        r = client.get("/api/recall-records/longmemeval-s/bm25")
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "bm25"
        assert len(body["records"]) > 0

    def test_recall_records_unknown_strategy_404(self, client):
        r = client.get("/api/recall-records/longmemeval-s/not_a_strategy")
        assert r.status_code == 404
