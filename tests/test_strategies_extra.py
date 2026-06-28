"""Additional strategy tests: graphiti timestamp parser, mem0g graph flag, base helpers."""

from __future__ import annotations

from datetime import datetime

from memory_arena.strategies.base import IngestRecord, RecallResult
from memory_arena.strategies.graphiti import GraphitiStrategy
from memory_arena.strategies.mem0 import Mem0Strategy
from memory_arena.strategies.mem0 import _build_config as _build_mem0_config
from memory_arena.strategies.mem0g import Mem0GraphStrategy
from memory_arena.strategies.mem0g import _build_config as _build_mem0g_config


class TestGraphitiTimestamp:
    def test_parse_iso(self):
        ts = GraphitiStrategy._parse_ts("2026-03-15T10:00:00+00:00")
        assert isinstance(ts, datetime)
        assert ts.year == 2026
        assert ts.month == 3
        assert ts.day == 15

    def test_parse_z_suffix(self):
        ts = GraphitiStrategy._parse_ts("2026-03-15T10:00:00Z")
        assert ts.year == 2026

    def test_parse_naive_gets_utc(self):
        ts = GraphitiStrategy._parse_ts("2026-03-15T10:00:00")
        assert ts.tzinfo is not None

    def test_parse_invalid_falls_back_to_now(self):
        ts = GraphitiStrategy._parse_ts("not-a-date")
        assert isinstance(ts, datetime)

    def test_parse_none(self):
        ts = GraphitiStrategy._parse_ts(None)
        assert isinstance(ts, datetime)


class TestMem0Config:
    def test_v2_default_has_no_graph(self):
        # v2 OSS removed graph_store; mem0.py builds a vector-only config.
        c = _build_mem0_config("run-x")
        assert c["vector_store"]["provider"] == "chroma"
        assert "graph_store" not in c

    def test_mem0g_v1_config_has_graph(self):
        # mem0g pins v1 specifically to keep the graph_store wired up.
        c = _build_mem0g_config("run-x")
        assert "graph_store" in c
        assert c["graph_store"]["provider"] == "neo4j"

    def test_collection_name_uses_run_id(self):
        c = _build_mem0_config("abc123")
        assert "abc123" in c["vector_store"]["config"]["collection_name"]


class TestMem0GraphSeparation:
    """v2.0.0 removed graph_store from OSS, so mem0g no longer subclasses
    Mem0Strategy — they pin different mem0ai majors and have different APIs."""

    def test_no_longer_subclass_of_mem0(self):
        # Intentional: mem0g (v1 API) and mem0 (v2 API) diverged at v2.0.0.
        assert not issubclass(Mem0GraphStrategy, Mem0Strategy)

    def test_graph_enabled_flag(self):
        s = Mem0GraphStrategy()
        assert s._graph_enabled is True

    def test_distinct_names(self):
        assert Mem0Strategy.name == "mem0"
        assert Mem0GraphStrategy.name == "mem0g"


class TestModels:
    def test_ingest_record_defaults(self):
        r = IngestRecord(session_id="s1")
        assert r.session_id == "s1"
        assert r.latency_ms == 0.0
        assert r.tokens_used == 0
        assert r.cost_usd == 0.0
        assert r.facts_extracted == 0
        assert r.error == ""

    def test_ingest_record_with_error(self):
        r = IngestRecord(session_id="s1", error="setup failed")
        assert r.error == "setup failed"

    def test_recall_result_defaults(self):
        r = RecallResult(answer="x")
        assert r.answer == "x"
        assert r.supporting_session_ids == []
        assert r.supporting_turn_ids == []
        assert r.retrieved_memories == []
        assert r.cost_usd == 0.0
        assert r.mock is False

    def test_recall_result_with_metadata(self):
        r = RecallResult(
            answer="x",
            supporting_session_ids=["s1"],
            retrieved_memories=[{"foo": "bar"}],
            tokens_used=42,
        )
        assert r.tokens_used == 42
        assert r.retrieved_memories == [{"foo": "bar"}]


class TestVendorRegistry:
    def test_strategy_names_are_canonical(self):
        from memory_arena.strategies import STRATEGY_NAMES

        for n in ("full_context", "recency_window", "naive_vector"):
            assert n in STRATEGY_NAMES

    def test_baseline_strategies_instantiate(self):
        from memory_arena.strategies import get_strategy

        for n in ("full_context", "recency_window", "naive_vector"):
            s = get_strategy(n)
            assert s.name == n
