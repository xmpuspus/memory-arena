"""Tests for memory_arena.benchmark.recall_metrics."""

from __future__ import annotations

import math

from memory_arena.benchmark.recall_metrics import (
    aggregate_recall_metrics,
    compute_memory_recall_metrics,
)


class TestSessionLevel:
    def test_perfect_hit(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1", "s2"],
            expected_session_ids={"s1"},
            k=5,
        )
        assert m["session_hit_at_k"] == 1.0
        assert m["session_recall_at_k"] == 1.0
        assert m["session_mrr"] == 1.0

    def test_miss(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s2", "s3"],
            expected_session_ids={"s1"},
            k=5,
        )
        assert m["session_hit_at_k"] == 0.0
        assert m["session_recall_at_k"] == 0.0
        assert m["session_mrr"] == 0.0

    def test_partial_recall(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1"],
            expected_session_ids={"s1", "s2"},
            k=5,
        )
        assert m["session_recall_at_k"] == 0.5

    def test_precision(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1", "s2", "s3"],
            expected_session_ids={"s1"},
            k=3,
        )
        assert abs(m["session_precision_at_k"] - (1 / 3)) < 1e-9

    def test_mrr_second_hit(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s9", "s1"],
            expected_session_ids={"s1"},
            k=5,
        )
        assert m["session_mrr"] == 0.5

    def test_ndcg_top_first(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1"],
            expected_session_ids={"s1"},
            k=5,
        )
        # 1 / log2(2) = 1.0 ideal; dcg = 1 / log2(2) = 1.0
        assert abs(m["session_ndcg_at_k"] - 1.0) < 1e-9

    def test_top_k_truncates(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s9", "s8", "s7", "s1"],
            expected_session_ids={"s1"},
            k=2,
        )
        assert m["session_hit_at_k"] == 0.0  # s1 is at position 4

    def test_no_expected_returns_zero(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1"],
            expected_session_ids=set(),
            k=5,
        )
        assert m["session_recall_at_k"] == 0.0
        assert m["session_hit_at_k"] == 0.0


class TestTurnLevel:
    def test_basic(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1"],
            expected_session_ids={"s1"},
            retrieved_turn_ids=["s1_t1"],
            expected_turn_ids={"s1_t1"},
            k=3,
        )
        assert m["turn_hit_at_k"] == 1.0
        assert m["turn_recall_at_k"] == 1.0
        assert m["turn_mrr"] == 1.0

    def test_turn_miss_with_session_hit(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1"],
            expected_session_ids={"s1"},
            retrieved_turn_ids=["s1_t9"],
            expected_turn_ids={"s1_t1"},
            k=3,
        )
        # Session retrieved but wrong turn
        assert m["session_hit_at_k"] == 1.0
        assert m["turn_hit_at_k"] == 0.0

    def test_turn_ids_optional(self):
        m = compute_memory_recall_metrics(
            retrieved_session_ids=["s1"],
            expected_session_ids={"s1"},
            k=3,
        )
        assert m["turn_hit_at_k"] == 0.0
        assert m["turn_recall_at_k"] == 0.0


class TestAggregate:
    def test_empty(self):
        agg = aggregate_recall_metrics([])
        assert agg["n"] == 0
        assert agg["mean_session_recall_at_k"] == 0.0

    def test_means(self):
        rows = [
            {
                "session_recall_at_k": 1.0,
                "session_hit_at_k": 1.0,
                "session_precision_at_k": 0.5,
                "session_mrr": 1.0,
                "session_ndcg_at_k": 1.0,
                "turn_recall_at_k": 1.0,
                "turn_hit_at_k": 1.0,
                "turn_precision_at_k": 0.5,
            },
            {
                "session_recall_at_k": 0.0,
                "session_hit_at_k": 0.0,
                "session_precision_at_k": 0.0,
                "session_mrr": 0.0,
                "session_ndcg_at_k": 0.0,
                "turn_recall_at_k": 0.0,
                "turn_hit_at_k": 0.0,
                "turn_precision_at_k": 0.0,
            },
        ]
        agg = aggregate_recall_metrics(rows)
        assert agg["n"] == 2
        assert math.isclose(agg["mean_session_recall_at_k"], 0.5)
        assert math.isclose(agg["mean_turn_recall_at_k"], 0.5)

    def test_mean_handles_missing_keys(self):
        rows = [{"session_recall_at_k": 1.0}, {"session_recall_at_k": 0.0}]
        agg = aggregate_recall_metrics(rows)
        assert math.isclose(agg["mean_session_recall_at_k"], 0.5)
