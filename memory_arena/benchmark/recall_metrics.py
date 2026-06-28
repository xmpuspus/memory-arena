"""Turn-level and session-level recall metrics for memory strategies.

Lifted from kb-arena's ir_metrics.py and retargeted at supporting_session_ids
and supporting_turn_ids instead of chunk ids.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _hit_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    head = retrieved[:k]
    return 1.0 if any(r in expected for r in head) else 0.0


def _recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    head = retrieved[:k]
    found = sum(1 for r in head if r in expected)
    return found / len(expected)


def _precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    head = retrieved[:k]
    if not head:
        return 0.0
    found = sum(1 for r in head if r in expected)
    return found / len(head)


def _mrr(retrieved: list[str], expected: set[str]) -> float:
    if not expected:
        return 0.0
    for i, r in enumerate(retrieved):
        if r in expected:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    head = retrieved[:k]
    dcg = 0.0
    for i, r in enumerate(head):
        if r in expected:
            dcg += 1.0 / math.log2(i + 2)
    ideal_n = min(len(expected), k)
    if ideal_n == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def compute_memory_recall_metrics(
    retrieved_session_ids: Iterable[str],
    expected_session_ids: set[str],
    retrieved_turn_ids: Iterable[str] | None = None,
    expected_turn_ids: set[str] | None = None,
    k: int = 10,
) -> dict:
    """Return all 5 IR metrics at session-level and turn-level."""
    sess_list = list(retrieved_session_ids)
    turn_list = list(retrieved_turn_ids) if retrieved_turn_ids is not None else []
    expected_turn_ids = expected_turn_ids or set()

    return {
        "k": k,
        "session_hit_at_k": _hit_at_k(sess_list, expected_session_ids, k),
        "session_recall_at_k": _recall_at_k(sess_list, expected_session_ids, k),
        "session_precision_at_k": _precision_at_k(sess_list, expected_session_ids, k),
        "session_mrr": _mrr(sess_list, expected_session_ids),
        "session_ndcg_at_k": _ndcg_at_k(sess_list, expected_session_ids, k),
        "turn_hit_at_k": _hit_at_k(turn_list, expected_turn_ids, k),
        "turn_recall_at_k": _recall_at_k(turn_list, expected_turn_ids, k),
        "turn_precision_at_k": _precision_at_k(turn_list, expected_turn_ids, k),
        "turn_mrr": _mrr(turn_list, expected_turn_ids),
        "turn_ndcg_at_k": _ndcg_at_k(turn_list, expected_turn_ids, k),
    }


def aggregate_recall_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "mean_session_recall_at_k": 0.0,
            "mean_session_hit_at_k": 0.0,
            "mean_session_mrr": 0.0,
            "mean_turn_recall_at_k": 0.0,
            "mean_turn_hit_at_k": 0.0,
        }

    def mean(key: str) -> float:
        return sum(r.get(key, 0.0) for r in rows) / len(rows)

    return {
        "n": len(rows),
        "mean_session_recall_at_k": mean("session_recall_at_k"),
        "mean_session_hit_at_k": mean("session_hit_at_k"),
        "mean_session_precision_at_k": mean("session_precision_at_k"),
        "mean_session_mrr": mean("session_mrr"),
        "mean_session_ndcg_at_k": mean("session_ndcg_at_k"),
        "mean_turn_recall_at_k": mean("turn_recall_at_k"),
        "mean_turn_hit_at_k": mean("turn_hit_at_k"),
        "mean_turn_precision_at_k": mean("turn_precision_at_k"),
    }


__all__ = [
    "aggregate_recall_metrics",
    "compute_memory_recall_metrics",
]
