"""Stub router module — kept for backward import compatibility.

The kb-arena multi-strategy intent classifier is not used in memory-arena v0.1
(memory strategies don't have an intent-routing layer the same way RAG strategies do).
"""

from __future__ import annotations


async def classify_intent(query: str, history: list[dict] | None = None) -> str:
    """Stub: every memory query is treated the same."""
    return "memory_recall"
