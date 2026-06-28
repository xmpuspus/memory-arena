"""Per-strategy wall-clock budget in the benchmark runner.

A strategy stuck in a non-LLM await (vendor socket, Neo4j/FalkorDB auth hang,
Chroma lock) must not block the whole ``asyncio.gather`` batch forever. The
runner wraps each strategy in ``_run_strategy_with_timeout`` so a hung strategy
is cancelled past its budget and recorded as an error while the rest of the
batch finishes and writes.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_arena.benchmark.runner import _run_strategy_with_timeout


@pytest.mark.asyncio
async def test_timeout_returns_error_result_instead_of_hanging():
    async def _hang():
        await asyncio.sleep(30)
        return {"strategy": "slow", "accuracy": 1.0}

    out = await _run_strategy_with_timeout(_hang(), "slow", timeout_s=0.05)

    assert out["timed_out"] is True
    assert out["strategy"] == "slow"
    assert "timed out" in out["errors"][0]


@pytest.mark.asyncio
async def test_fast_strategy_passes_through_untouched():
    async def _fast():
        return {"strategy": "quick", "accuracy": 0.5}

    out = await _run_strategy_with_timeout(_fast(), "quick", timeout_s=5)

    assert out == {"strategy": "quick", "accuracy": 0.5}
    assert "timed_out" not in out
