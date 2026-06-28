"""Generator × judge robustness sweep.

Question this answers: how much of the v0.1.5 leaderboard depends on
our specific (gen=Sonnet 4.6, judge=Opus 4.7) configuration? Re-runs
the top-K strategies under all four cells of the 2x2:

    gen=Sonnet 4.6  × judge=Opus 4.7   (default — already in results/)
    gen=Sonnet 4.6  × judge=GPT-4o     (already in cross_judge_report.json)
    gen=GPT-4o      × judge=Opus 4.7   (NEW)
    gen=GPT-4o      × judge=GPT-4o     (NEW)

For each cell, computes per-question accuracy and Spearman rank
correlation against the default cell. Writes
results/robustness_report.json.

Cost: ~$8 across top-5 strategies (3 seeds × 16 questions × 2 new gens
on the OpenAI side, plus the judge calls). Compute budget is the
gating factor, not engineering — re-running benchmarks for new
generators means re-running the strategy lifecycle (setup → ingest →
recall → teardown), which is the same code path as `memory-arena
benchmark`. This script wires that together with the
generator-as-parameter knob and writes a single report.

Status: shipped, not yet executed against live APIs. Documented as a
v0.1.6 deliverable in CHANGELOG.

Usage (when ready to spend the $8):
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \
        python scripts/robustness.py --top-k 5 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
REPORT = RESULTS / "robustness_report.json"

GEN_MODELS = ["claude-sonnet-4-6", "gpt-4o"]
JUDGE_MODELS = ["claude-opus-4-7", "gpt-4o"]


def _spearman(rank_a: list[str], rank_b: list[str]) -> float:
    if len(rank_a) != len(rank_b) or len(rank_a) < 2:
        return 0.0
    name_to_a = {s: i for i, s in enumerate(rank_a)}
    name_to_b = {s: i for i, s in enumerate(rank_b)}
    n = len(rank_a)
    diffs_sq = sum((name_to_a[s] - name_to_b[s]) ** 2 for s in rank_a)
    return 1 - (6 * diffs_sq) / (n * (n**2 - 1))


def _load_top_strategies(top_k: int) -> list[str]:
    summaries = []
    for p in RESULTS.glob("longmemeval-s_*_summary.json"):
        d = json.loads(p.read_text())
        summaries.append((d["strategy"], d.get("accuracy", 0.0)))
    summaries.sort(key=lambda t: -t[1])
    return [s for s, _ in summaries[:top_k]]


async def _run_cell(
    strategies: list[str],
    seeds: list[int],
    gen_model: str,
    judge_model: str,
) -> dict:
    """Re-run `memory-arena benchmark` programmatically with the given
    generator + judge override. Returns {strategy: accuracy_mean}.

    Implementation note: ``run_memory_benchmark`` writes per-strategy JSON
    files under ``results/`` (one per ``corpus_strategy[_seedN].json``)
    rather than returning the run dict. This loop iterates one strategy
    at a time, awaits the run, then reads the JSON it just produced to
    pick up accuracy. The settings object is mutated in-process; do not
    run this in parallel with other benchmark runs.
    """
    from memory_arena.benchmark.runner import run_memory_benchmark
    from memory_arena.settings import settings

    orig_gen = settings.generate_model
    orig_judge = settings.judge_model
    settings.generate_model = gen_model
    settings.judge_model = judge_model
    try:
        results: dict[str, list[float]] = {s: [] for s in strategies}
        for seed in seeds:
            for strategy in strategies:
                await run_memory_benchmark(
                    corpus="longmemeval-s",
                    strategy=strategy,
                    questions="smoke",
                    cost_cap=3.0,
                    top_k=5,
                    seed=seed,
                )
                seed_suffix = f"_seed{seed}"
                result_path = RESULTS / f"longmemeval-s_{strategy}{seed_suffix}.json"
                if not result_path.exists():
                    # The runner writes the unsuffixed file when seed is None;
                    # tolerate that as a fallback (e.g. tests passing seed=None).
                    result_path = RESULTS / f"longmemeval-s_{strategy}.json"
                if result_path.exists():
                    data = json.loads(result_path.read_text())
                    results[strategy].append(float(data.get("accuracy", 0.0)))
        return {s: sum(v) / len(v) for s, v in results.items() if v}
    finally:
        settings.generate_model = orig_gen
        settings.judge_model = orig_judge


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()

    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")):
        raise SystemExit("Both OPENAI_API_KEY and ANTHROPIC_API_KEY required.")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    strategies = _load_top_strategies(args.top_k)
    print(f"Robustness sweep: {len(strategies)} strategies × {len(seeds)} seeds × 4 cells")

    cells: dict[str, dict[str, float]] = {}
    for gen in GEN_MODELS:
        for judge in JUDGE_MODELS:
            cell_key = f"gen={gen}|judge={judge}"
            print(f"  [cell] {cell_key}")
            cells[cell_key] = await _run_cell(strategies, seeds, gen, judge)

    # Rank correlation: each cell's strategy ordering vs the default cell.
    default = "gen=claude-sonnet-4-6|judge=claude-opus-4-7"
    ranks = {key: sorted(d, key=lambda s: -d[s]) for key, d in cells.items()}
    correlations = {key: _spearman(ranks[default], ranks[key]) for key in cells if key != default}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "cells": cells,
                "ranks": ranks,
                "spearman_vs_default": correlations,
            },
            indent=2,
        )
    )
    print(f"\nWrote {REPORT}")
    for key, rho in correlations.items():
        print(f"  ρ vs default ({key}): {rho:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
