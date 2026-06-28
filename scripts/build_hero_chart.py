"""Generate the README hero chart: accuracy vs cost Pareto frontier.

Reads results/longmemeval-s_<strategy>_summary.json (3-seed bootstrap output;
falls back to per-seed JSONs and aggregates inline if summary is absent), and
writes docs/hero.png at 1200x630 px / 144 dpi.

Tufte-leaning encoding (shape + color, redundant for color-blind safety):
  - circle ●  navy  : pure-Python baselines & retrievers (open-source)
  - triangle ▲ coral: vendor SDKs (Mem0, Graphiti, Cognee, LangMem, Memori)
  - diamond ◆ grey  : full_context (no retrieval; ceiling reference)
  - x         coral : config-failed-at-default (vendor's shipped default
                       broke ingest; result reflects an empty store)

Style notes:
  - thin spines on the left + bottom only (top + right removed — Tufte would
    delete unnecessary borders)
  - sparse gridlines, axis labels concise
  - direct strategy labels next to each point — no separate strategy legend,
    only a small shape/color legend in the lower-right
  - "winner quadrant" arrow kept (one annotation, hand-drawn-feeling, not chartjunk)

Usage:
    python scripts/build_hero_chart.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Shared visual language — see scripts/_chart_style.py.
sys.path.insert(0, str(Path(__file__).parent))
from _chart_style import (  # noqa: E402, I001
    TIER_VISUAL,
    category_legend_handles,
    color_for as _color_for,
    desaturate as _desaturate,
    marker_for as _marker_for,
    tier_for_acc as _tier_for_acc,
)


def load_summaries(corpus: str = "longmemeval-s") -> list[dict]:
    """Return one dict per strategy with mean accuracy, cost, and CI bounds.

    Prefers a bootstrap summary file if present; otherwise aggregates per-seed
    files inline using a 3-seed mean and 1.96 * (std/sqrt(n)) for CI.
    """
    summary_files = sorted(RESULTS_DIR.glob(f"{corpus}_*_summary.json"))
    if summary_files:
        out: list[dict] = []
        for p in summary_files:
            d = json.loads(p.read_text())
            # aggregate_bootstrap.py writes total_cost_usd; the chart's plot
            # logic expects cost_usd. Normalize here.
            d["cost_usd"] = d.get("total_cost_usd", 0.0)
            d["ci"] = d.get("accuracy_ci", 0.0)
            out.append(d)
        return out

    per_seed: dict[str, list[dict]] = {}
    for p in sorted(RESULTS_DIR.glob(f"{corpus}_*_seed*.json")):
        # filename pattern: <corpus>_<strategy>_seed<N>.json
        stem = p.stem  # longmemeval-s_naive_vector_seed0
        prefix = f"{corpus}_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix) :]
        # rest = naive_vector_seed0
        if "_seed" not in rest:
            continue
        strategy_name = rest.rsplit("_seed", 1)[0]
        per_seed.setdefault(strategy_name, []).append(json.loads(p.read_text()))

    # If no _seedN files, fall back to plain results files (single-seed v0.1.4 style)
    if not per_seed:
        for p in sorted(RESULTS_DIR.glob(f"{corpus}_*.json")):
            if "_seed" in p.stem or p.stem.endswith("_summary"):
                continue
            stem = p.stem
            prefix = f"{corpus}_"
            if not stem.startswith(prefix):
                continue
            strategy_name = stem[len(prefix) :]
            per_seed.setdefault(strategy_name, []).append(json.loads(p.read_text()))

    out: list[dict] = []
    for strategy, runs in per_seed.items():
        accs = [r.get("accuracy", 0.0) or 0.0 for r in runs]
        costs = [r.get("total_cost_usd", 0.0) or 0.0 for r in runs]
        statuses = {r.get("status", "ok") for r in runs}
        n = len(accs)
        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_cost = float(np.mean(costs)) if costs else 0.0
        if n > 1:
            sem = float(np.std(accs, ddof=1) / math.sqrt(n))
            ci = 1.96 * sem
        else:
            # Single-seed: no CI; render zero-width error bar.
            ci = 0.0
        status = "config-failed-at-default" if "config-failed-at-default" in statuses else "ok"
        out.append(
            {
                "strategy": strategy,
                "accuracy": mean_acc,
                "ci": ci,
                "cost_usd": mean_cost,
                "n_seeds": n,
                "status": status,
            }
        )
    return out


def build_chart() -> Path:
    summaries = load_summaries()
    # All benchmarked strategies are shown for transparency, quantum included.
    # qiss lands on top of naive_vector (it IS naive_vector: cos^2 can't reorder
    # cosine) and sqr trails; the caption says so, so the purple stars read as
    # "no quantum advantage" rather than a win. The dedicated charts
    # (docs/quantum_experiments.png, docs/compression_frontier.png) carry the why.
    if not summaries:
        raise SystemExit(
            "No result files found. Run `memory-arena benchmark --corpus longmemeval-s "
            "--seed 0` (and seeds 1, 2) first."
        )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)  # 1920x1080

    # Avoid log10(0) on cost. Floor cost at $0.001 (a tenth of a cent) for plot
    # purposes; the table in the README shows the raw $0.0000 separately.
    for s in summaries:
        s["plot_cost"] = max(s["cost_usd"], 0.001)

    # Shade the "best" quadrant: high accuracy AND low cost (top-left).
    # Threshold: cost <= $0.10, accuracy >= 30 pp. Keeps mem0/mem0g and the
    # cheap pure-Python retrievers in-zone, leaves full_context (ceiling)
    # and the cloud-throttled vendors clearly out. Very low alpha so the
    # data points stay primary; no border (Tufte: avoid chartjunk).
    # Cap the zone's top at observed-max + small headroom so the shaded
    # area doesn't sprawl into empty space (no strategy is above ~42%).
    best_cost_usd = 0.10
    best_acc_pp = 30.0
    observed_max_acc = max(s["accuracy"] * 100 for s in summaries)
    zone_top = min(100.0, observed_max_acc + 5.0)
    ax.fill_between(
        [1e-4, best_cost_usd],
        best_acc_pp,
        zone_top,
        color="#7fbf7b",
        alpha=0.13,
        linewidth=0,
        zorder=0,
    )
    ax.text(
        0.0011,
        zone_top - 0.5,
        "best: high accuracy / low cost",
        fontsize=8,
        color="#2f6b2f",
        style="italic",
        ha="left",
        va="top",
        alpha=0.9,
        zorder=1,
    )

    failed = [s for s in summaries if s["status"] == "config-failed-at-default"]
    ok = [s for s in summaries if s["status"] == "ok"]

    # Render order: tier 3 first (back), tier 2 next, tier 1 last (front).
    # Z-order matches saturation hierarchy so the winners physically sit
    # on top of context.
    ok_by_tier = {"tier3": [], "tier2": [], "tier1": []}
    for s in ok:
        ok_by_tier[_tier_for_acc(s["accuracy"])].append(s)

    for tier_key in ("tier3", "tier2", "tier1"):
        tv = TIER_VISUAL[tier_key]
        z = 2 if tier_key == "tier3" else (3 if tier_key == "tier2" else 4)
        for s in ok_by_tier[tier_key]:
            cat_color = _color_for(s["strategy"])
            color = _desaturate(cat_color, tv["saturation"])
            mk = _marker_for(s["strategy"])
            # Stars read smaller at a given ms; bump so they match the others.
            base_ms = 14 if mk == "*" else (11 if mk in {"^", "D"} else 9)
            ax.errorbar(
                s["plot_cost"],
                s["accuracy"] * 100,
                yerr=s["ci"] * 100,
                fmt=mk,
                ms=base_ms * tv["size_mult"],
                color=color,
                ecolor=color,
                elinewidth=1.2 if tier_key == "tier1" else 0.9,
                capsize=2 if tier_key == "tier1" else 1.5,
                alpha=tv["alpha"],
                zorder=z,
            )
    for s in failed:
        ax.scatter(
            s["plot_cost"],
            s["accuracy"] * 100,
            marker="x",
            s=120,
            color=_color_for(s["strategy"]),
            alpha=0.85,
            linewidths=2.2,
            zorder=2,
        )

    # Direct labeling, Tufte-style. Two collision-handling rules:
    #   1. Strategies at near-identical (cost, accuracy) get merged into one
    #      label ("mem0 / mem0g").
    #   2. Strategies at the same accuracy band but different cost (e.g.
    #      cognee at $0.02 vs graphiti at $0.03 both at ~19%) get their
    #      labels alternated above/below the point so they don't read as
    #      one jammed-together blob.
    from math import log10

    def _bucket(s: dict) -> tuple[int, int]:
        """Hash a point into a coarse (log_cost, accuracy) bucket. Same
        bucket → merge into one label (catches mem0/mem0g)."""
        return (round(log10(max(s["cost_usd"], 0.001)) * 6), round(s["accuracy"] * 30))

    buckets: dict[tuple[int, int], list[dict]] = {}
    for s in summaries:
        buckets.setdefault(_bucket(s), []).append(s)

    # Group merged-bucket points by accuracy band so we can alternate
    # label-above vs label-below within each horizontal band.
    # Each entry now carries the tier (max tier within a merged group)
    # so we can match label weight/color to the marker emphasis.
    bands: dict[int, list[tuple[float, float, str, str]]] = {}
    for group in buckets.values():
        plot_cost = sum(s["plot_cost"] for s in group) / len(group)
        accuracy = sum(s["accuracy"] for s in group) / len(group)
        label = " / ".join(s["strategy"] for s in group)
        # If a merged group has any tier-1 member, render the whole label
        # as tier 1 (mem0/mem0g case — both are tier 1, no demotion).
        tiers = [_tier_for_acc(s["accuracy"]) for s in group]
        merged_tier = "tier1" if "tier1" in tiers else ("tier2" if "tier2" in tiers else "tier3")
        band_key = round(accuracy * 30)  # ~3pp accuracy bins
        bands.setdefault(band_key, []).append((plot_cost, accuracy, label, merged_tier))

    for band_points in bands.values():
        # Sort by cost so the leftmost gets one offset, next the alternate.
        band_points.sort(key=lambda t: t[0])
        for i, (plot_cost, accuracy, label, tier_key) in enumerate(band_points):
            tv = TIER_VISUAL[tier_key]
            if len(band_points) > 1 and i % 2 == 1:
                # Below the point — for the second/fourth/etc. label in a band.
                offset = (8, -10)
            else:
                offset = (8, 5)
            ax.annotate(
                label,
                xy=(plot_cost, accuracy * 100),
                xytext=offset,
                textcoords="offset points",
                fontsize=tv["label_size"],
                color=tv["label_color"],
                weight=tv["label_weight"],
                zorder=5 if tier_key == "tier1" else 3,
            )

    ax.set_xscale("log")
    # Y-axis: include the worst-case CI top so error bars are not visually
    # truncated (Tufte: lie factor must = 1). Add a small headroom for labels.
    max_ci_top = max((s["accuracy"] * 100 + s["ci"] * 100) for s in summaries)
    ax.set_ylim(-2, max(60.0, max_ci_top + 5.0))
    ax.set_xlabel("Total cost (USD, log scale): measurable component only")
    seed_counts = {s.get("n_seeds", 1) for s in summaries}
    if seed_counts == {1}:
        seed_label = "single seed"
    elif len(seed_counts) == 1:
        seed_label = f"mean of {next(iter(seed_counts))} seeds, 95% CI"
    else:
        # Mixed seed counts across the board (e.g. some 3-seed, some 5-seed,
        # some single): a single number would be misleading.
        seed_label = "bootstrap mean, 95% CI"
    ax.set_ylabel(f"Accuracy (%): judge score, {seed_label}")
    # Derive N from the actual smoke run rather than hardcoding it; if the
    # underlying corpus grows, the title tracks.
    n_questions = max((s.get("n_questions") or 16) for s in summaries) if summaries else 16
    ax.set_title(
        f"Memory Arena: Pareto frontier on LongMemEval-S "
        f"(smoke, {n_questions} questions, top_k=5)",
        fontsize=11,
        loc="left",
        pad=10,
    )

    # Tufte spine cleanup — remove the box, keep only the data-bearing axes.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.tick_params(colors="#666", which="both")

    # Sparser, lower-ink gridlines (Tufte: erase non-data-ink).
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.12)
    ax.set_axisbelow(True)
    # No "winner quadrant →" annotation — the axes ("Accuracy ↑" + "Cost log
    # scale →") already tell the reader that top-left dominates. Adding text
    # is chartjunk and risks overlapping data points (caught by Xavier's
    # review, 2026-04-30).

    # Shared category legend (matches the taxonomy chart). Frame removed
    # per Tufte: frames around a 3-item legend are non-data-ink. Tier
    # encoding is absorbed from the chart body via marker size + label
    # weight.
    ax.legend(
        handles=category_legend_handles(),
        loc="lower left",
        fontsize=8,
        frameon=False,
        handletextpad=0.4,
        borderpad=0.6,
    )

    # Lock dimensions: don't pass bbox_inches="tight" because adjustText
    # extends the labels outside the original bounding box, ballooning the
    # rendered file (~848KB). Stick with the figure's declared figsize.
    out = DOCS_DIR / "hero.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return out


if __name__ == "__main__":
    build_chart()
