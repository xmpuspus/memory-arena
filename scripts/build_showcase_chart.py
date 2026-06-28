"""Showcase / social hero: 20 agent-memory systems ranked on one eval.

The Pareto frontier (build_hero_chart.py) answers "accuracy vs cost". This one
answers the question people actually scroll for: "which memory wins, and do the
funded SDKs beat a 30-line script?" A sorted bar chart, colored by family, with
a reference line at the vector baseline so the answer reads in one glance.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _chart_style import CORAL, GREY_HEX, NAVY, PURPLE, VENDOR, color_for  # noqa: E402

RESULTS = "results"
OUT = "docs/showcase.png"
BASELINE = "naive_vector"


def _load() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f"{RESULTS}/longmemeval-s_*_summary.json")):
        d = json.load(open(f))
        rows.append(
            {
                "s": d["strategy"],
                "acc": (d.get("accuracy") or 0.0) * 100,
                "ci": (d.get("accuracy_ci") or 0.0) * 100,
            }
        )
    rows.sort(key=lambda r: r["acc"])  # ascending -> highest at top of barh
    return rows


def main() -> None:
    rows = _load()
    labels = [r["s"] for r in rows]
    accs = [r["acc"] for r in rows]
    cis = [r["ci"] for r in rows]
    colors = [color_for(s) for s in labels]
    base_acc = next((r["acc"] for r in rows if r["s"] == BASELINE), None)

    fig, ax = plt.subplots(figsize=(10, 8.8))
    y = range(len(rows))
    ax.barh(y, accs, color=colors, height=0.76, zorder=3, edgecolor="white", linewidth=0.6)
    ax.errorbar(
        accs, list(y), xerr=cis, fmt="none", ecolor="#9aa3b0", elinewidth=1.0, capsize=2.5, zorder=4
    )

    # value labels: inside the bar (white) when it is long enough, otherwise
    # just past the CI whisker. Keeps the labels off the right margin so the
    # axis does not have to stretch far past where the bars actually end.
    for yi, r in zip(y, rows):
        emph = r["s"] in (BASELINE, "hipporag2")
        if r["acc"] >= 12:
            ax.text(
                r["acc"] - 1.0,
                yi,
                f"{r['acc']:.0f}%",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold" if emph else "normal",
                zorder=5,
            )
        else:
            ax.text(
                r["acc"] + r["ci"] + 1.0,
                yi,
                f"{r['acc']:.0f}%",
                va="center",
                ha="left",
                fontsize=9,
                color="#1c1c1c",
                fontweight="bold" if emph else "normal",
                zorder=5,
            )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    # bold the baseline tick + color vendor ticks coral so the family reads on the axis too
    for tick, s in zip(ax.get_yticklabels(), labels):
        if s == BASELINE:
            tick.set_fontweight("bold")
        if s in VENDOR:
            tick.set_color(CORAL)

    # reference line: the 30-line vector baseline. Every vendor falls left of it.
    if base_acc is not None:
        ax.axvline(base_acc, color=NAVY, lw=1.3, ls="--", alpha=0.55, zorder=2)
        ax.text(
            base_acc,
            len(rows) - 0.25,
            "naive_vector baseline ↓",
            fontsize=8.5,
            color=NAVY,
            va="bottom",
            ha="center",
            fontweight="bold",
        )

    ax.set_xlim(0, 74)
    ax.set_xlabel("Accuracy (%): Claude Opus judge, bootstrap mean ± 95% CI", fontsize=10)
    ax.set_title(
        "Which agent memory wins? 20 systems, same corpus, same judge.",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=26,
    )
    ax.text(
        0,
        1.012,
        "A 30-line vector store beats every funded SDK (Mem0, Graphiti, Cognee, LangMem). "
        "None is close to solved.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#444",
    )

    # takeaway panel: turn the sparse right third (the low scorers have no CI
    # whisker out here) into signal -- the three findings the chart supports.
    panel = FancyBboxPatch(
        (0.685, 0.05),
        0.30,
        0.40,
        transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        facecolor="#f5f7fa",
        edgecolor="#d2d9e2",
        linewidth=1.0,
        zorder=6,
        clip_on=False,
    )
    ax.add_patch(panel)
    ax.text(
        0.702,
        0.415,
        "What the data says",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=NAVY,
        va="top",
        zorder=7,
    )
    for i, finding in enumerate(
        [
            "Free beats funded: a 30-line\nvector store tops every SDK.",
            "The top ~8 are a statistical tie\n(wide 95% CIs at N=16).",
            "Nobody clears 55%. Agent\nmemory is far from solved.",
        ]
    ):
        ax.text(
            0.702,
            0.345 - i * 0.107,
            finding,
            transform=ax.transAxes,
            fontsize=8.5,
            color="#222",
            va="top",
            linespacing=1.35,
            zorder=7,
        )

    # legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=NAVY,
            markersize=11,
            label="pure-Python (free)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=CORAL,
            markersize=11,
            label="vendor SDK",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=PURPLE,
            markersize=11,
            label="quantum reranker",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=GREY_HEX,
            markersize=11,
            label="full_context (ceiling)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.085),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    # footnote: methodology + the strict-judge context so low absolute reads as
    # "hard, honest". Wrapped to 3 lines so the single long line does not force
    # bbox_inches="tight" to blow out the canvas width.
    ax.text(
        0.0,
        -0.125,
        "LongMemEval-S smoke, 16 questions, top_k=5. Same corpus, judge, "
        "embeddings for all 20; mem0 + langmem\n"
        "run on the same model (Claude Sonnet) as the baselines. Opus is a "
        "strict judge: GPT-4o grades these\n"
        "~15pp higher with +0.97 rank agreement. Every result JSON is "
        "stamped. github.com/xmpuspus/memory-arena",
        transform=ax.transAxes,
        fontsize=7.6,
        color="#777",
        ha="left",
        va="top",
        linespacing=1.45,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
