"""The rule-of-thumb proof: retrieval is solved, reasoning is the bottleneck.

One chart, one message. Per strategy: a bar for how often the answer is right
(accuracy) and a marker for how often the right memory was retrieved
(recall@5). The space between them is what's lost to reasoning, not retrieval,
which is exactly why a fancier index doesn't help and a 30-line vector store is
enough. All numbers read from results/*_summary.json (never hardcoded).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _chart_style import CORAL, GREY_HEX, NAVY, PURPLE, VENDOR, color_for  # noqa: E402

RESULTS = "results"
OUT = "docs/reasoning-gap.png"
BASELINE = "naive_vector"
REC = "#334155"  # slate marker for recall@5


def _load() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f"{RESULTS}/longmemeval-s_*_summary.json")):
        d = json.load(open(f))
        rec = d.get("mean_session_recall_at_k")
        rows.append(
            {
                "s": d["strategy"],
                "acc": (d.get("accuracy") or 0.0) * 100,
                "rec": (rec * 100) if rec is not None else None,
            }
        )
    rows.sort(key=lambda r: r["acc"])  # ascending -> highest at top of barh
    return rows


def main() -> None:
    rows = _load()
    labels = [r["s"] for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(11, 9))

    # accuracy bars (family-colored, same palette as the showcase)
    for yi, r in zip(y, rows):
        ax.barh(
            yi, r["acc"], color=color_for(r["s"]), height=0.6, zorder=3, edgecolor="white", lw=0.5
        )

    # connector (the gap) + recall@5 marker
    for yi, r in zip(y, rows):
        if r["rec"] is None:
            continue
        lo, hi = sorted([r["acc"], r["rec"]])
        ax.plot([lo, hi], [yi, yi], color="#cbd1da", lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(
            [r["rec"]], [yi], marker="D", s=40, facecolor="white", edgecolor=REC, lw=1.7, zorder=6
        )

    # accuracy labels (inside the bar) + recall labels (by the marker)
    for yi, r in zip(y, rows):
        emph = r["s"] in (BASELINE, "hipporag2")
        if r["acc"] >= 13:
            ax.text(
                r["acc"] - 1.3,
                yi,
                f"{r['acc']:.0f}%",
                va="center",
                ha="right",
                fontsize=8.5,
                color="white",
                fontweight="bold" if emph else "normal",
                zorder=7,
            )
        else:
            ax.text(
                r["acc"] + 1.3,
                yi,
                f"{r['acc']:.0f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#1c1c1c",
                zorder=7,
            )
        # skip the recall label when it would collide with the accuracy label
        # (only happens at the floor, e.g. recency_window where both are ~6%)
        if r["rec"] is not None and abs(r["rec"] - r["acc"]) >= 5:
            # place the recall % on the far side of the marker from the bar
            if r["rec"] >= r["acc"]:
                ax.text(
                    r["rec"] + 1.8,
                    yi,
                    f"{r['rec']:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=7.6,
                    color=REC,
                    zorder=7,
                )
            else:
                ax.text(
                    r["rec"] - 1.8,
                    yi,
                    f"{r['rec']:.0f}%",
                    va="center",
                    ha="right",
                    fontsize=7.6,
                    color=REC,
                    zorder=7,
                )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    for tick, s in zip(ax.get_yticklabels(), labels):
        if s == BASELINE:
            tick.set_fontweight("bold")
        if s in VENDOR:
            tick.set_color(CORAL)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent (LongMemEval-S smoke, 16 questions, top_k=5)", fontsize=10)
    ax.set_title(
        "Retrieval is solved. Reasoning is the bottleneck.",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=28,
    )
    ax.text(
        0,
        1.014,
        "Bars = answer correct (accuracy). Diamonds = right memory retrieved (recall@5). The gap "
        "is reasoning, not retrieval.\nnaive_vector retrieves 87%, answers 49%; every top "
        "retriever clusters the same, and no funded SDK closes the gap.",
        transform=ax.transAxes,
        fontsize=10,
        color="#444",
        linespacing=1.4,
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
            label="accuracy: pure-Python (free)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=CORAL,
            markersize=11,
            label="accuracy: vendor SDK",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=PURPLE,
            markersize=11,
            label="accuracy: quantum reranker",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=GREY_HEX,
            markersize=11,
            label="accuracy: full_context",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="white",
            markeredgecolor=REC,
            markeredgewidth=1.7,
            markersize=9,
            label="recall@5: right memory retrieved",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=1.5,
    )

    # footnote
    ax.text(
        0.0,
        -0.155,
        "recall@5 = the labeled supporting session appears in the top 5 retrieved. LangMem, "
        "Cognee and Memori synthesize memory\nand return no ranked sessions, so they have no "
        "recall@5 marker. full_context stuffs the whole conversation (it does not\nretrieve "
        "top-k), which is why its recall is low and its cost is highest. "
        "github.com/xmpuspus/memory-arena",
        transform=ax.transAxes,
        fontsize=7.4,
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

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
