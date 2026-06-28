"""Generate the agent-memory taxonomy figure: 2D placement of all 21 strategies.

Two axes that capture the design space:

  X (horizontal): WHEN computation happens.
                  Left  = read-time (cheap to ingest, expensive to query)
                  Right = write-time (expensive to ingest, cheap to query)

  Y (vertical):  WHAT the storage representation is.
                  Bottom = raw text / token windows
                  Middle = vector embeddings
                  Upper  = derived facts (extracted statements)
                  Top    = graph (entities + edges)

Visual language is shared with the hero chart via _chart_style.py:

  Shape  = CATEGORY  (circle = pure-Python, triangle = vendor SDK,
                      diamond = full_context ceiling)
  Color  = CATEGORY  (navy / coral / grey at base saturation)
  Size   = TIER      (≥35% accuracy: large; 20–34%: medium; <20%: small)
  Sat.   = TIER      (≥35%: full; 20–34%: 50%; <20%: 28% — toward grey)
  Label  = TIER      (≥35%: bold black; lower tiers: muted grey)

The figure is the "skinny diagram" that should travel — readers learn the
mental model in one glance and place new strategies on it without re-reading.

Output: docs/taxonomy.png (1400x900 @ 144 dpi).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Shared visual language — see scripts/_chart_style.py.
sys.path.insert(0, str(Path(__file__).parent))
from _chart_style import (  # noqa: E402, I001
    NAVY,
    TIER_VISUAL,
    category_legend_handles,
    color_for as _color_for,
    desaturate as _desaturate,
    marker_for as _marker_for,
    tier_for_acc as _tier,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
RESULTS = REPO_ROOT / "results"
OUT = DOCS / "taxonomy.png"

GRID = "#e6e9ee"

# (x, y) in 0..1 unit space. (0,0) bottom-left = read-time + raw text.
# `side` controls whether the label sits to the left or right of the marker.
PLACEMENTS: dict[str, tuple[float, float, str]] = {
    # raw text / token-window strategies (bottom band)
    "full_context": (0.05, 0.08, "right"),
    "recency_window": (0.18, 0.05, "right"),
    "bm25": (0.30, 0.18, "right"),
    # vector strategies (middle-bottom band)
    "naive_vector": (0.50, 0.32, "right"),
    "hyde": (0.32, 0.42, "right"),
    "hybrid_rrf": (0.40, 0.30, "left"),
    # quantum rerankers over naive_vector's vector store: same representation
    # (vector band), a touch more read-time work (they rerank per query).
    "qiss": (0.55, 0.42, "right"),
    "sqr": (0.58, 0.25, "right"),
    # derived-text strategies (middle-upper band)
    "raptor": (0.62, 0.50, "left"),
    "reflection": (0.70, 0.45, "left"),
    "persona_profile": (0.65, 0.58, "right"),
    "karpathy_llm_wiki": (0.78, 0.66, "right"),
    # extracted facts (upper band)
    "langmem": (0.79, 0.52, "right"),
    "memori": (0.93, 0.62, "left"),
    "mem0": (0.74, 0.72, "left"),
    # graph strategies (top band)
    "graphiti": (0.83, 0.82, "right"),
    "graphiti_falkor": (0.72, 0.92, "left"),
    "cognee": (0.94, 0.92, "left"),
    # new in v0.1.7 — A-MEM is a structured LLM-maintained note system, HippoRAG 2
    # is a graph+vector hybrid with personalized PageRank
    "amem": (0.86, 0.70, "right"),
    "hipporag2": (0.70, 0.76, "left"),
}


def _load_accuracies() -> dict[str, float]:
    """Return only strategies that actually have a result file.

    Strategies registered in code but never benchmarked (e.g. amem,
    hipporag2 before v0.1.7 sweep) are omitted; the taxonomy figure
    skips them rather than rendering a misleading 0% marker.
    """
    out: dict[str, float] = {}
    for p in RESULTS.glob("longmemeval-s_*_summary.json"):
        d = json.loads(p.read_text())
        out[d["strategy"]] = d.get("accuracy", 0.0)
    return out


def build() -> Path:
    accs = _load_accuracies()
    fig, ax = plt.subplots(figsize=(13.5, 8.7), dpi=160)  # ~2160x1390 trimmed → ≥1920 wide

    # Hairline rules at band boundaries. Same Y-axis information as
    # an alternating-fill background, much less ink — Tufte: maximize
    # data-ink ratio.
    bands = [
        (0.00, 0.22, "raw text / window"),
        (0.22, 0.48, "vector embeddings"),
        (0.48, 0.70, "derived text"),
        (0.70, 1.00, "graph / structured"),
    ]
    for y0, _, _ in bands[1:]:  # skip the first boundary (chart bottom)
        ax.axhline(y0, xmin=0.0, xmax=1.0, color=GRID, linewidth=0.6, zorder=0)
    for _, y1, label in bands:
        ax.text(
            0.005,
            y1 - 0.012,
            label,
            fontsize=8,
            color="#a8b0bd",
            style="italic",
            va="top",
            ha="left",
            zorder=1,
        )

    # Render order matches saturation hierarchy: tier 3 first (back),
    # tier 1 last (front). Z-order reinforces who wins the eye.
    # Strategies with no result file are dropped (not rendered as 0%
    # which would silently fabricate data).
    by_tier = {"tier3": [], "tier2": [], "tier1": []}
    skipped: list[str] = []
    for name, (x, y, side) in PLACEMENTS.items():
        if name not in accs:
            skipped.append(name)
            continue
        by_tier[_tier(accs[name])].append((name, x, y, side))
    if skipped:
        print(f"taxonomy: skipping (no result file): {', '.join(skipped)}")

    base_size = 200  # tier-1 markers are this × size_mult
    for tier_key in ("tier3", "tier2", "tier1"):
        tv = TIER_VISUAL[tier_key]
        z = 2 if tier_key == "tier3" else (3 if tier_key == "tier2" else 4)
        for name, x, y, side in by_tier[tier_key]:
            acc = accs.get(name, 0.0)
            cat_color = _color_for(name)
            color = _desaturate(cat_color, tv["saturation"])
            mk = _marker_for(name)
            # Stars read smaller at a given area; scale them up to match.
            marker_size = base_size * tv["size_mult"] * (1.7 if mk == "*" else 1.0)
            ax.scatter(
                x,
                y,
                marker=mk,
                s=marker_size,
                color=color,
                edgecolor="white",
                linewidth=1.4,
                alpha=tv["alpha"],
                zorder=z,
            )

            # Two-line label for tier 1 (name in bold black + accuracy in
            # bold navy). Single-line for lower tiers.
            offset_x = 0.013 if side == "right" else -0.013
            ha = "left" if side == "right" else "right"
            if tier_key == "tier1":
                ax.text(
                    x + offset_x,
                    y + 0.012,
                    name,
                    fontsize=tv["label_size"],
                    color=tv["label_color"],
                    weight=tv["label_weight"],
                    ha=ha,
                    va="center",
                    zorder=5,
                )
                ax.text(
                    x + offset_x,
                    y - 0.018,
                    f"{acc * 100:.0f}%",
                    fontsize=tv["label_size"] + 1,
                    color=NAVY,
                    weight="bold",
                    ha=ha,
                    va="center",
                    zorder=5,
                )
            else:
                ax.text(
                    x + offset_x,
                    y,
                    f"{name}\n{acc * 100:.0f}%",
                    fontsize=tv["label_size"],
                    color=tv["label_color"],
                    weight=tv["label_weight"],
                    ha=ha,
                    va="center",
                    zorder=3,
                )

    # Axis labels.
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xticks([0.05, 0.95])
    ax.set_xticklabels(["read-time work", "write-time work"], fontsize=10, color="#3a4555")
    ax.set_yticks([0.05, 0.95])
    ax.set_yticklabels(
        ["raw text", "graph"],
        fontsize=10,
        color="#3a4555",
        rotation=90,
        va="center",
    )

    # Big arrow hints
    ax.annotate(
        "",
        xy=(0.99, -0.03),
        xytext=(0.01, -0.03),
        arrowprops=dict(arrowstyle="->", color=GRID, lw=1.0),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-0.02, 1.04),
        xytext=(-0.02, -0.02),
        arrowprops=dict(arrowstyle="->", color=GRID, lw=1.0),
        annotation_clip=False,
    )

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    n_plotted = sum(len(v) for v in by_tier.values())
    title = (
        f"Memory Arena: agent-memory architecture taxonomy "
        f"({n_plotted} of {len(PLACEMENTS)} strategies plotted)"
    )
    ax.set_title(
        title,
        fontsize=12,
        loc="left",
        pad=12,
    )

    # Shared category legend (matches the hero chart). Frame removed
    # per Tufte: a 3-item legend doesn't need a box.
    ax.legend(
        handles=category_legend_handles(),
        loc="lower right",
        fontsize=8,
        frameon=False,
        handletextpad=0.4,
        borderpad=0.6,
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    return OUT


if __name__ == "__main__":
    build()
