"""Shared visual language for the README charts.

Both `build_hero_chart.py` and `build_taxonomy_chart.py` import from
here so the two figures speak the same visual vocabulary:

  - Shape encodes CATEGORY: circle = pure-Python, triangle = vendor
    SDK, diamond = full_context (ceiling reference).
  - Base color encodes CATEGORY: navy / coral / grey.
  - Size + label weight encode TIER (>=35% / 20-34% / <20%). Tufte:
    pick one redundant encoding and stick to it; previous versions
    triple-encoded with saturation+alpha+size.
  - Markers are always filled. Hollow rings looked wrong on triangles.

The thresholds + colors live here too so a tier boundary can be moved
in exactly one place. Import everything from this module; do not
duplicate constants in the chart scripts.
"""

from __future__ import annotations

from matplotlib.lines import Line2D

# Category palette (Tufte / MBB navy + coral)
NAVY = "#1f3b73"
CORAL = "#e3614c"
GREY_HEX = "#7c7c7c"
# Quantum family gets its own hue + marker so qiss/sqr don't fall through to the
# grey "unknown" fallback. Purple is distinct from navy/coral under the common
# color-blindness types, and the star marker reads as a distinct fourth shape.
PURPLE = "#6a4c93"

# Strategy classification
PURE_PYTHON = {
    "full_context",  # also flagged as ceiling, see GREY below
    "recency_window",
    "naive_vector",
    "bm25",
    "hybrid_rrf",
    "hyde",
    "persona_profile",
    "reflection",
    "raptor",
    "karpathy_llm_wiki",
    "amem",
    "hipporag2",
}
VENDOR = {"mem0", "mem0g", "graphiti", "graphiti_falkor", "cognee", "langmem", "memori"}
# Quantum / quantum-inspired rerankers over naive_vector.
QUANTUM = {"qiss", "sqr"}
GREY = {"full_context"}  # singled out as the "stuff everything in" ceiling

# Tier thresholds (in 0..1 fraction) — change here to move the cut in
# every chart at once.
TIER1_MIN_ACC = 0.35
TIER2_MIN_ACC = 0.20

TIER_VISUAL = {
    "tier1": {
        "size_mult": 1.6,
        "alpha": 1.0,
        "saturation": 1.0,
        "label_weight": "bold",
        "label_size": 9,
        "label_color": "#1c1c1c",
    },
    "tier2": {
        "size_mult": 0.85,
        "alpha": 1.0,
        "saturation": 1.0,
        "label_weight": "normal",
        "label_size": 8,
        "label_color": "#5a6878",
    },
    "tier3": {
        "size_mult": 0.75,
        "alpha": 1.0,
        "saturation": 1.0,
        "label_weight": "normal",
        "label_size": 7.5,
        "label_color": "#8a93a3",
    },
}


def color_for(strategy: str) -> str:
    """Base CATEGORY color, before tier-saturation desaturation."""
    if strategy in GREY:
        return GREY_HEX
    if strategy in QUANTUM:
        return PURPLE
    if strategy in VENDOR:
        return CORAL
    if strategy in PURE_PYTHON:
        return NAVY
    # Unknown strategy: full 7-char hex so downstream desaturate() does
    # not crash on a 4-char shorthand.
    return "#888888"


def marker_for(strategy: str) -> str:
    """Matplotlib marker for the strategy's CATEGORY."""
    if strategy in GREY:
        return "D"
    if strategy in QUANTUM:
        return "*"
    if strategy in VENDOR:
        return "^"
    return "o"


def tier_for_acc(acc_fraction: float) -> str:
    if acc_fraction >= TIER1_MIN_ACC:
        return "tier1"
    if acc_fraction >= TIER2_MIN_ACC:
        return "tier2"
    return "tier3"


def desaturate(hex_color: str, t: float, grey_rgb: tuple = (160, 162, 168)) -> str:
    """Blend hex_color toward neutral grey. t=1 returns hex_color, t=0 returns grey."""
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    out = tuple(int(g + (c - g) * t) for c, g in zip(rgb, grey_rgb, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*out)


def category_legend_handles() -> list[Line2D]:
    """Three-entry legend (circle/triangle/diamond) for both charts.

    Identical styling so the two figures read as a set, not as siblings
    that grew up apart. Tier encoding is intentionally not in the legend
    — readers absorb it from the chart body via size/saturation, and the
    README captions verbalize it.
    """
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NAVY,
            markeredgecolor=NAVY,
            markersize=10,
            label="pure-Python (●)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=CORAL,
            markeredgecolor=CORAL,
            markersize=11,
            label="vendor SDK (▲)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=PURPLE,
            markeredgecolor=PURPLE,
            markersize=13,
            label="quantum reranker (★)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=GREY_HEX,
            markeredgecolor=GREY_HEX,
            markersize=9,
            label="full_context ceiling (◆)",
        ),
    ]
