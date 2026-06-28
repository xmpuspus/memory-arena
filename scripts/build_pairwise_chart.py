"""Generate the pairwise significance heatmap: which gaps are real?

For each (strategy_A, strategy_B) pair, we paired-bootstrap the per-question
accuracy difference across all available seeds and questions. Each cell
shows the mean difference (A - B). Colour encodes both direction and
statistical significance:

  blue    A beats B by >5 pp AND 95% bootstrap CI excludes 0  (real win)
  red     A loses to B by >5 pp AND CI excludes 0             (real loss)
  white   gap is < 5 pp OR CI includes 0                       (statistical tie)

Reading the chart: pick a row (A). Blue cells in that row are strategies A
beats convincingly; red cells are strategies that beat A convincingly;
white cells are ties.

Output: docs/pairwise.png (1500x1300 @ 144 dpi).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"
OUT = DOCS / "pairwise.png"

N_BOOTSTRAP = 2000
RNG_SEED = 7
MIN_REAL_GAP_PP = 5.0


def _load_per_question(strategy: str) -> dict[str, list[float]]:
    """question_id -> [accuracy across all seeds]."""
    out: dict[str, list[float]] = defaultdict(list)
    for p in sorted(RESULTS.glob(f"longmemeval-s_{strategy}_seed*.json")):
        d = json.loads(p.read_text())
        for r in d.get("recall_records", []):
            score = r.get("score", {})
            if isinstance(score, dict):
                out[r["question_id"]].append(float(score.get("accuracy", 0.0)))
    if not out:
        # Fall back to single-seed file (no _seedN suffix).
        p = RESULTS / f"longmemeval-s_{strategy}.json"
        if p.exists():
            d = json.loads(p.read_text())
            for r in d.get("recall_records", []):
                score = r.get("score", {})
                if isinstance(score, dict):
                    out[r["question_id"]].append(float(score.get("accuracy", 0.0)))
    return out


def _all_strategies() -> list[str]:
    out = []
    for p in sorted(RESULTS.glob("longmemeval-s_*_summary.json")):
        d = json.loads(p.read_text())
        out.append((d["strategy"], d.get("accuracy", 0.0)))
    out.sort(key=lambda t: -t[1])
    return [s for s, _ in out]


def _paired_bootstrap_diff(
    a_per_q: dict[str, list[float]],
    b_per_q: dict[str, list[float]],
    rng: random.Random,
) -> tuple[float, float, float]:
    """Returns (mean_diff_pp, ci_lo_pp, ci_hi_pp) for accuracy(A) - accuracy(B).

    Pairs are by question_id. Each iteration re-samples the question set
    with replacement and within each question takes the mean across the
    available seeds for both A and B.
    """
    qids = sorted(set(a_per_q) & set(b_per_q))
    if not qids:
        return 0.0, 0.0, 0.0

    obs_diff = sum(np.mean(a_per_q[q]) - np.mean(b_per_q[q]) for q in qids) / len(qids)
    boots: list[float] = []
    for _ in range(N_BOOTSTRAP):
        sample = [qids[rng.randrange(len(qids))] for _ in qids]
        boots.append(sum(np.mean(a_per_q[q]) - np.mean(b_per_q[q]) for q in sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * N_BOOTSTRAP)]
    hi = boots[int(0.975 * N_BOOTSTRAP)]
    return obs_diff * 100, lo * 100, hi * 100


def _color_cell(mean_pp: float, lo: float, hi: float) -> str:
    """White = tie; navy = significant win; coral = significant loss."""
    sig = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    if not sig or abs(mean_pp) < MIN_REAL_GAP_PP:
        return "#f4f6fa"
    if mean_pp > 0:
        # Saturation by magnitude
        intensity = min(1.0, abs(mean_pp) / 30.0)
        return _blend("#1f3b73", "#cdd9eb", intensity)
    intensity = min(1.0, abs(mean_pp) / 30.0)
    return _blend("#e3614c", "#fbe1da", intensity)


def _blend(color_full: str, color_base: str, t: float) -> str:
    """t=1 -> color_full, t=0 -> color_base."""
    full = tuple(int(color_full[i : i + 2], 16) for i in (1, 3, 5))
    base = tuple(int(color_base[i : i + 2], 16) for i in (1, 3, 5))
    out = tuple(int(b + (f - b) * t) for f, b in zip(full, base, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*out)


def build() -> Path:
    rng = random.Random(RNG_SEED)
    strategies = _all_strategies()
    n = len(strategies)
    per_q = {s: _load_per_question(s) for s in strategies}

    # Compute the matrix.
    mean_mat = np.zeros((n, n))
    sig_mat = np.zeros((n, n), dtype=bool)
    for i, a in enumerate(strategies):
        for j, b in enumerate(strategies):
            if i == j:
                continue
            mean, lo, hi = _paired_bootstrap_diff(per_q[a], per_q[b], rng)
            mean_mat[i, j] = mean
            sig_mat[i, j] = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)

    fig, ax = plt.subplots(figsize=(13.0, 11.3), dpi=170)  # ~2080x1808 trimmed -> >=1600 each side
    tie_color = "#f4f6fa"
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: "no comparison possible." Blend into tie color
                # rather than draw a dark block, which previously read as
                # "this cell is significant" to the eye.
                ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=tie_color))
                ax.text(
                    j + 0.5,
                    n - 1 - i + 0.5,
                    "-",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#a8b0bd",
                )
                continue
            mean_pp = mean_mat[i, j]
            color = tie_color
            sig = sig_mat[i, j]
            if sig and abs(mean_pp) >= MIN_REAL_GAP_PP:
                if mean_pp > 0:
                    intensity = min(1.0, abs(mean_pp) / 30.0)
                    color = _blend("#1f3b73", "#cdd9eb", intensity)
                else:
                    intensity = min(1.0, abs(mean_pp) / 30.0)
                    color = _blend("#e3614c", "#fbe1da", intensity)
            # No cell border (Tufte: removed white separators that
            # carried no information).
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=color))
            label = f"{mean_pp:+.0f}"
            # Standardize text color to dark ink; the cell color already
            # encodes magnitude. Removing the white-on-saturated branch
            # eliminates a design-variation channel that wasn't data-bearing.
            ax.text(
                j + 0.5,
                n - 1 - i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="#1c1c1c",
                weight="bold" if (sig and abs(mean_pp) >= 18) else "normal",
            )

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels(strategies, rotation=60, ha="right", fontsize=8)
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels(strategies, fontsize=8)
    ax.set_aspect("equal")
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)

    ax.set_title(
        "Pairwise accuracy gap (row − column, percentage points)\n"
        "navy = row beats column significantly · coral = column beats row · "
        "white = tie",
        fontsize=10,
        loc="left",
        pad=12,
    )
    ax.text(
        0,
        -2.4,
        "Significance: paired bootstrap of per-question diffs (n=2000 resamples), "
        "95% CI excludes 0 AND |gap| ≥ 5 pp.",
        fontsize=8,
        color="#5a6878",
        style="italic",
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    return OUT


if __name__ == "__main__":
    build()
