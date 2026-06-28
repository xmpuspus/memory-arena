"""Quantum experiment figure: the honest negative result in one panel.

Reads results/longmemeval-s_quantum_experiments.json and plots sqr's Recall@5
against n_qubits with the cosine baseline as a ceiling line, annotated with the
PCA variance kept at each qubit budget. The story it tells: more qubits recover
retrieval (tracking variance) but the SWAP-test reranker never beats the
closed-form cosine.

Output: docs/quantum_experiments.png (1600x900 @ 150 dpi).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _chart_style import NAVY, PURPLE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "longmemeval-s_quantum_experiments.json"
OUT = REPO_ROOT / "docs" / "quantum_experiments.png"


def build() -> Path:
    data = json.loads(RESULTS.read_text())
    overall = data["overall"]
    variance = data["pca_variance"]
    qubits = sorted(int(k) for k in variance)
    sqr_recall = [overall[f"sqr_q{nq}"]["recall@5"] for nq in qubits]
    var_pct = [variance[str(nq)] for nq in qubits]
    cosine = overall["cosine"]["recall@5"]

    fig, ax = plt.subplots(figsize=(10.7, 6.0), dpi=150)

    # Cosine ceiling (the closed-form baseline qiss-single reproduces exactly).
    ax.axhline(cosine, color=NAVY, linestyle="--", linewidth=1.4, zorder=1)
    ax.text(
        qubits[-1],
        cosine + 0.012,
        f"cosine baseline (naive_vector / qiss-single) = {cosine:.2f}",
        color=NAVY,
        fontsize=9,
        ha="right",
        va="bottom",
        weight="bold",
    )

    # sqr Recall@5 vs n_qubits.
    ax.plot(qubits, sqr_recall, color=PURPLE, linewidth=1.6, zorder=2)
    ax.scatter(
        qubits,
        sqr_recall,
        marker="*",
        s=320,
        color=PURPLE,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    for nq, rec, vp in zip(qubits, sqr_recall, var_pct, strict=True):
        ax.annotate(
            f"{rec:.2f}\n({vp:.0%} var)",
            xy=(nq, rec),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#5a4b78",
        )

    ax.set_xticks(qubits)
    ax.set_xticklabels([f"{nq} qubits\n({2**nq}d)" for nq in qubits], fontsize=9)
    ax.set_xlabel("SWAP-test register size (more qubits = more embedding variance kept)")
    ax.set_ylabel("Session Recall@5 (deterministic, exact)")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(qubits[0] - 0.4, qubits[-1] + 0.4)

    ax.set_title(
        "sqr recovers retrieval as qubits rise, but never beats the closed-form cosine",
        fontsize=12,
        loc="left",
        pad=10,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.tick_params(colors="#666")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.12)
    ax.set_axisbelow(True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    return OUT


if __name__ == "__main__":
    build()
