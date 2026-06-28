"""Embedding compression cost frontier: recall@5 vs bytes-per-vector.

Connects the quantum-reranker result to the real production cost lever. In a RAG
pipeline the similarity computation (where quantum plays) is negligible; the cost
drivers are LLM tokens and, for retrieval specifically, embedding storage / memory
bandwidth / ANN search at scale. The lever there is COMPRESSION, not the
similarity measure. This maps the honest frontier:

  rank all turns in the COMPRESSED representation (you store compressed, you
  search compressed), top-5 turns, dedup to sessions, deterministic Recall@5.

Methods: full float32, Matryoshka truncation (text-embedding-3-large is MRL, so
first-N + renormalize == the API `dimensions` output, verified here), PCA
truncation, int8 scalar quantization, binary (sign) quantization,
Matryoshka+binary, and the quantum encoding (PCA-16 amplitude state), which is
rank-identical to the real SWAP test (cos^2 is monotone in cos) and plotted as
the dominated point.

Writes results/longmemeval-s_compression_frontier.json and
docs/compression_frontier.png.

Usage:
    python scripts/compression_frontier.py [corpus]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _chart_style import CORAL, GREY_HEX, NAVY, PURPLE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_JSON = RESULTS_DIR / "longmemeval-s_compression_frontier.json"
OUT_PNG = REPO_ROOT / "docs" / "compression_frontier.png"

TOP_K = 5
RUN_ID = "qexp"
MATRYOSHKA_DIMS = [1024, 512, 256, 128, 64]
PCA_DIMS = [256, 128, 64, 32, 16]
BINARY_TRUNC = [1024, 512, 256]  # Matryoshka-truncate then binarize


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _recall_at_k(ranked_sessions: list[str], gold: set[str], k: int = TOP_K) -> float:
    if not gold:
        return 0.0
    head = ranked_sessions[:k]
    return sum(1 for s in head if s in gold) / len(gold)


def _sessions_from_order(order, sids: list[str]) -> list[str]:
    seen: list[str] = []
    for i in order:
        s = sids[i]
        if s and s not in seen:
            seen.append(s)
    return seen


async def _load(corpus: str):
    from memory_arena.sessions.loaders import load_sessions
    from memory_arena.strategies.naive_vector import NaiveVectorStrategy

    sessions = load_sessions(corpus)
    total = sum(len(s.turns) for s in sessions)
    strat = NaiveVectorStrategy()
    await strat.setup(RUN_ID)
    col = strat._get_collection()
    if col.count() < total:
        for s in sessions:
            await strat.ingest_session(s)
    return col


def main(corpus: str) -> Path:
    import asyncio

    from memory_arena.benchmark.questions import load_memory_questions
    from memory_arena.settings import settings
    from memory_arena.strategies.embeddings import OpenAIEmbedding

    col = asyncio.run(_load(corpus))
    everything = col.get(include=["embeddings", "metadatas"])
    docs_full = np.asarray(everything["embeddings"], dtype=float)
    sids = [m.get("session_id", "") for m in everything["metadatas"]]
    print(f"{docs_full.shape[0]} turns x {docs_full.shape[1]} dims")

    questions = load_memory_questions(corpus, subset="full")
    ef = OpenAIEmbedding()
    queries_full = np.asarray(ef([q.question for q in questions]), dtype=float)
    golds = [set(q.ground_truth.supporting_session_ids) for q in questions]
    cats = [q.category for q in questions]

    # Verify Matryoshka: first-N + renormalize ~= the API `dimensions` output.
    import openai

    client = openai.OpenAI(api_key=settings.openai_api_key)
    probe = questions[0].question
    api256 = np.asarray(
        client.embeddings.create(model=settings.embedding_model, input=[probe], dimensions=256)
        .data[0]
        .embedding,
        dtype=float,
    )
    trunc256 = _unit(np.asarray(ef([probe])[0], dtype=float)[:256])
    mrl_cos = float(_unit(api256) @ trunc256)
    print(f"Matryoshka check: cos(API dims=256, truncate+renorm) = {mrl_cos:.4f}")

    # PCA basis fit on the corpus turns.
    from sklearn.decomposition import PCA

    pca = {n: PCA(n_components=n).fit(docs_full) for n in PCA_DIMS}

    def repr_for(name: str):
        """Return (doc_matrix, query_matrix, sim) for a method."""
        if name == "full":
            return _unit_rows(docs_full), _unit_rows(queries_full), "dot"
        if name.startswith("matryoshka_"):
            n = int(name.split("_")[1])
            return _unit_rows(docs_full[:, :n]), _unit_rows(queries_full[:, :n]), "dot"
        if name.startswith("pca_"):
            n = int(name.split("_")[1])
            return (
                _unit_rows(pca[n].transform(docs_full)),
                _unit_rows(pca[n].transform(queries_full)),
                "dot",
            )
        if name == "int8":
            d = np.clip(np.round(_unit_rows(docs_full) * 127), -127, 127)
            q = np.clip(np.round(_unit_rows(queries_full) * 127), -127, 127)
            return d, q, "dot"
        if name == "binary":
            return (
                np.sign(docs_full) + (docs_full == 0),
                np.sign(queries_full) + (queries_full == 0),
                "dot",
            )
        if name.startswith("binary_"):
            n = int(name.split("_")[1])
            d = docs_full[:, :n]
            q = queries_full[:, :n]
            return np.sign(d) + (d == 0), np.sign(q) + (q == 0), "dot"
        raise ValueError(name)

    def bytes_for(name: str) -> int:
        if name == "full":
            return 4 * docs_full.shape[1]
        if name == "int8":
            return docs_full.shape[1]
        if name == "binary":
            return docs_full.shape[1] // 8
        n = int(name.split("_")[1])
        return n // 8 if name.startswith("binary_") else 4 * n

    family = {
        "full": "full",
        "int8": "quantized",
        "binary": "quantized",
        **{f"matryoshka_{n}": "matryoshka" for n in MATRYOSHKA_DIMS},
        **{f"pca_{n}": "pca" for n in PCA_DIMS},
        **{f"binary_{n}": "matryoshka+binary" for n in BINARY_TRUNC},
    }
    methods = (
        ["full"]
        + [f"matryoshka_{n}" for n in MATRYOSHKA_DIMS]
        + [f"pca_{n}" for n in PCA_DIMS]
        + ["int8", "binary"]
        + [f"binary_{n}" for n in BINARY_TRUNC]
    )

    def evaluate(dmat, qmat) -> tuple[float, float, dict]:
        sims = qmat @ dmat.T  # (n_queries, n_docs)
        recalls, hits = [], []
        by_cat: dict[str, list[float]] = {}
        for i in range(sims.shape[0]):
            order = np.argsort(-sims[i], kind="stable")
            sess = _sessions_from_order(order, sids)
            r = _recall_at_k(sess, golds[i])
            recalls.append(r)
            hits.append(1.0 if any(s in golds[i] for s in sess[:TOP_K]) else 0.0)
            by_cat.setdefault(cats[i], []).append(r)
        cat_means = {c: round(sum(v) / len(v), 4) for c, v in by_cat.items()}
        return (
            round(sum(recalls) / len(recalls), 4),
            round(sum(hits) / len(hits), 4),
            cat_means,
        )

    results = {}
    for name in methods:
        d, q, _ = repr_for(name)
        r, h, by_cat = evaluate(d, q)
        results[name] = {
            "family": family[name],
            "bytes_per_vector": bytes_for(name),
            "recall@5": r,
            "hit@5": h,
            "by_category": by_cat,
        }

    # Quantum encoding == PCA-16 ranking (cos^2 monotone in cos); verify the real
    # SWAP test reproduces the PCA-16 cosine order on one query, then reuse it.
    quantum_recall = results["pca_16"]["recall@5"]
    swap_verified = None
    try:
        from memory_arena.strategies.quantum.circuits import swap_test_fidelities
        from memory_arena.strategies.quantum.utils import amplitude_encode

        dred = pca[16].transform(docs_full)
        qred = pca[16].transform(queries_full[:1])
        q_amp = amplitude_encode(qred[0], 4)
        d_amps = [amplitude_encode(dr, 4) for dr in dred]
        fid = np.asarray(swap_test_fidelities(q_amp, d_amps, 4, shots=0))
        cos_order = np.argsort(-(_unit_rows(dred) @ _unit(qred[0])), kind="stable")
        swap_order = np.argsort(-fid, kind="stable")
        swap_verified = bool((cos_order[:TOP_K] == swap_order[:TOP_K]).all())
    except Exception as exc:  # qiskit not installed
        print(f"(SWAP-test verification skipped: {exc})")
    results["quantum_pca16_swap"] = {
        "family": "quantum",
        "bytes_per_vector": 4 * 16,
        "recall@5": quantum_recall,
        "hit@5": results["pca_16"]["hit@5"],
        "swap_equals_pca16_topk": swap_verified,
        "note": "SWAP-test fidelity == cos^2 on PCA-16, so ranking == pca_16",
    }

    OUT_JSON.write_text(
        json.dumps({"corpus": corpus, "n_questions": len(questions), "methods": results}, indent=2)
    )

    print("\n=== Compression frontier: Recall@5 vs bytes/vector ===")
    rows = sorted(results.items(), key=lambda kv: -kv[1]["bytes_per_vector"])
    for name, r in rows:
        print(
            f"  {name:22} {r['family']:18} {r['bytes_per_vector']:>6} B   "
            f"recall@5={r['recall@5']:.3f}  hit@5={r['hit@5']:.3f}"
        )
    print(f"\nSWAP-test reproduces PCA-16 top-5 ranking: {swap_verified}")

    _chart(results)
    print(f"Wrote {OUT_JSON}\nWrote {OUT_PNG}")
    return OUT_PNG


def _chart(results: dict) -> None:
    colors = {
        "full": "#1c1c1c",
        "matryoshka": NAVY,
        "pca": GREY_HEX,
        "quantized": CORAL,
        "matryoshka+binary": CORAL,
        "quantum": PURPLE,
    }
    markers = {
        "full": "D",
        "matryoshka": "o",
        "pca": "o",
        "quantized": "^",
        "matryoshka+binary": "s",
        "quantum": "*",
    }
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=150)

    for fam in ["matryoshka", "pca"]:
        pts = sorted(
            [(r["bytes_per_vector"], r["recall@5"]) for r in results.values() if r["family"] == fam]
        )
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(
                xs,
                ys,
                color=colors[fam],
                linewidth=1.3,
                alpha=0.5,
                zorder=1,
                linestyle="--" if fam == "pca" else "-",
            )

    seen_fam = set()
    for name, r in results.items():
        fam = r["family"]
        ax.scatter(
            r["bytes_per_vector"],
            r["recall@5"],
            marker=markers[fam],
            s=300 if fam == "quantum" else (150 if fam == "full" else 90),
            color=colors[fam],
            edgecolor="white",
            linewidth=1.0,
            zorder=4 if fam == "quantum" else 3,
            label=fam if fam not in seen_fam else None,
        )
        seen_fam.add(fam)

    # Label the load-bearing points.
    for name in ["full", "matryoshka_256", "binary", "pca_16", "quantum_pca16_swap", "binary_512"]:
        if name in results:
            r = results[name]
            tag = {
                "quantum_pca16_swap": "quantum (PCA-16)",
                "binary": "binary-3072",
                "binary_512": "binary-512",
                "pca_16": "PCA-16",
            }.get(name, name)
            ax.annotate(
                tag,
                xy=(r["bytes_per_vector"], r["recall@5"]),
                xytext=(6, -12 if "binary_512" in name or "pca_16" in name else 6),
                textcoords="offset points",
                fontsize=8,
                color="#444",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Bytes per vector (log scale): storage + memory-bandwidth cost")
    ax.set_ylabel("Session Recall@5 (deterministic, exact)")
    ax.set_title(
        "Embedding compression frontier: where bytes can be cut, and where quantum sits",
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax.set_ylim(0.55, 1.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.tick_params(colors="#666", which="both")
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.12)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "longmemeval-s")
