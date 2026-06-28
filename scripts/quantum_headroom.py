"""Why quantum reranking can't win here, and a real attempt to break the pattern.

Two structural reasons quantum fidelity reranking ties or loses to plain cosine:

  1. Tautology: quantum fidelity F = |<q|d>|^2 = cos^2(theta), a monotone of
     cosine, so it reproduces cosine's exact ranking.
  2. No headroom: this measures whether cosine already retrieves every gold
     session that is even present in the candidate pool. If cosine's Recall@5
     == the pool ceiling, no reranker (quantum or classical) can improve it.

Then it tests the one quantum-inspired method that is NOT a monotone of a single
cosine: the density-matrix session representation rho_s = mean_t |t><t|, scored
by <q|rho_s|q> = mean_t (q.t)^2. Compared against session-level max (best turn,
what naive_vector effectively does), centroid cosine, and a soft top-m mean,
overall and per category, against gold supporting_session_ids.

Usage:
    python scripts/quantum_headroom.py [corpus]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
POOL = 30
TOP_K = 5
RUN_ID = "qexp"


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _recall_at_k(ranked_sessions: list[str], gold: set[str], k: int = TOP_K) -> float:
    if not gold:
        return 0.0
    head = ranked_sessions[:k]
    return sum(1 for s in head if s in gold) / len(gold)


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
    from memory_arena.strategies.embeddings import OpenAIEmbedding

    col = asyncio.run(_load(corpus))
    questions = load_memory_questions(corpus, subset="full")

    # All turn embeddings grouped by session (for session-level density matrices).
    everything = col.get(include=["embeddings", "metadatas"])
    by_session: dict[str, list[np.ndarray]] = defaultdict(list)
    for emb, meta in zip(everything["embeddings"], everything["metadatas"], strict=True):
        by_session[meta.get("session_id", "")].append(_unit(emb))
    session_mats = {s: np.vstack(v) for s, v in by_session.items() if s}
    all_sids = list(session_mats)

    ef = OpenAIEmbedding()

    def session_scores(qv: np.ndarray, mode: str) -> list[str]:
        scored = []
        for sid, mat in session_mats.items():
            sims = mat @ qv  # cosine of each turn to query (unit rows)
            sq = sims**2
            if mode == "max_sq":  # best turn (what naive_vector effectively does)
                score = float(sq.max())
            elif mode == "mean_sq":  # density matrix <q|rho|q>
                score = float(sq.mean())
            elif mode == "top3_sq":  # soft density matrix over the 3 best turns
                score = float(np.sort(sq)[-3:].mean())
            elif mode == "centroid":  # classical mean-vector cosine
                score = float(_unit(mat.mean(axis=0)) @ qv)
            else:
                score = 0.0
            scored.append((sid, score))
        scored.sort(key=lambda t: -t[1])
        return [s for s, _ in scored]

    modes = ["max_sq", "top3_sq", "mean_sq", "centroid"]
    rows: dict[str, list[float]] = defaultdict(list)
    rows_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cosine_recall: list[float] = []
    ceiling: list[float] = []
    cosine_recall_cat: dict[str, list[float]] = defaultdict(list)
    ceiling_cat: dict[str, list[float]] = defaultdict(list)

    for q in questions:
        gold = set(q.ground_truth.supporting_session_ids)
        qv = _unit(np.asarray(ef([q.question])[0], dtype=float))

        # cosine turn-level top-5 (naive_vector path) + pool ceiling.
        res = col.query(
            query_embeddings=[list(map(float, qv))], n_results=POOL, include=["metadatas"]
        )
        metas = res["metadatas"][0]
        pool_sessions: list[str] = []
        for m in metas:
            sid = m.get("session_id", "")
            if sid and sid not in pool_sessions:
                pool_sessions.append(sid)
        cosine_recall.append(_recall_at_k(pool_sessions, gold))
        cosine_recall_cat[q.category].append(_recall_at_k(pool_sessions, gold))
        # ceiling: gold reachable anywhere in the pool of 30 turns.
        ceiling.append(_recall_at_k(pool_sessions, gold, k=len(pool_sessions)))
        ceiling_cat[q.category].append(_recall_at_k(pool_sessions, gold, k=len(pool_sessions)))

        for mode in modes:
            r = _recall_at_k(session_scores(qv, mode), gold)
            rows[mode].append(r)
            rows_cat[q.category][mode].append(r)

    def m(v):
        return round(sum(v) / len(v), 4) if v else 0.0

    payload = {
        "corpus": corpus,
        "n_questions": len(questions),
        "n_sessions": len(all_sids),
        "overall": {
            "cosine_turn_topk_recall@5": m(cosine_recall),
            "pool_ceiling_recall": m(ceiling),
            **{f"session_{mode}_recall@5": m(rows[mode]) for mode in modes},
        },
        "by_category": {
            cat: {
                "cosine_recall@5": m(cosine_recall_cat[cat]),
                "pool_ceiling": m(ceiling_cat[cat]),
                **{f"session_{mode}@5": m(rows_cat[cat][mode]) for mode in modes},
                "n": len(cosine_recall_cat[cat]),
            }
            for cat in cosine_recall_cat
        },
    }
    out = RESULTS_DIR / f"{corpus}_quantum_headroom.json"
    out.write_text(json.dumps(payload, indent=2))

    print("=== Headroom: does cosine already hit the pool ceiling? ===")
    print(
        f"  overall: cosine Recall@5 = {m(cosine_recall):.3f}  "
        f"pool ceiling = {m(ceiling):.3f}  "
        f"headroom = {m(ceiling) - m(cosine_recall):+.3f}"
    )
    for cat in payload["by_category"]:
        c = payload["by_category"][cat]
        print(
            f"  {cat:26} cosine={c['cosine_recall@5']:.3f}  ceiling={c['pool_ceiling']:.3f}  "
            f"headroom={c['pool_ceiling'] - c['cosine_recall@5']:+.3f}  (n={c['n']})"
        )

    print("\n=== Density-matrix (quantum-inspired) vs pooling, session-level Recall@5 ===")
    print(f"  {'cosine turn-level (naive)':28} {m(cosine_recall):.3f}")
    for mode in modes:
        tag = {
            "max_sq": "best-turn (max |<q|t>|^2)",
            "top3_sq": "soft density (top-3 mean)",
            "mean_sq": "density matrix <q|rho|q>",
            "centroid": "centroid cosine",
        }[mode]
        print(f"  session {tag:28} {m(rows[mode]):.3f}")

    print("\n=== multi_session_reasoning (where headroom is) ===")
    msr = payload["by_category"].get("multi_session_reasoning", {})
    if msr:
        print(f"  cosine={msr['cosine_recall@5']:.3f}  ceiling={msr['pool_ceiling']:.3f}")
        for mode in modes:
            print(f"  session_{mode}@5 = {msr[f'session_{mode}@5']:.3f}")

    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "longmemeval-s"
    main(corpus)
