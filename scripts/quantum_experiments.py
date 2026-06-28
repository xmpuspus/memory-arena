"""Definitive retrieval-level experiments for the quantum rerankers.

Reranking in statevector mode is deterministic (no LLM, no sampling) and
Recall@k against gold supporting ids needs no generation, so these numbers are
exact and cheap to rerun. That lets us answer two mechanistic questions
without judge/seed noise:

Q1  Does quantum *interference* help retrieval? Compare, on the SAME decomposed
    subqueries, coherent superposition |Sum a_i|q_i>|^2 (has cross-terms) vs the
    incoherent mixture Sum a_i^2 |<q_i|d>|^2 (cross-terms removed) vs classical
    RRF. coherent - incoherent IS the interference effect, everything else fixed.
    Tested rerank-only (shipped qiss architecture) and retrieve-and-fuse (union
    pool, interference's best case). Broken out by category.

Q2  Is sqr's accuracy loss purely the PCA encoding? Sweep n_qubits in {4,5,6,7}
    (16/32/64/128 dims) and watch whether sqr's Recall@5 climbs toward
    naive_vector's as the kept variance rises.

Writes results/<corpus>_quantum_experiments.json.

Usage:
    python scripts/quantum_experiments.py [corpus]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "tmp"

# Pool size is overridable so we can match the benchmark's fanout (top_k*4=20)
# and reconcile the deterministic recall with the end-to-end run.
POOL = int(os.environ.get("QEXP_POOL", "30"))  # full-query candidate pool
POOL_SUB = 15  # per-subquery pool for retrieve-and-fuse union
TOP_K = 5  # evaluation cutoff, matches the published Recall@5
RRF_C = 60
PCA_SAMPLE = 512
N_QUBITS_GRID = [4, 5, 6, 7]
RUN_ID = "qexp"
POOL_KEY = f"_pool_ceiling@{POOL}"

_DECOMPOSE_SYSTEM = (
    "You rewrite a user's question into standalone SEARCH QUERIES that retrieve the user's "
    "own past chat messages. Output 2 to 4 short keyword-style search queries, one per line. "
    "Each query is a noun phrase or topic, NOT a sentence addressed to anyone. "
    "Do NOT answer the question. Do NOT refuse. Do NOT give advice or instructions. "
    "Do NOT say 'I', 'you', or 'your'. Do NOT mention email, accounts, calendars, or websites. "
    "If the question is atomic, output a single search query. "
    "Example: 'How many projects have I led?' -> 'projects led' / 'project leadership role' / "
    "'leading a project'."
)


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n else v


async def _ingest(corpus: str):
    from memory_arena.sessions.loaders import load_sessions
    from memory_arena.strategies.naive_vector import NaiveVectorStrategy

    sessions = load_sessions(corpus)
    total_turns = sum(len(s.turns) for s in sessions)
    strat = NaiveVectorStrategy()
    await strat.setup(RUN_ID)
    col = strat._get_collection()
    if col.count() >= total_turns:
        print(f"reuse collection ({col.count()} turns already embedded)")
        return strat, col
    print(f"ingesting {len(sessions)} sessions ({total_turns} turns) ...")
    for s in sessions:
        await strat.ingest_session(s)
    print(f"ingested; collection has {col.count()} turns")
    return strat, col


# Reject refusal/advice/meta lines so the superposition isn't diluted by junk
# (Haiku's cheap path refused first-person questions; Sonnet + a strict prompt
# plus this filter keep only real search-query fragments).
_REFUSAL_MARKERS = (
    "i don't",
    "i do not",
    "i can't",
    "i cannot",
    "i'm happy",
    "i am happy",
    "could you",
    "if you'd",
    "if you would",
    "please ",
    "check your",
    "log into",
    "call the",
    "review your",
    "look through",
    "to answer",
    "here are",
    "sorry",
)


def _clean_subqueries(text: str, question: str) -> list[str]:
    subs: list[str] = []
    for ln in text.splitlines():
        s = ln.strip("-*0123456789. \t").strip()
        if not s:
            continue
        low = s.lower()
        if any(m in low for m in _REFUSAL_MARKERS):
            continue
        if len(s) > 90:  # full sentences/advice, not a search query
            continue
        subs.append(s)
    # Always include the original question so the full-query signal is present.
    if question not in subs:
        subs.append(question)
    return subs or [question]


async def _decompose(corpus: str, questions) -> dict[str, list[str]]:
    # v2 cache: strict search-query rewriter on the generate model.
    cache = CACHE_DIR / f"{corpus}_decompositions_v2.json"
    if cache.exists():
        print(f"reuse decomposition cache {cache.name}")
        return json.loads(cache.read_text())
    from memory_arena.llm.client import LLMClient

    llm = LLMClient()
    out: dict[str, list[str]] = {}
    for q in questions:
        resp = await llm.generate(q.question, "", _DECOMPOSE_SYSTEM, max_tokens=200)
        out[q.id] = _clean_subqueries(resp.text, q.question)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    print(f"decomposed {len(out)} questions -> {cache.name}")
    return out


def _query_pool(col, qvec: np.ndarray, n: int):
    res = col.query(
        query_embeddings=[list(map(float, qvec))],
        n_results=n,
        include=["embeddings", "metadatas", "distances"],
    )
    embs = res["embeddings"][0] if res.get("embeddings") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    return embs, metas


def _candidate(meta, emb):
    return {
        "turn_id": meta.get("turn_id", ""),
        "session_id": meta.get("session_id", ""),
        "emb": _unit(emb),
    }


def _union_pool(col, qvec, subvecs):
    """Dedup union of full-query top-POOL_SUB and each subquery top-POOL_SUB."""
    seen: dict[str, dict] = {}
    for v in [qvec, *subvecs]:
        embs, metas = _query_pool(col, v, POOL_SUB)
        for emb, meta in zip(embs, metas, strict=False):
            tid = meta.get("turn_id", "")
            if tid and tid not in seen:
                seen[tid] = _candidate(meta, emb)
    return list(seen.values())


def _rank(cands: list[dict], scores: np.ndarray) -> list[dict]:
    order = np.argsort(-scores, kind="stable")
    return [cands[i] for i in order]


def _eval(ranked: list[dict], gold_sessions: set[str], gold_turns: set[str]) -> dict:
    from memory_arena.benchmark.recall_metrics import compute_memory_recall_metrics

    sess: list[str] = []
    for c in ranked:
        if c["session_id"] and c["session_id"] not in sess:
            sess.append(c["session_id"])
    turns = [c["turn_id"] for c in ranked]
    return compute_memory_recall_metrics(sess, gold_sessions, turns, gold_turns, k=TOP_K)


# --- scoring functions over a candidate pool D (unit rows) given query/subqueries ---


def score_cosine(dmat, q, subs):
    return dmat @ q


def score_cos2(dmat, q, subs):
    return (dmat @ q) ** 2


def score_coherent(dmat, q, subs):
    qstate = _unit(np.sum(_unit_rows(np.vstack(subs)), axis=0))
    return (dmat @ qstate) ** 2


def score_incoherent(dmat, q, subs):
    sub_unit = _unit_rows(np.vstack(subs))
    w = 1.0 / sub_unit.shape[0]
    return np.sum(w * (dmat @ sub_unit.T) ** 2, axis=1)


def score_rrf(dmat, q, subs):
    sub_unit = _unit_rows(np.vstack(subs))
    n = dmat.shape[0]
    rrf = np.zeros(n)
    for i in range(sub_unit.shape[0]):
        sims = dmat @ sub_unit[i]
        order = np.argsort(-sims, kind="stable")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        rrf += 1.0 / (RRF_C + ranks)
    return rrf


def main(corpus: str) -> Path:
    import asyncio

    from memory_arena.benchmark.questions import load_memory_questions
    from memory_arena.strategies.quantum.circuits import swap_test_fidelities
    from memory_arena.strategies.quantum.utils import PCAReducer, amplitude_encode, state_dim

    strat, col = asyncio.run(_ingest(corpus))
    questions = load_memory_questions(corpus, subset="full")
    print(f"{len(questions)} questions (full set)")
    decomp = asyncio.run(_decompose(corpus, questions))

    from memory_arena.strategies.embeddings import OpenAIEmbedding

    ef = OpenAIEmbedding()

    # Fit one PCA reducer per n_qubits on a corpus sample (matches sqr production).
    sample = col.get(include=["embeddings"], limit=PCA_SAMPLE)
    sample_embs = np.asarray(sample["embeddings"], dtype=float)
    reducers = {}
    for nq in N_QUBITS_GRID:
        r = PCAReducer(state_dim(nq)).fit(sample_embs)
        reducers[nq] = r
        print(f"PCA n_qubits={nq} ({state_dim(nq)} dims): variance={r.variance_explained:.3f}")

    rerank_methods = {
        "cosine": score_cosine,
        "qiss_single_cos2": score_cos2,
        "coherent_rerank": score_coherent,
        "incoherent_rerank": score_incoherent,
    }
    fuse_methods = {
        "coherent_fuse": score_coherent,
        "incoherent_fuse": score_incoherent,
        "rrf_fuse": score_rrf,
    }

    rows: dict[str, list[dict]] = defaultdict(list)
    rows_by_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pool_ceiling: list[dict] = []
    multi_ceiling: list[dict] = []
    n_decomposed = 0

    for q in questions:
        gold_s = set(q.ground_truth.supporting_session_ids)
        gold_t = set(q.ground_truth.supporting_turn_ids)
        subs_txt = decomp.get(q.id, [q.question])
        is_multi = len(subs_txt) > 1
        if is_multi:
            n_decomposed += 1

        qvec = _unit(np.asarray(ef([q.question])[0], dtype=float))
        subvecs = [_unit(np.asarray(v, dtype=float)) for v in ef(subs_txt)] if is_multi else [qvec]

        # Full-query pool (rerank-only + sqr).
        f_embs, f_metas = _query_pool(col, qvec, POOL)
        full = [_candidate(m, e) for m, e in zip(f_metas, f_embs, strict=False)]
        d_full = _unit_rows(np.vstack([c["emb"] for c in full]))
        # ceiling: gold-in-pool@POOL
        pool_ceiling.append(_eval(full, gold_s, gold_t))

        for name, fn in rerank_methods.items():
            ranked = _rank(full, fn(d_full, qvec, subvecs))
            m = _eval(ranked, gold_s, gold_t)
            rows[name].append(m)
            rows_by_cat[q.category][name].append(m)

        # sqr sweep on the full pool (deterministic statevector).
        for nq in N_QUBITS_GRID:
            red = reducers[nq]
            q_red = red.transform([qvec])[0]
            d_red = red.transform(np.vstack([c["emb"] for c in full]))
            q_amp = amplitude_encode(q_red, nq)
            d_amps = [amplitude_encode(d, nq) for d in d_red]
            fids = np.asarray(swap_test_fidelities(q_amp, d_amps, nq, shots=0))
            ranked = _rank(full, fids)
            m = _eval(ranked, gold_s, gold_t)
            rows[f"sqr_q{nq}"].append(m)
            rows_by_cat[q.category][f"sqr_q{nq}"].append(m)

        # Retrieve-and-fuse on the union pool (interference's best case).
        multi = _union_pool(col, qvec, subvecs) if is_multi else full
        d_multi = _unit_rows(np.vstack([c["emb"] for c in multi]))
        multi_ceiling.append(_eval(multi, gold_s, gold_t))
        for name, fn in fuse_methods.items():
            ranked = _rank(multi, fn(d_multi, qvec, subvecs))
            m = _eval(ranked, gold_s, gold_t)
            rows[name].append(m)
            rows_by_cat[q.category][name].append(m)

    def agg(ms: list[dict]) -> dict:
        n = len(ms)
        return {
            "n": n,
            "recall@5": round(sum(r["session_recall_at_k"] for r in ms) / n, 4),
            "hit@5": round(sum(r["session_hit_at_k"] for r in ms) / n, 4),
            "turn_recall@5": round(sum(r["turn_recall_at_k"] for r in ms) / n, 4),
            "mrr": round(sum(r["session_mrr"] for r in ms) / n, 4),
        }

    summary = {m: agg(v) for m, v in rows.items()}
    summary[POOL_KEY] = agg(pool_ceiling)
    summary["_union_ceiling"] = agg(multi_ceiling)

    by_cat = {cat: {m: agg(v) for m, v in methods.items()} for cat, methods in rows_by_cat.items()}

    payload = {
        "corpus": corpus,
        "n_questions": len(questions),
        "n_decomposed": n_decomposed,
        "pool": POOL,
        "top_k": TOP_K,
        "pca_variance": {nq: round(reducers[nq].variance_explained, 4) for nq in N_QUBITS_GRID},
        "overall": summary,
        "by_category": by_cat,
    }
    out = RESULTS_DIR / f"{corpus}_quantum_experiments.json"
    out.write_text(json.dumps(payload, indent=2))

    asyncio.run(strat.teardown()) if False else None  # keep collection for reruns

    # ---- print tables ----
    print(f"\n=== Q1: fusion methods (overall, {len(questions)}q; {n_decomposed} decomposed) ===")
    order1 = [
        "cosine",
        "qiss_single_cos2",
        "coherent_rerank",
        "incoherent_rerank",
        "coherent_fuse",
        "incoherent_fuse",
        "rrf_fuse",
        POOL_KEY,
        "_union_ceiling",
    ]
    for m in order1:
        s = summary[m]
        print(
            f"  {m:24} recall@5={s['recall@5']:.3f} hit@5={s['hit@5']:.3f} "
            f"turn_recall@5={s['turn_recall@5']:.3f} mrr={s['mrr']:.3f}"
        )

    print("\n=== Q1: multi_session_reasoning only (the decomposable category) ===")
    msr = by_cat.get("multi_session_reasoning", {})
    for m in [
        "cosine",
        "coherent_rerank",
        "incoherent_rerank",
        "coherent_fuse",
        "incoherent_fuse",
        "rrf_fuse",
    ]:
        if m in msr:
            s = msr[m]
            print(
                f"  {m:24} recall@5={s['recall@5']:.3f} hit@5={s['hit@5']:.3f} "
                f"turn_recall@5={s['turn_recall@5']:.3f} (n={s['n']})"
            )

    print("\n=== Q2: sqr Recall@5 vs n_qubits (does it climb toward cosine?) ===")
    print(
        f"  {'cosine (baseline)':20} recall@5={summary['cosine']['recall@5']:.3f} "
        f"turn_recall@5={summary['cosine']['turn_recall@5']:.3f}"
    )
    for nq in N_QUBITS_GRID:
        s = summary[f"sqr_q{nq}"]
        print(
            f"  sqr_q{nq} ({state_dim(nq):>3}d, var={reducers[nq].variance_explained:.0%}) "
            f"recall@5={s['recall@5']:.3f} turn_recall@5={s['turn_recall@5']:.3f}"
        )

    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "longmemeval-s"
    main(corpus)
