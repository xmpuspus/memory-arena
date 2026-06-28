"""Aggregate per-seed result JSONs into per-strategy bootstrap summaries.

Reads results/<corpus>_<strategy>_seed{N}.json for N in {0, 1, 2, ...} and
writes results/<corpus>_<strategy>_summary.json with mean accuracy, 95% CI,
plus the secondary metrics consumed by the README, dashboard, and charts.

The CI is a real non-parametric bootstrap over per-question scores. Each
question's per-seed scores are averaged into a single per-question score,
then the resulting vector is resampled with replacement n_boot times to
produce the (lo, hi) quantiles. With 16 questions this gives a calibrated
view of question-set sensitivity that 1.96 * SEM over 3 seed-means cannot.

Usage:
    python scripts/aggregate_bootstrap.py [corpus]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

N_BOOTSTRAP = 5000
RNG_SEED = 0
BOOTSTRAP_METHOD = "non_parametric_question_level"


def bootstrap_ci(
    per_question_scores: list[float],
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI over per-question scores.

    Resamples question-level scores with replacement n_boot times, returns
    (mean, ci_low, ci_high) at the (1 - alpha) confidence level. An empty
    input returns (NaN, NaN, NaN); a single-element input returns
    (value, value, value) — bootstrap can't widen what isn't there.
    """
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
    arr = np.asarray(per_question_scores, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_val = float(arr.mean())
    if arr.size == 1:
        return mean_val, mean_val, mean_val
    n = arr.size
    # Vectorized resample: (n_boot, n) draws, mean along axis=1.
    samples = rng.choice(arr, size=(n_boot, n), replace=True)
    boot_means = samples.mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return mean_val, lo, hi


def _per_question_scores(runs: list[dict], field: str = "accuracy") -> list[float]:
    """Pool per-question scores across seeds.

    For each question_id seen in any seed, average its score across all seeds
    that touched it, then return the resulting vector. This makes the
    bootstrap unit "the question" rather than "the seed", which is the right
    resampling unit for memory-arena (16 questions × 3 seeds).
    """
    by_qid: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for rec in run.get("recall_records", []) or []:
            qid = rec.get("question_id")
            if not qid:
                continue
            score = rec.get("score") or {}
            if isinstance(score, dict) and field in score:
                by_qid[qid].append(float(score[field]))
    return [float(np.mean(scores)) for scores in by_qid.values() if scores]


def aggregate(corpus: str = "longmemeval-s") -> list[Path]:
    per_strategy: dict[str, list[dict]] = defaultdict(list)
    pattern = f"{corpus}_*_seed*.json"
    for p in sorted(RESULTS_DIR.glob(pattern)):
        stem = p.stem  # longmemeval-s_naive_vector_seed0
        prefix = f"{corpus}_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix) :]
        if "_seed" not in rest:
            continue
        strategy = rest.rsplit("_seed", 1)[0]
        per_strategy[strategy].append(json.loads(p.read_text()))

    written: list[Path] = []
    for strategy, runs in per_strategy.items():
        summary = _summarize(strategy, runs)
        out = RESULTS_DIR / f"{corpus}_{strategy}_summary.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        written.append(out)
        ci_lo = summary["ci_low_95"]
        ci_hi = summary["ci_high_95"]
        print(
            f"{strategy}: n_seeds={summary['n_seeds']:>1} n_q={summary['n_questions']:>2}  "
            f"acc={summary['mean_accuracy']:.3f}  "
            f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]  "
            f"cost=${summary['total_cost_usd']:.3f}"
        )
    return written


def _summarize(strategy: str, runs: list[dict]) -> dict:
    rng = np.random.default_rng(RNG_SEED)

    pq_acc = _per_question_scores(runs, field="accuracy")
    mean_acc, ci_lo, ci_hi = bootstrap_ci(pq_acc, rng=rng)
    # Symmetric half-width retained for backward-compat consumers (legacy charts).
    acc_ci_half = ((ci_hi - ci_lo) / 2.0) if not np.isnan(ci_hi) else 0.0

    # Cost / latency stay seed-level — they're per-run aggregates, not per-question.
    costs = [r.get("total_cost_usd", 0.0) or 0.0 for r in runs]
    latencies = [r.get("avg_recall_latency_ms", 0.0) or 0.0 for r in runs]
    recalls = [
        r.get("mean_session_recall_at_k")
        for r in runs
        if r.get("mean_session_recall_at_k") is not None
    ]
    hits = [
        r.get("mean_session_hit_at_k") for r in runs if r.get("mean_session_hit_at_k") is not None
    ]
    cost_mean = float(np.mean(costs)) if costs else 0.0
    cost_ci = _seed_level_ci_half(costs)
    lat_mean = float(np.mean(latencies)) if latencies else 0.0
    lat_ci = _seed_level_ci_half(latencies)
    rec_mean = float(np.mean(recalls)) if recalls else None
    rec_ci = _seed_level_ci_half(recalls) if recalls else None
    hit_mean = float(np.mean(hits)) if hits else None
    hit_ci = _seed_level_ci_half(hits) if hits else None

    statuses = [r.get("status", "ok") for r in runs]
    # If even one seed flagged config-failed-at-default, surface it. Vendors that
    # need an MEMORI_API_KEY etc. should not get a green light from one lucky run.
    status = "config-failed-at-default" if "config-failed-at-default" in statuses else "ok"
    ingest_failure_rates = [r.get("ingest_failure_rate", 0.0) or 0.0 for r in runs]
    questions_evaluated = [r.get("questions_evaluated", 0) for r in runs]
    accs_per_seed = [r.get("accuracy", 0.0) or 0.0 for r in runs]

    by_cat = _aggregate_by_category(runs)

    metadata = runs[-1].get("metadata", {}) if runs else {}

    return {
        "strategy": strategy,
        "corpus": runs[-1].get("corpus", "longmemeval-s"),
        "status": status,
        "n_seeds": len(runs),
        "n_questions": len(pq_acc),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_method": BOOTSTRAP_METHOD,
        # Primary non-parametric bootstrap fields.
        "mean_accuracy": mean_acc if not np.isnan(mean_acc) else 0.0,
        "ci_low_95": ci_lo if not np.isnan(ci_lo) else 0.0,
        "ci_high_95": ci_hi if not np.isnan(ci_hi) else 0.0,
        # Backward-compat: chart scripts still read these.
        "accuracy": mean_acc if not np.isnan(mean_acc) else 0.0,
        "accuracy_ci": acc_ci_half,
        "ci_low": ci_lo if not np.isnan(ci_lo) else 0.0,
        "ci_high": ci_hi if not np.isnan(ci_hi) else 0.0,
        "accuracy_per_seed": accs_per_seed,
        "total_cost_usd": cost_mean,
        "total_cost_ci": cost_ci,
        "avg_recall_latency_ms": lat_mean,
        "avg_recall_latency_ci_ms": lat_ci,
        "mean_session_recall_at_k": rec_mean,
        "mean_session_recall_at_k_ci": rec_ci,
        "mean_session_hit_at_k": hit_mean,
        "mean_session_hit_at_k_ci": hit_ci,
        "ingest_failure_rate": float(np.mean(ingest_failure_rates))
        if ingest_failure_rates
        else 0.0,
        "questions_evaluated_per_seed": questions_evaluated,
        "accuracy_by_category": by_cat,
        "metadata": metadata,
    }


def _seed_level_ci_half(values: list[float]) -> float:
    """1.96 * SEM half-width over per-seed values. Used for cost/latency where
    we don't have per-question granularity. Returns 0.0 for n<2."""
    if not values or len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return 1.96 * sem


def _aggregate_by_category(runs: list[dict]) -> dict:
    """Mean accuracy per category across seeds, with per-question bootstrap CI.

    For each category we pool the per-question scores (averaged across seeds)
    and run the same non-parametric bootstrap as the headline metric.
    """
    by_cat_runs: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        for cat, info in (r.get("accuracy_by_category") or {}).items():
            counts[cat].append(info.get("n", 0))
            by_cat_runs[cat].append(info)

    # Per-category per-question pool: re-walk recall_records and bucket by category.
    per_cat_pq: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for rec in run.get("recall_records", []) or []:
            cat = rec.get("category") or "unknown"
            qid = rec.get("question_id")
            if not qid:
                continue
            score = rec.get("score") or {}
            if isinstance(score, dict) and "accuracy" in score:
                per_cat_pq[cat][qid].append(float(score["accuracy"]))

    rng = np.random.default_rng(RNG_SEED + 1)
    out: dict = {}
    for cat, by_qid in per_cat_pq.items():
        pq = [float(np.mean(v)) for v in by_qid.values() if v]
        mean, lo, hi = bootstrap_ci(pq, rng=rng)
        ci_half = ((hi - lo) / 2.0) if not np.isnan(hi) else 0.0
        out[cat] = {
            "accuracy": mean if not np.isnan(mean) else 0.0,
            "accuracy_ci": ci_half,
            "ci_low_95": lo if not np.isnan(lo) else 0.0,
            "ci_high_95": hi if not np.isnan(hi) else 0.0,
            "n": int(np.mean(counts[cat])) if counts[cat] else len(pq),
        }
    return out


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "longmemeval-s"
    files = aggregate(corpus)
    print(f"\nWrote {len(files)} summary files.")
