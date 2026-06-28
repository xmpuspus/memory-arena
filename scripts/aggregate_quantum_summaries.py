"""Build per-strategy bootstrap summaries for the quantum rerankers only.

The full aggregate_bootstrap.py rebuilds every strategy's summary from its
``*_seed*.json`` files. The quantum smoke runs single-seed (no ``--seed``), so
their results land in ``results/<corpus>_<strategy>.json``. This script reuses
aggregate_bootstrap's ``_summarize`` to write ONLY the requested strategies'
``*_summary.json``, leaving the published 17 summaries and all seed files
untouched.

Usage:
    python scripts/aggregate_quantum_summaries.py [corpus] [strategy ...]
    # defaults: longmemeval-s  qiss sqr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_bootstrap import _summarize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def build(corpus: str, strategies: list[str]) -> list[Path]:
    written: list[Path] = []
    for strategy in strategies:
        # Prefer seed files if a seeded run exists; else the single-seed raw file.
        seed_files = sorted(RESULTS_DIR.glob(f"{corpus}_{strategy}_seed*.json"))
        if seed_files:
            runs = [json.loads(p.read_text()) for p in seed_files]
        else:
            raw = RESULTS_DIR / f"{corpus}_{strategy}.json"
            if not raw.exists():
                print(f"skip {strategy}: no results at {raw}")
                continue
            runs = [json.loads(raw.read_text())]
        summary = _summarize(strategy, runs)
        out = RESULTS_DIR / f"{corpus}_{strategy}_summary.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        written.append(out)
        print(
            f"{strategy}: n_seeds={summary['n_seeds']} n_q={summary['n_questions']} "
            f"acc={summary['mean_accuracy']:.3f} "
            f"recall@5={summary.get('mean_session_recall_at_k')} "
            f"cost=${summary['total_cost_usd']:.4f} -> {out.name}"
        )
    return written


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "longmemeval-s"
    strategies = sys.argv[2:] if len(sys.argv) > 2 else ["qiss", "sqr"]
    files = build(corpus, strategies)
    print(f"\nWrote {len(files)} summary file(s).")
