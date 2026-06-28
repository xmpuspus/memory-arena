"""Generate docs/per-question-comparison.md — static 'ask all 16' page.

For every question in the smoke corpus, show ground truth + a row per
strategy with the answer it produced (truncated) and the judge score.
This is the closest thing to an interactive playground you can ship
without standing up a service: a static markdown table per question,
all in one document, generated from the per-seed result JSONs.

Re-run: `python scripts/build_per_question_comparison.py`
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
QUESTIONS_FILE = REPO_ROOT / "datasets" / "longmemeval-s" / "processed" / "questions.jsonl"
OUT = REPO_ROOT / "docs" / "per-question-comparison.md"

DISPLAY_ORDER = [
    "mem0g",
    "mem0",
    "persona_profile",
    "naive_vector",
    "reflection",
    "raptor",
    "hybrid_rrf",
    "hyde",
    "full_context",
    "bm25",
    "langmem",
    "graphiti",
    "karpathy_llm_wiki",
    "cognee",
    "recency_window",
    "memori",
]

CATEGORY_BLURB = {
    "information_extraction": "Single fact buried in one session.",
    "multi_session_reasoning": "Counts/lists drawn from many sessions.",
    "knowledge_update": "Latest version of a fact that changed over time.",
    "temporal": "Time-aware question (when, how long, in what order).",
}


def _load_questions() -> dict[str, dict]:
    out = {}
    with open(QUESTIONS_FILE) as f:
        for ln in f:
            q = json.loads(ln)
            out[q["id"]] = q
    return out


def _load_records() -> dict[str, dict[str, dict]]:
    """{question_id: {strategy: record}} — first seed only for compactness."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for strategy in DISPLAY_ORDER:
        path = RESULTS / f"longmemeval-s_{strategy}_seed0.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        for r in d.get("recall_records", []):
            out[r["question_id"]][strategy] = r
    return out


def _trunc(s: str, n: int = 240) -> str:
    s = (s or "").replace("\n", " ").replace("|", "\\|").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _score(r: dict) -> tuple[float, float | None]:
    sc = r.get("score") or {}
    if isinstance(sc, dict):
        acc = float(sc.get("accuracy", 0.0))
    else:
        acc = float(sc or 0.0)
    ir = r.get("ir") or {}
    rk = None
    if isinstance(ir, dict):
        v = ir.get("session_recall_at_k")
        if v is not None:
            rk = float(v)
    return acc, rk


def render() -> None:
    questions = _load_questions()
    records = _load_records()

    lines: list[str] = []
    lines.append("# Memory Arena — per-question comparison\n")
    lines.append(
        "Every question in the LongMemEval-S smoke corpus, with what "
        "every strategy answered (seed 0) and how the judge scored it. "
        "Truncated to 240 chars per cell to keep the page readable; "
        "full answers in `results/longmemeval-s_<strategy>_seed0.json`.\n"
    )
    lines.append("Sorted by category. Inside each category, by question_id.\n")

    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid, q in questions.items():
        by_cat[q["category"]].append(qid)

    # Stable category order matching the README narrative
    cat_order = [
        "information_extraction",
        "multi_session_reasoning",
        "knowledge_update",
        "temporal",
    ]
    for cat in cat_order:
        if cat not in by_cat:
            continue
        lines.append(f"\n## {cat}\n")
        lines.append(f"_{CATEGORY_BLURB.get(cat, '')}_\n")
        for qid in sorted(by_cat[cat]):
            q = questions[qid]
            gt = q["ground_truth"]
            lines.append(f"\n### `{qid}` — {q['question']}\n")
            lines.append(f"**Ground truth:** {gt.get('answer')}\n")
            supp = gt.get("supporting_session_ids", []) or []
            if supp:
                lines.append(
                    "**Supporting sessions:** " + ", ".join(f"`{s}`" for s in supp[:3]) + "\n"
                )
            lines.append("\n| Strategy | Acc | R@5 | Answer |")
            lines.append("|----------|----:|----:|--------|")
            for strategy in DISPLAY_ORDER:
                r = records.get(qid, {}).get(strategy)
                if r is None:
                    lines.append(f"| `{strategy}` | — | — | _(no record)_ |")
                    continue
                acc, rk = _score(r)
                rk_s = f"{rk:.2f}" if rk is not None else "—"
                ans = _trunc(r.get("answer", ""), 220)
                lines.append(f"| `{strategy}` | {acc:.2f} | {rk_s} | {ans} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    render()
