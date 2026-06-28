"""Dump a markdown leaderboard from the latest seed-0 result JSONs.

Output is printed to stdout. Intended for pasting into the README under
"## Real benchmark numbers". Numbers are seed-0 only; multi-seed CIs come
from `aggregate_bootstrap.py`'s `_summary.json`.

Usage:
    .venv/bin/python scripts/dump_results_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

# Plain-English column from the prior README's table (preserved manually).
DESCRIPTIONS = {
    "naive_vector": "ChromaDB + text-embedding-3-large.",
    "bm25": "Pure-Python lexical baseline (rank-bm25).",
    "recency_window": 'Last 20 turns. The "chatbot with no memory" baseline.',
    "hyde": "Hypothetical Document Embeddings.",
    "persona_profile": "One-shot Haiku persona stuffed as system context.",
    "hybrid_rrf": "RRF fusion of vector + BM25 (k=60).",
    "raptor": "Hierarchical k-means with LLM cluster summaries.",
    "reflection": "Synthetic LLM-authored summaries every 4 sessions.",
    "karpathy_llm_wiki": "Three-layer LLM-maintained wiki + lint pass every 10 sessions.",
    "full_context": (
        "Stuffs every turn into the prompt. Expensive and forgets nothing "
        "until the prompt overflows."
    ),
    "mem0": "Mem0 v2 vendor default (Anthropic-Sonnet now that v2 fixed the adapter bug).",
    "mem0g": "Mem0 v1 + Neo4j graph_store (v2 dropped graph_store from OSS).",
    "graphiti": "Chunked 2-turn episodes, gpt-4o entity extraction.",
    "cognee": "Cognee 1.x with GRAPH_COMPLETION search.",
    "langmem": "LangGraph InMemoryStore + create_memory_store_manager.",
    "memori": "Memori 3.x BYODB Postgres; cloud quota throttled (see caveats).",
    "amem": "A-MEM (NeurIPS 2025): structured notes + LLM-driven link evolution.",
    "hipporag2": "HippoRAG 2 (ICML 2025): OpenIE triples + personalized PageRank.",
}


def main() -> None:
    rows: list[dict] = []
    for p in sorted(RESULTS.glob("longmemeval-s_*_seed0.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if d.get("strategy") in DESCRIPTIONS:
            rows.append(d)
    rows.sort(key=lambda r: r.get("accuracy", 0.0), reverse=True)

    print("| Strategy | Accuracy | Recall@k | Cost | Latency | What it does, in plain English |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        s = r["strategy"]
        acc = r.get("accuracy", 0.0)
        recall = r.get("recall_at_k", 0.0)
        cost = r.get("total_cost_usd", 0.0)
        lat = r.get("avg_recall_latency_ms", 0.0)
        desc = DESCRIPTIONS.get(s, "")
        print(f"| `{s}` | {acc:.2%} | {recall:.2%} | ${cost:.4f} | {lat:.0f}ms | {desc} |")

    # Errors summary
    err_lines = []
    for r in rows:
        errs = r.get("errors") or []
        if errs:
            err_lines.append(f"  - `{r['strategy']}`: {len(errs)} swallowed errors")
    if err_lines:
        print("\n**Strategies with non-empty `errors[]`:**\n")
        for line in err_lines:
            print(line)


if __name__ == "__main__":
    main()
