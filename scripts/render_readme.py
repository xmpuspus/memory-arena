"""Render README.md headline table from per-strategy bootstrap summaries.

Reads results/longmemeval-s_<strategy>_summary.json and rewrites the
"## Real Benchmark Numbers" section of README.md with the bootstrap means,
95% CIs, ingest health badge, and Recall@k as "—" when not measurable.

Idempotent: re-runnable. Looks for `<!-- BENCHMARK_TABLE_START -->` and
`<!-- BENCHMARK_TABLE_END -->` markers and replaces what's between them.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"

START = "<!-- BENCHMARK_TABLE_START -->"
END = "<!-- BENCHMARK_TABLE_END -->"

PLAIN_ENGLISH = {
    "full_context": (
        "Pastes the entire chat history into every prompt. Ceiling reference; expensive."
    ),
    "recency_window": "Only remembers the last 20 messages. Floor baseline.",
    "naive_vector": "Embeds every message and pulls the closest matches by cosine similarity.",
    "bm25": "Old-school keyword search. What Google did before vectors.",
    "hybrid_rrf": "Reciprocal rank fusion of vector + BM25. Blended ranking.",
    "hyde": (
        "Guesses what the answer might look like first, then searches for messages like the guess."
    ),
    "persona_profile": (
        "Builds a one-page bio of the user up front and pastes it into every answer."
    ),
    "reflection": (
        "Periodically writes journal-style summaries of recent sessions, then searches both."
    ),
    "raptor": "Hierarchical k-means clustering with LLM cluster summaries (Sarthi et al. 2024).",
    "karpathy_llm_wiki": (
        "LLM maintains a markdown wiki of entity pages with cross-links and citations."
    ),
    "amem": (
        "A-MEM (NeurIPS 2025): writes an LLM note per memory and periodically links "
        "and revises related notes."
    ),
    "hipporag2": (
        "HippoRAG 2 (ICML 2025): extracts entity triples into a graph and ranks passages "
        "by personalized PageRank."
    ),
    "qiss": (
        "Reranks vector hits by quantum fidelity (cosine squared) over the same embeddings; "
        "pure NumPy, optional multi-query superposition fusion."
    ),
    "sqr": (
        "Reranks vector hits with a real SWAP-test circuit on the Qiskit Aer simulator "
        "(PCA-reduced amplitude encoding, exact statevector)."
    ),
    "mem0": "Vendor SDK that extracts standalone facts and stores them as memories.",
    "mem0g": "Mem0 plus a Neo4j graph that links facts to entities.",
    "graphiti": "Temporal knowledge graph with valid_at/invalid_at edges (Zep OSS).",
    "graphiti_falkor": (
        "Same Graphiti algorithm on FalkorDB (Redis graph engine) instead of Neo4j; "
        "isolates the database to test the engine's latency claim."
    ),
    "cognee": "Knowledge-graph memory: add → cognify → search(GRAPH_COMPLETION).",
    "langmem": (
        "LangChain's memory store: extracts facts as they happen and recalls them by similarity."
    ),
    "memori": "SQL-native fact store with augmentation pipeline.",
}


def _fmt_pct(value: float | None, ci: float | None = None) -> str:
    if value is None:
        return "-"
    s = f"{value * 100:.1f}%"
    if ci is not None and ci > 0:
        s += f" ±{ci * 100:.1f}"
    return s


VENDOR_STRATEGIES = {"mem0", "mem0g", "graphiti", "graphiti_falkor", "cognee", "langmem", "memori"}


def _fmt_cost(value: float | None, strategy: str, status: str) -> str:
    """Render the cost cell.

    Pure-Python: full cost; no footnote.
    Vendor SDK: memory-arena-paid generation cost only; ‡ footnote flags
    that vendor-internal LLM calls (mem0 extraction, cognee cognify, etc.)
    are NOT counted. Showing the partial measurable cost is more honest
    than showing "—" and contradicting the hero chart.
    """
    if status == "config-failed-at-default":
        return "-†"
    if value is None:
        return "-"
    if value < 0.01:
        s = f"${value:.4f}"
    elif value < 1:
        s = f"${value:.3f}"
    else:
        s = f"${value:.2f}"
    if strategy in VENDOR_STRATEGIES:
        s += "‡"
    return s


def _fmt_recall(value: float | None) -> str:
    if value is None:
        return "-§"
    return f"{value * 100:.1f}%"


def _fmt_lat(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}ms"


def render_table() -> str:
    rows: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("longmemeval-s_*_summary.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        return "_No bootstrap summaries yet. Run `python scripts/aggregate_bootstrap.py` first._"
    rows.sort(key=lambda r: -r.get("accuracy", 0.0))

    lines = [
        "| Strategy | Accuracy (95% CI) | Recall@5 | Cost | Latency | Status | What it does |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        recall_at_k_measurable = r.get("mean_session_recall_at_k") is not None
        rk = r.get("mean_session_recall_at_k") if recall_at_k_measurable else None
        # memori is cloud-throttled without MEMORI_API_KEY; flag inline.
        plain = PLAIN_ENGLISH.get(r["strategy"], "")
        if r["strategy"] == "memori":
            plain += " ‖"
        lines.append(
            "| `"
            + r["strategy"]
            + "` | "
            + _fmt_pct(r.get("accuracy"), r.get("accuracy_ci"))
            + " | "
            + _fmt_recall(rk)
            + " | "
            + _fmt_cost(r.get("total_cost_usd"), r["strategy"], r.get("status", "ok"))
            + " | "
            + _fmt_lat(r.get("avg_recall_latency_ms"))
            + " | "
            + r.get("status", "ok")
            + " | "
            + plain
            + " |"
        )
    lines.append("")
    has_failed = any(r.get("status") == "config-failed-at-default" for r in rows)
    footnote_parts = ["**Footnotes.**"]
    if has_failed:
        footnote_parts.append(
            "**†** vendor's default config caused >50% ingest failure during this run; "
            "result reflects an empty store. PRs welcome (see CHANGELOG v0.2 invitation)."
        )
    footnote_parts.extend(
        [
            "**‡** memory-arena-paid generation cost only; vendor SDKs run additional "
            "internal LLM calls (mem0 extraction, graphiti entity extraction, langmem "
            "fact extraction, cognee cognify, etc.) that aren't counted. True cost is "
            "this number plus the vendor-internal component.",
            "**§** Recall@k requires the strategy to return chat-session pointers; "
            "LangMem/Cognee/Memori store extracted facts, neither maps to LongMemEval "
            "session IDs.",
            "**‖** memori at this score reflects the no-`MEMORI_API_KEY` baseline (cloud "
            "augmentation throttled to ~zero). Set the key for the vendor's intended "
            "throughput; PRs welcome.",
        ]
    )
    lines.append(" ".join(footnote_parts))
    lines.append("")
    n_seeds_distribution = sorted({r.get("n_seeds", 1) for r in rows})
    if n_seeds_distribution == [3]:
        seed_note = "3-seed bootstrap mean ± 95% CI"
    elif n_seeds_distribution == [1]:
        seed_note = "single-seed (no CI)"
    else:
        # Mixed seed counts (e.g. some 3-seed, some 5-seed, some single): report
        # how many have multi-seed (>=3) bootstrap CIs, not an exact-count match.
        n_multi = sum(1 for r in rows if r.get("n_seeds", 1) >= 3)
        seed_note = (
            f"{n_multi}/{len(rows)} strategies have 3-or-more-seed bootstrap CIs; "
            f"the rest are single-seed (no CI)"
        )
    lines.append(
        f"_{len(rows)} strategies. {seed_note}. "
        "top_k=5 held constant. Judge: claude-opus-4-7. Generation: claude-sonnet-4-6 "
        "(strategies that route through their own SDK use the SDK default). "
        f"Hardware/SDK versions stamped in `results/<strategy>_summary.json` `metadata` block._"
    )
    return "\n".join(lines)


def update_readme() -> bool:
    txt = README.read_text()
    if START not in txt or END not in txt:
        # Insert markers after the existing "Real Benchmark Numbers" header
        marker = "## Real Benchmark Numbers"
        if marker not in txt:
            print(f"Could not find '{marker}' in README. Add markers manually.")
            return False
        end_section = txt.find("## ", txt.index(marker) + len(marker))
        if end_section == -1:
            end_section = len(txt)
        replaced = (
            txt[: txt.index(marker)]
            + f"## Real Benchmark Numbers (all strategies)\n\n{START}\n"
            + render_table()
            + f"\n{END}\n\n"
            + txt[end_section:]
        )
    else:
        before = txt.split(START)[0]
        after = txt.split(END, 1)[1]
        replaced = before + START + "\n" + render_table() + "\n" + END + after
    README.write_text(replaced)
    print(f"Updated {README} ({len(replaced)} bytes)")
    return True


if __name__ == "__main__":
    update_readme()
