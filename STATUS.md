# Memory Arena Status

Updated: 2026-05-24

## Current state (v0.1.8, leveling release)

- **20 strategies on the leaderboard** (21 registered in
  `memory_arena/strategies/__init__.py`; `mem0g` is registered but excluded
  from the leaderboard, see below): pure-Python (12): `full_context`,
  `recency_window`, `naive_vector`, `bm25`, `hybrid_rrf`, `hyde`,
  `persona_profile`, `reflection`, `raptor`, `karpathy_llm_wiki`, `hipporag2`,
  `amem`. Quantum rerankers (2): `qiss`, `sqr`. Vendor SDK (6): `mem0`,
  `graphiti`, `graphiti_falkor`, `cognee`, `langmem`, `memori`.
- **`mem0` (31.4%) and `langmem` (33.0%) are leveled to Claude Sonnet.** mem0's
  old 50.9% was v1 + gpt-4o-mini; the drop is ~16pp v1->v2 regression, ~3pp
  model. langmem *rose* +6pp on Sonnet (26.9% -> 33.0%), now the top vendor.
  `graphiti` and `cognee` can't be cleanly leveled in the shared env (graphiti
  0.13 has no Anthropic client; cognee 1.0.3 vs the pinned starlette) and stay
  vendor-default. `mem0g` is excluded: v2 removed the OSS graph store.
- **Corpus on disk**: LongMemEval-S smoke subset, **16 questions across 4
  categories** (`information_extraction`, `multi_session_reasoning`,
  `temporal`, `knowledge_update`, n=4 each), 82 sessions.
- **Judge**: `claude-opus-4-7`. **Generator**: `claude-sonnet-4-6` for
  strategies that do not pin their own. **Embeddings**:
  `text-embedding-3-large` (3072 dims).
- **Bootstrap**: 13/20 benchmarked strategies have 3-seed CIs (mem0 now 5); the remainder are
  single-seed (no CI rendered).
- **Cross-judge sanity**: 19 of 20 strategies re-graded under GPT-4o yields
  Spearman +0.967 on the ranks (GPT-4o more lenient in absolute terms, same
  order; see `results/cross_judge_report.json`).
- **Tests**: 362, mock-based, no live API calls.
- **CI**: ruff check + ruff format + pytest gates a PR.

## Acceptance gates (last full run)

| Gate | Status | Evidence |
| --- | --- | --- |
| `pip install -e '.[dev]'` clean | PASS | virtualenv at `.venv/`; package version installs without errors. |
| `pip install 'memory-arena[mem0,graphiti,cognee,langmem,memori]'` | PASS | All vendor SDKs install alongside the package. |
| `ruff check .` | PASS | "All checks passed!" |
| `ruff format --check .` | PASS | All files formatted. |
| `pytest tests/ -v` | PASS (362) | Mock-based, no live API calls. |
| `cd web && npx next build` | PASS | 4 static routes, ~91-92KB first-load JS. |
| Static bundle in `memory_arena/static/` | PASS | Mounted by `chatbot/api.py` lifespan. |
| `memory-arena --help` | PASS | 11 commands. |
| `memory-arena health` | PASS | Reports API key state and 21 registered strategies. |
| `docker compose up -d neo4j postgres` | PASS | Postgres on host port 5433, Neo4j on 7687, both healthy. |
| 7-axis evaluator | PASS | structural + sources + judge + memo + temporal + update + abstention F1. |
| Real benchmark on all in-scope strategies | PASS (20 on leaderboard; `mem0` leveled to v2+Sonnet 31.4%/5-seed; `mem0g` excluded) | LongMemEval-S smoke with real LLM calls. |
| Per-category metrics return null when N=0 | PASS | Runner emits `null`; dashboard renders "-". |
| Sortable benchmark dashboard with latency | PASS | `web/app/benchmark/page.tsx`; default sort = accuracy desc. |
| Real screenshots of Home/Benchmark/Recall Lab/Arena | PASS | Captured against `memory-arena serve` on port 8002. |
| Showcase hero + social card | PASS | `docs/showcase.png` (sorted-bar leaderboard), `docs/social-preview.png` (OG card). |

## Shipped in v0.1.8

- **Cross-judge** (19-way, GPT-4o, Spearman +0.967 on the ranks).
- **`amem` (38.7%) + `hipporag2` (54.5%) benchmark numbers** in `results/`.
- **mem0 + langmem leveled** to Claude Sonnet; **mem0g dropped**; board now 20.
- Recall Lab wheel-fallback fix; per-strategy timeout + live cost ticker.

Deferred to v0.2: Benjamini-Hochberg q-value column, generator robustness
sweep, per-question Pearson cross-judge, the full 500-question corpus.

## Next steps (v0.2)

1. Tuned-mode runner that records vendor-recommended config for each system.
2. Live tests in `tests/live/` for each vendor SDK.
3. Audit module retargeted as a memory-gap analyzer.
4. Arena ELO engine wired so the dashboard leaderboard reflects actual matches.
5. **Full LongMemEval-S corpus** (500 questions) replacing the smoke subset
   for the headline table; tightens per-category CIs from N=4 to ~125.
6. Mem0 / Mem0g / Cognee session-aware ingest formatter pass.
7. Tests for the 9 retriever strategies.
8. **Provider-managed memory track** (OpenAI Memory, Anthropic memory tool)
   separated from the headline so the generator-control caveat is explicit.

## Caveats

- **Memori cloud quota**: Memori 3.x routes augmentation through a cloud
  quota service that 429s anonymous IPs. Set `MEMORI_API_KEY` for full
  throughput; without it accuracy floors near zero.
- **Full-context cost cap**: `full_context` hits the cap on the smoke
  subset (12/16 questions evaluated at `--cost-cap 5`). Bump to `25+` to
  evaluate all 16.
- **mem0 / mem0g LLM swap**: both vendor SDKs are scored using
  `gpt-4o-mini` for their internal extraction call due to a vendor
  Anthropic-adapter bug in `mem0ai==0.1.114`. Embeddings are pinned to
  ours (`text-embedding-3-large`). This is the one place in the table
  where the generator differs from the cross-strategy default; the
  caveat is hoisted to its own README subsection.

## v0.1.5 numbers (historical, 16 questions / 4 categories, smoke)

These were the v0.1.5 ship numbers (single-seed for several strategies);
the v0.1.6 numbers in `results/*_summary.json` supersede these. Kept
here only as a regression marker so anyone bisecting can confirm
direction.

| Strategy | Accuracy | Recall@k | Cost | Latency |
| --- | --- | --- | --- | --- |
| reflection | 44.12% | 87.50% | $0.488 | 7579ms |
| persona_profile | 43.38% | 89.06% | $0.200 | 6841ms |
| naive_vector | 39.63% | 89.06% | $0.087 | 3548ms |
| raptor | 36.40% | 88.33% | $0.107 | 3461ms |
| hybrid_rrf | 35.13% | 80.73% | $0.711 | 16737ms |
| hyde | 31.44% | 76.88% | $0.182 | 11815ms |
| bm25 | 27.13% | 79.17% | $0.090 | 10499ms |
| karpathy_llm_wiki | 26.13% | 57.81% | $1.270 | 3953ms |
| full_context | 25.50% | 8.33% | $5.263 | 8551ms |
| langmem | 18.13% | 0.00% | $0.044 | 2950ms |
| graphiti | 15.75% | 0.00% | $0.032 | 3745ms |
| recency_window | 5.75% | 6.25% | $0.318 | 5284ms |
| mem0g | 2.13% | 0.00% | $0.011 | 3177ms |
| mem0 | 1.38% | 0.00% | $0.010 | 2558ms |
| cognee | 1.00% | 0.00% | $0.014 | 2736ms |
| memori | 0.88% | 0.00% | $0.013 | 2214ms |

The published v0.1.6 numbers in the README are the bootstrap-aggregated
3-seed (where available) values from `results/longmemeval-s_*_summary.json`.

## How to reproduce v0.1.6

```bash
cd memory-arena
source .venv/bin/activate
export MEM_ARENA_NEO4J_PASSWORD=...

docker compose up -d neo4j postgres

memory-arena download-longmemeval
memory-arena ingest-sessions --corpus longmemeval-s

for SEED in 0 1 2; do
  memory-arena benchmark --corpus longmemeval-s \
    --strategy 'bm25,naive_vector,recency_window,hybrid_rrf,hyde,persona_profile,reflection,raptor,karpathy_llm_wiki' \
    --cost-cap 3 --top-k 5 --seed $SEED
done

for SEED in 0 1 2; do
  memory-arena benchmark --corpus longmemeval-s \
    --strategy 'mem0,graphiti,langmem,memori' \
    --cost-cap 3 --top-k 5 --seed $SEED
done

python scripts/aggregate_bootstrap.py
python scripts/render_readme.py
python scripts/build_hero_chart.py
python scripts/build_taxonomy_chart.py
python scripts/build_pairwise_chart.py
```
