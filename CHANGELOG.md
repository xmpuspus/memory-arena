# Changelog

All notable changes to Memory Arena.

## [Unreleased]

## [0.1.8] - 2026-06-28 - Level the field: vendor SDKs on the same model, mem0g dropped, full cross-judge

### Changed

- **Leveled the vendor SDKs to the harness model where possible.** `mem0`
  (now mem0ai 2.0.2, v2) and `langmem` run their internal fact-extraction on
  Claude Sonnet, the same model as the pure-Python baselines, so the
  leaderboard isolates the memory architecture rather than the model. The
  generator confound the README used to call its "biggest asterisk" is ~3pp
  (mem0), measured directly; the bigger mem0 mover was the v1->v2 OSS
  regression (~16pp). `langmem` *rose* +6pp on Sonnet (26.9% -> 33.0%), so its
  default model was selling it short. `graphiti` (no Anthropic client on
  graphiti-core 0.13) and `cognee` (starlette pin conflict) can't be cleanly
  leveled in one env and stay vendor-default, documented.
- **Dropped `mem0g` from the leaderboard.** mem0 v2.0.0 removed the OSS graph
  store, so mem0g only runs on the deprecated v1 + gpt-4o-mini. Graph memory is
  covered by `graphiti`; the strategy stays in the repo. The board is now 20.

### Added

- **`amem` (A-MEM, NeurIPS 2025) and `hipporag2` (HippoRAG 2, ICML 2025)
  benchmarked** on 3 seeds. `hipporag2` (54.5%) posts the top point estimate on
  the board, a from-scratch pure-Python graph method ahead of every vendor SDK.
- **19-way cross-judge.** Re-grading 19 of the 20 strategies with GPT-4o yields
  Spearman rho = +0.967 on the ranks (`results/cross_judge_report.json`);
  GPT-4o is more lenient in absolute terms but agrees on order, and also ranks
  every vendor below the pure-Python tier.
- **Per-strategy wall-clock timeout** in the runner so one hung strategy can't
  block the batch, and a **live cost ticker** in the benchmark progress.

### Fixed

- **Recall Lab worked only in local dev.** The dashboard's recall-records
  endpoint read a bare `{corpus}_{strategy}.json` the wheel snapshot doesn't
  ship; it now falls back to the seed-0 file. Plus a reconciliation pass that
  corrected every drifted count and number across the README and docs.

## [0.1.7] - 2026-06-22 - Quantum rerankers (QISS + SQR) and Graphiti-on-FalkorDB

### Added

- **`qiss` strategy (Quantum-Inspired Semantic Similarity).** Pure-NumPy
  reranker over `naive_vector`'s Chroma index, zero new dependencies. Reranks
  candidates by quantum-state fidelity `Tr(rho_q rho_d) = |<q|d>|^2 = cos^2`
  over the same OpenAI embeddings, so its single-query ordering equals
  `naive_vector`'s by construction (the unit-tested invariant). On
  LongMemEval-S smoke (16 questions, top_k=5, 3-seed bootstrap): 50.5% accuracy
  (95% CI overlapping the whole top tier, tied with naive_vector's 49.2%),
  recall@5 ~0.85 (matches naive_vector within retrieval noise), ~$0.09. An
  optional multi-query
  superposition mode (off by default) fuses sub-query states into
  `|Q> = a1|q1> + a2|q2> + ...` and scores with interference cross-terms that
  classical rank fusion cannot express.
- **`sqr` strategy (Simulated Quantum Reranker).** Runs a real SWAP-test
  circuit on the Qiskit Aer simulator (exact statevector by default; shots is
  a config knob). Embeddings are PCA-reduced to `2^n_qubits` dims (16 at the
  default 4 qubits), amplitude-encoded, and compared via the ancilla
  probability `|<q|d>|^2 = 2*P(0) - 1`. On LongMemEval-S smoke: 40.4% accuracy,
  66.7% recall@5, $0.067, 3714ms. The accuracy drop vs `naive_vector` is the
  honest cost of amplitude encoding: squeezing 3072-d embeddings into 16 dims
  keeps only 36% of the variance (see
  `results/longmemeval-s_quantum_diagnostics.json`). Simulated quantum
  machinery buys nothing over the closed-form cosine on this task; the finding
  is the dimensionality-reduction tax, not a quantum advantage. Needs
  `pip install 'memory-arena[quantum]'` (qiskit + qiskit-aer).
- **Quantum diagnostics** at `results/longmemeval-s_quantum_diagnostics.json`
  via `scripts/quantum_diagnostics.py`: PCA variance-explained vs n_qubits
  (15.3% at 4 dims up to 88.4% at 128 dims) and SWAP-test error vs shots
  (mean abs error 0.0645 at 64 shots down to 0.0057 at 16384). Computed, not
  narrated.
- **`graphiti_falkor` strategy.** Zep's Graphiti temporal knowledge graph
  running on FalkorDB (a Redis-based, GraphBLAS graph engine) instead of
  Neo4j. It subclasses `graphiti` and swaps only the graph driver, so ingest
  and recall are identical and the engine is the single variable. On
  LongMemEval-S smoke (16 questions, seed 0, top_k=5): 22.5% accuracy, 63.0%
  recall@5, 3223ms recall latency, $0.034 measurable cost. Under the same
  library version (graphiti-core 0.17), Neo4j scores 26.7% / 66.1% recall@5 /
  3054ms vs FalkorDB's 22.5% / 63.0% / 3223ms: statistically tied (16
  questions, single seed, ~14pp CI), with Neo4j marginally ahead on every axis
  and zero ingest failures (FalkorDB's driver hits ~1.2%). FalkorDB's headline
  graph-query latency advantage does not surface end-to-end because the
  answer-generation LLM call dominates recall latency. The published `graphiti`
  row stays on 0.13 for baseline reproducibility, so the table's
  graphiti-vs-falkor delta is version-confounded (0.17 alone lifts Neo4j ~3pp).
- `amem` and `hipporag2` are now listed in the README strategy tables
  (registered; their benchmarks land in v0.1.8).

### Changed

- Strategy count is now 21 registered, 19 benchmarked (`amem` and `hipporag2`
  pending). Badges, headline, taxonomy figure (19 of 21 plotted), strategy
  tables, and the dashboard updated.

### Findings (retrieval experiments)

- `scripts/quantum_experiments.py` runs deterministic, retrieval-level sweeps
  (statevector reranking has no LLM/sampling, so Recall@5 is exact and noise
  free). Two definitive results, written to
  `results/longmemeval-s_quantum_experiments.json` and visualized in
  `docs/quantum_experiments.png` (`scripts/build_quantum_chart.py`):
  - **Quantum interference is inert here.** Coherent superposition fusion equals
    the incoherent mixture in every cell (0.938 == 0.938 overall, 0.812 == 0.812
    on multi-session reasoning) and both stay below single-query cosine (0.953).
    The cross-terms are nonzero but too small to reorder the top-5; dense text
    sub-query embeddings are near-collinear. The qiss multi-query mode stays off
    by default.
  - **sqr's loss is purely the PCA encoding.** Recall@5 climbs 0.75 to 0.90 as
    n_qubits goes 4 to 7 (variance 37% to 84%), approaching cosine's 0.95, so the
    SWAP test is faithful. End-to-end accuracy stays flat (sqr@4 40.4% to sqr@7
    40.9%): more qubits buy bigger circuits for retrieval still at or below
    cosine. No operating point beats the closed-form cosine.
- **Root cause (`scripts/quantum_headroom.py`, `results/longmemeval-s_quantum_headroom.json`):
  there is no reranker-side fix, quantum or classical.** Quantum fidelity is
  `|<q|d>|^2 = cos^2`, a monotone of cosine, so quantum reranking reproduces
  cosine's exact order by construction. Headroom does exist (multi-session
  reasoning: cosine 0.812 vs pool ceiling 0.938, +12.5pp), but the gold sessions
  cosine ranks low sit low because they phrase the same concept in different
  words, and every embedding-similarity score (cosine, density matrix
  `<q|rho|q>`, centroid) ranks them equally low: all tie at 0.812 there. The
  bottleneck is the embedding's semantic coverage, not the scoring math, so the
  only levers are upstream of similarity (query rewriting, cross-session
  aggregation, graph propagation), none of which is what the quantum methods do.
- **Compression cost frontier (`scripts/compression_frontier.py`,
  `results/longmemeval-s_compression_frontier.json`, `docs/compression_frontier.png`):
  the quantum encoding is dominated on the one cost axis it theoretically
  claimed.** In a RAG pipeline the similarity computation quantum touches is
  negligible next to LLM tokens and embedding storage / memory bandwidth, so the
  real lever is compression. Ranking turns in the compressed representation
  (store compressed, search compressed): binary quantization holds full Recall@5
  (0.938) at 384 bytes (32x) and 128 bytes (96x, Matryoshka+binary), and 0.922
  at 64 bytes; Matryoshka truncation holds to about 256 dims. The quantum
  encoding (PCA-16 amplitude state, verified rank-identical to the real SWAP
  test) gets 0.766 at 64 bytes, beaten by classical binary_512 (0.922 at the
  same 64 bytes) and even binary_256 (0.849 at half the bytes). Verified that
  Matryoshka truncate+renormalize equals the text-embedding-3-large API
  `dimensions` output (cosine 1.0000).
- Fixed a latent bug in `qiss._decompose_query`: the cheap classify path made
  the model refuse first-person questions and emit advice instead of search
  queries. Now uses a strict rewriter prompt on the generate model plus a
  refusal filter.

### Notes

- `qiss` is always available (pure NumPy). `sqr` is gated behind the optional
  `[quantum]` extra and dropped cleanly from the registry when qiskit is
  absent, so the core install and CI stay light.
- `graphiti_falkor` needs `graphiti-core>=0.17` (first release with the
  FalkorDB driver), which requires `pydantic>=2.11.5` and so cannot coexist
  with the core `pydantic==2.10.4` pin. Install it in its own environment:
  `pip install 'memory-arena[falkordb]'`, then `docker compose up -d falkordb`
  (host port 6381). Intermittent `Failed to parse query parameter 'nodes'`
  warnings from the FalkorDB driver cause a ~1.2% ingest-failure rate.

## [0.1.6] - 2026-05-02 - Arena framing, sortable dashboard, live recall lab

### Added

- **Sortable benchmark dashboard.** Click any column on `/benchmark` to
  re-rank by accuracy, recall@5, cost, or latency. Default sort is
  accuracy descending. Column headers carry hover tooltips defining
  every metric.
- **Live Recall Lab** at `/recall-lab`: per-question HIT/MISS for every
  strategy, real `supporting_session_ids` against expected, drill-down
  into judge rationale and answer text. Backed by
  `/api/recall-records/{corpus}/{strategy}`.
- **Hero chart** at `docs/hero.png`, accuracy vs log-cost Pareto with
  95% CI bars, navy/coral/grey palette, inline strategy labels.
  Generated by `scripts/build_hero_chart.py`.
- **Taxonomy chart** at `docs/taxonomy.png` placing 16 strategies in
  one design space (write-time × representation; tier emphasis via
  shape, color, size, saturation). Shared style module at
  `scripts/_chart_style.py`.
- **Pairwise significance heatmap** at `docs/pairwise.png` showing
  which strategy gaps clear the 95% bootstrap CI threshold.
- **3-seed bootstrap CIs** on every metric the README displays (`--seed
  0,1,2`). Aggregated by `scripts/aggregate_bootstrap.py` into
  per-strategy `_summary.json` with mean ± 95% CI half-widths.
- **Reproducibility metadata** stamped into every result JSON: commit
  SHA, installed package versions for all 16 strategy SDKs + core deps,
  model IDs (Sonnet 4.6 / Haiku 4.5 / Opus 4.7 / text-embedding-3-large),
  host, timestamp, seed.
- **Ingest-health badge**, any strategy with >50% ingest failure at
  default config gets `status: "config-failed-at-default"`. Headline
  table shows the badge; vendors invited to PR a working default.
- **`--version` flag** on the CLI.
- **`--port 0`** auto-picks a free port for `serve` and `demo`.
- **`--host` flag** opt-in for LAN exposure (with explicit warning).
- **`memory-arena demo`** opens `/` (the dashboard) directly.
- **Cross-judge sweep tooling** at `scripts/cross_judge.py` (used to
  produce the v0.1.7 GPT-4o vs Opus 4.7 stability report).
- **`results/.costs.jsonl`**, append-only ledger so the orchestrator
  can sum cumulative spend across runs.
- **PyPI placeholders** `memoryarena` and `memarena` claimed as
  redirects to `memory-arena`.

### Changed

- **Tagline** to the Arena framing: "The vendor-neutral arena for
  agent memory. 16 strategies, one evaluator, full reproducibility."
- **README** opens with the one-liner, hero chart, three "what
  surprised me" bullets, then "Try it in 10 seconds." Vendor caveat
  follows the chart with a v0.2 invitation framing.
- **CLI default bind** is `127.0.0.1` for `serve` and `demo`. Override
  via `--host 0.0.0.0` for LAN exposure.
- **`docker-compose.yml`** ports bound to `127.0.0.1` only.
- **Mem0 default LLM** is `openai/gpt-4o-mini` (Mem0's documented
  default).
- **Top_k=5** held constant across all 16 strategies.
- **Recall@k semantics**, vendors whose data model doesn't carry
  chat-session pointers (LangMem, Cognee, Memori) declare
  `recall_at_k_measurable = False` at the class level; the aggregator
  emits null and the dashboard renders "-".
- **Embedder** read from `settings.embedding_model` across vendor
  strategies that accept an embedder kwarg, so vendor and pure-Python
  strategies use the same embedding model.
- **Tenacity** wired into `memory_arena/llm/client.py` retries
  (3 attempts, exponential 1s/2s, per-attempt 60 s timeout, WARNING
  log via `before_sleep`).
- **`update_precision`** returns `None` when fact_versions is empty;
  the bootstrap aggregator propagates `None` through to the dashboard.
- **Smoke-loader path** points at `datasets/longmemeval-s/questions/smoke_synthetic/`
  where the synthetic abstention YAML files actually live, so the
  abstention F1 axis loads on smoke runs that ship with the synthetic
  file.
- **`scripts/robustness.py`** rewritten to call
  `run_memory_benchmark` with the current signature.
- **Strategy + test counts** reconciled to **16 strategies**, **274
  tests** (`pytest --collect-only -q`) across README, CLAUDE.md,
  STATUS.md.
- **`SECURITY.md`** dependency-pinning claim refined: most direct deps
  are exact-pinned; vendor SDKs (cognee, langmem, memori) and a few
  large libs (numpy, scikit-learn) use bounded ranges to track
  fast-moving releases.
- **`SECURITY.md`** rate-limiting claim sharpened; AI prompt-injection
  caveat added for untrusted corpora.
- **`scripts/ship.sh`** reads version from `pyproject.toml` via
  `tomllib` with `pip show` fallback.
- **`pyproject.toml`** uses `dynamic = ["version"]` from
  `memory_arena/__init__.py` (single source of truth).
- **`.github/workflows/publish.yml`** OIDC Trusted Publishing on tag
  push with API-token fallback wired through workflow-level
  `env.PYPI_API_TOKEN`.
- **`MEM_ARENA_NEO4J_PASSWORD`** is required (no default fallback).
- **`memori.py`** raises `ImportError` with an install hint instead of
  subprocess-pip-installing `psycopg` at runtime.
- **Cognee extras** pin `s3fs>=2024.0.0` and `fsspec>=2024.0.0`.

### Removed

- **Letta** strategy deferred to vendor-pins; the slow per-step
  context loop made benchmark runs costly without a way to budget
  them.
- **ELO Arena** route hidden from the nav and redirected to
  `/benchmark` until the real implementation lands in v0.2.


## [0.1.4] - 2026-04-30 - Karpathy's LLM Wiki (17th strategy)

### Added
- **karpathy_llm_wiki** strategy implementing [Karpathy's three-layer pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Raw chat sessions stay immutable; the LLM maintains a separate wiki of markdown entity pages cross-linked via `[[wikilinks]]` and citing sources via `[session=<id>]` markers. Three workflows are all implemented:
  - **Ingest**, one LLM call per session emits a JSON array of `create` / `append` page ops.
  - **Query**, two-stage recall: Haiku selects pages from `index.md` + recent log, Sonnet synthesizes the answer using only the loaded pages.
  - **Lint**, every 10 ingested sessions, scan the 30 largest pages and emit `rewrite` / `merge` ops to dedupe entities and resolve contradictions when the user changes their mind.
- New `KarpathyLlmWikiStrategy` registered as the 17th strategy. Always-available (no vendor SDK).

### Real benchmark number

| Strategy            | Accuracy | Recall@k | Latency  | Cost     | What it does, in plain English                                                                |
| ------------------- | -------- | -------- | -------- | -------- | --------------------------------------------------------------------------------------------- |
| karpathy_llm_wiki   | 26.13%   | 57.81%   |  3953ms  | $1.270   | LLM maintains a wiki of entity pages with cross-links and citations. Periodic lint pass merges duplicates and resolves contradictions. |

Lands 8th of 17 between bm25 (27.13%) and full_context (25.50%). Strong on multi_session_reasoning (51%) where cross-linked pages help; weaker on information_extraction (21%) where naive_vector still wins. v0.2 ingest-prompt tuning should close that gap.

### Frontend

- `web/lib/api.ts` STRATEGIES extended to 17 (added karpathy_llm_wiki with its label, color, and description).
- Home page heading now reads "The 17 strategies"; hero copy and "How it works" rewritten to mention Karpathy's wiki.
- Layout meta description and arena leaderboard updated to 17.

## [0.1.3] - 2026-04-30 - 16 strategies, redesigned benchmark dashboard

### Added
- Nine new strategies, all real-benchmarked end-to-end on LongMemEval-S:
  - `bm25`, pure-Python lexical baseline using rank-bm25.
  - `hybrid_rrf`, Reciprocal Rank Fusion of vector + BM25 rankings (k=60).
  - `hyde`, Hypothetical Document Embeddings: LLM writes a plausible answer first, embeds that for retrieval.
  - `persona_profile`, builds a JSON profile of the user via Haiku once, then stuffs it as system context for every recall.
  - `reflection`, generative-agents pattern: writes synthetic LLM-authored summaries every N sessions and indexes them alongside raw turns.
  - `raptor`, hierarchical k-means clustering with LLM cluster summaries (Sarthi et al. 2024).
  - `cognee`, open-source knowledge-graph memory with `add` / `cognify` / `search(GRAPH_COMPLETION)`.
  - `langmem`, LangGraph InMemoryStore with `create_memory_store_manager`.
  - `memori`, SQL-native agent memory using Memori 3.x's `enqueue` / `wait` augmentation pipeline.
- Benchmark dashboard is now sortable by accuracy / recall / latency / cost / name. Default sort is accuracy descending. Adds a Latency column.
- "Memory-specific axes" section shows update precision / temporal correctness / abstention F1 with the per-category question count in parentheses, so a strategy that did 0 abstention questions renders "-" rather than a misleading default.

### Changed
- Per-category metrics (`update_precision`, `temporal_correctness`, `abstention_f1`) now return `null` from the benchmark runner when no question of that category was evaluated, instead of defaulting to 1.0. The dashboard renders "-" for null. Also adds `update_n` / `temporal_n` / `abstention_n` counts and an `accuracy_by_category` breakdown to every result JSON.
- Cognee strategy force-sets `LLM_API_KEY` via `cognee.config.set_llm_api_key(...)` plus pins embedding dimensions to 1536 (the max for `text-embedding-3-small`; cognee defaults to 3072).
- Memori strategy uses Memori 3.x's `Memori.augmentation.enqueue(AugmentationInput(...))` and `recall(query, limit=k)` (the 2.x `process_user_message` / `top_k=` APIs were removed).
- Reflection / persona_profile / hyde each prefix their inner ChromaDB collection with the strategy name to avoid collection collisions when running concurrently with `naive_vector`.
- Runner's end-of-run print line tolerates `None` for `abstention_f1` (renders "-") so strategies with zero abstention questions don't crash the runner.

### Real benchmark numbers (all 16 strategies, LongMemEval-S smoke subset)

| Strategy        | Accuracy | Recall@k | Latency  | Cost     | What it does, in plain English                                                                |
| --------------- | -------- | -------- | -------- | -------- | --------------------------------------------------------------------------------------------- |
| reflection      | 44.12%   | 87.50%   |  7579ms  | $0.488   | Periodically writes journal-style summaries of recent chats, searches both summaries and raw turns. |
| persona_profile | 43.38%   | 89.06%   |  6841ms  | $0.200   | Builds a one-page bio of the user up front, pastes it into every answer.                       |
| naive_vector    | 39.63%   | 89.06%   |  3548ms  | $0.087   | Stores every message as a meaning fingerprint, pulls the closest matches by similarity.        |
| raptor          | 36.40%   | 88.33%   |  3461ms  | $0.107   | Clusters similar messages into a hierarchy and summarizes each level, like a table of contents. |
| hybrid_rrf      | 35.13%   | 80.73%   | 16737ms  | $0.711   | Runs meaning-search and keyword-search side by side, blends the rankings.                      |
| hyde            | 31.44%   | 76.88%   | 11815ms  | $0.182   | Guesses an answer first, then searches for messages that look like that guess.                 |
| bm25            | 27.13%   | 79.17%   | 10499ms  | $0.090   | Old-school keyword search, what Google did before vectors.                                    |
| full_context    | 25.50%   |  8.33%   |  8551ms  | $5.263   | Pastes the entire chat history into every prompt. Expensive; forgets nothing until overflow.   |
| letta           | 19.00%   |  6.25%   | 13049ms  | $0.000   | Agent that edits its own scratchpad of facts after every user message.                         |
| langmem         | 18.13%   |  0.00%   |  2950ms  | $0.044   | LangChain's memory store: extracts facts as they happen, recalls by similarity.                |
| graphiti        | 15.75%   |  0.00%   |  3745ms  | $0.032   | Time-aware knowledge graph: who said what, about whom, and when.                               |
| recency_window  |  5.75%   |  6.25%   |  5284ms  | $0.318   | Only remembers the last 20 messages, the "chatbot with no memory" baseline.                   |
| mem0g           |  2.13%   |  0.00%   |  3177ms  | $0.011   | Mem0 plus a Neo4j graph linking facts to entities and to each other.                           |
| mem0            |  1.38%   |  0.00%   |  2558ms  | $0.010   | Extracts standalone facts ("user lives in Tokyo") from chats and stores them.                  |
| cognee          |  1.00%   |  0.00%   |  2736ms  | $0.014   | Builds a knowledge graph from the chat, answers via the graph as context.                      |
| memori          |  0.88%   |  0.00%   |  2214ms  | $0.013   | Stores extracted facts in Postgres for SQL-native recall.                                      |

### Caveats
- `memori` runs but Memori 3.x routes its augmentation runtime through a cloud quota service that 429s anonymous IPs after a few requests. Set `MEMORI_API_KEY` for full throughput. Without it the benchmark completes but augmentation is disabled and accuracy drops near zero.
- `full_context` hit the cost cap mid-run (12/16 questions evaluated).

## [0.1.2] - 2026-04-29 - All 7 strategies real-benchmarked end-to-end

### Added
- Mem0g, Letta, and Graphiti now run end-to-end on the LongMemEval-S subset and have real result JSON in `results/`. The dashboard renders all 7 rows.
- `docker/postgres-init.sql` enables the pgvector extension on first DB initialization so Letta migrates cleanly.
- `MEM_ARENA_POSTGRES_HOST_PORT` (default 5433) and `MEM_ARENA_LETTA_HOST_PORT` (default 8283) env knobs let Memory Arena coexist with other local services.

### Changed
- Graphiti strategy now chunks each session into 2-turn episodes and instantiates the underlying OpenAIClient with `model=gpt-4o, max_tokens=16384` so entity extraction does not hit the default 8192-token output cap.
- Letta strategy uses the `api_key=` keyword (the SDK dropped `token=`) and the `openai/gpt-4o-mini` model with `openai/text-embedding-3-small` embedding (the cloud-only `letta/letta-free` placeholder returns HTTP 401 on OSS deployments).
- Postgres host port moved from 5432 to 5433 to avoid the common host-Postgres collision; the docker network still uses `postgres:5432` internally so Letta connects without changes.
- Benchmark runner is robust to partial failures: strategies that crash in setup write a result JSON with default-zero metrics so the dashboard renders all 7 rows uniformly.

### Real benchmark numbers (all 7 strategies, vendor defaults)

| Strategy        | Accuracy | Recall@k | Cost     |
| --------------- | -------- | -------- | -------- |
| naive_vector    | 39.63%   | 89.06%   | $0.087   |
| full_context    | 25.50%   |  8.33%   | $5.263   |
| letta           | 19.00%   |  6.25%   | $0.000   |
| graphiti        | 15.75%   |  0.00%   | $0.032   |
| recency_window  |  5.75%   |  6.25%   | $0.318   |
| mem0g           |  2.13%   |  0.00%   | $0.011   |
| mem0            |  1.38%   |  0.00%   | $0.010   |

### Documented
- `docs/vendor-pins.md` updated with: graphiti chunked-episode + 16K cap, Letta SDK `token` -> `api_key` rename, `letta/letta-free` is cloud-only, `mem0g` needs `langchain-neo4j`, pgvector init script, host Postgres collision workaround.

## [0.1.1] - 2026-04-29 - Real benchmark + demo assets

### Added
- Real benchmark execution against LongMemEval-S subset (16 questions, 82 sessions). Per-strategy result JSON committed in `results/longmemeval-s_*.json`.
- Real screenshots in `docs/screenshot-{home,benchmark,recall-lab,arena}.png` captured via agent-browser against the live FastAPI dashboard.
- Real terminal demo GIF in `docs/demo.gif` (212KB, 29s) recorded with vhs. Shows actual `memory-arena health`, ingested-corpus stats, and per-strategy summary numbers from the result JSON.
- New `/api/recall-records/{corpus}/{strategy}` endpoint for the Recall Lab dashboard page.
- `docs/demo_summary.py` helper that prints headline numbers from a strategy's result JSON.

### Changed
- `LongMemEvalLoader` coerces upstream integer answers to strings (Pydantic v2 strictness).
- Anthropic provider auto-omits `temperature` for claude-4-6+ models. Newer Anthropic models reject the parameter.
- `/api/benchmark/{corpus}` returns a flat row projection the dashboard expects (strategy, accuracy, recall@k, costs, latency).
- The synthetic smoke YAML corpus moved to `datasets/longmemeval-s/questions/smoke_synthetic/` so the runner can fall through to the real `processed/questions.jsonl` produced by `ingest-sessions`.

### Documented
- `docs/vendor-pins.md` records two known issues:
  - graphiti-core 0.13.0 hits OpenAI's 8192-token output cap during entity extraction on full LongMemEval haystack episodes (each ~6-8 long turns).
  - claude-sonnet-4-6 / opus-4-7 reject the temperature parameter; the provider now skips it for these IDs.

### Real benchmark numbers (run_id 39f530f5 / 8fe157a7)

| Strategy        | Accuracy | Recall@k | Cost    | Notes                                              |
| --------------- | -------- | -------- | ------- | -------------------------------------------------- |
| naive_vector    | 39.63%   | 89.06%   | $0.087  | Best baseline. ChromaDB + text-embedding-3-large.  |
| full_context    | 25.50%   | 8.33%    | $5.263  | Hit cost cap mid-run (12/16 questions evaluated).  |
| recency_window  |  5.75%   |  6.25%   | $0.318  | Last 20 turns. Limited recall as expected.         |
| mem0            |  1.38%   |  0.00%   | $0.010  | Vendor default config. v0.2 will add tuned mode.   |

Tests: 267 pass. Ruff: clean. Format: clean.

## [0.1.0] - 2026-04-29 - Initial scaffold

### Added
- Memory Arena scaffolded from the kb-arena codebase as a structural template.
- 7 memory strategies subclassing a fresh `MemoryStrategy` ABC with `setup / ingest_session / recall / teardown` lifecycle:
  - `full_context`, `recency_window`, `naive_vector` (no extra dependencies)
  - `mem0`, `mem0g`, `letta`, `graphiti` (vendor SDKs in optional-extras; degrade gracefully when missing)
- 7-axis evaluator: structural, sources, judge, memo, plus three memory-specific axes (temporal correctness, update precision, abstention F1).
- `LongMemEvalLoader` for the upstream JSON corpus from HuggingFace.
- Session-level and turn-level recall metrics: `Recall@k`, `Hit@k`, `Precision@k`, `MRR`, `nDCG@k`.
- Benchmark runner with run-id namespace isolation, sequential per-strategy ingest, parallel across strategies, cost cap.
- Recall Lab - retrieval-only loop, ~10x cheaper than the full benchmark.
- Typer CLI: `init-corpus`, `download-longmemeval`, `ingest-sessions`, `build-memory`, `benchmark`, `recall-lab`, `report`, `serve`, `arena`, `demo`, `health`.
- FastAPI dashboard server mounting the bundled Next.js static export.
- Next.js 14 + Tailwind dashboard with Home, Benchmark, Recall Lab, ELO Arena pages.
- Docker compose with Neo4j (always-on), Postgres-pgvector (always-on), Letta (profile=letta), api+web (profile=full).
- `cypher/memory_schema.cypher` - universal memory graph schema (User / Session / Turn / Fact / Entity + temporal edges).
- 267 tests covering schema, loaders, strategies, evaluator, recall metrics, runner, recall_lab, chatbot API, CLI, settings, exceptions, and the synthetic smoke YAML files.
