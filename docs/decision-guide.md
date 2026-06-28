# Memory Arena: Decision Guide

Practical guidance for choosing a memory architecture. Numbers come
from the LongMemEval-S smoke run (16 questions across 4 categories,
3 seeds where available; median 95% bootstrap CI half-width ~18 pp).
Read [`docs/case-studies.md`](case-studies.md) for the qualitative
reasoning behind each recommendation.

## Quick decision tree

```
Q: Does your agent need to track facts that change over time?
   (e.g., "what's the user's current job?", "where do they live now?")
│
├── YES, frequently     → mem0 / mem0g (vendor SDK with explicit update tracking)
│                         alt: cognee (graph supersession) for graph-shaped data
│
└── NO, mostly static facts
    │
    Q: Is "I don't want to manage vendor infra / API key rotation / billing" a hard preference?
    │
    ├── YES             → naive_vector (30 lines of ChromaDB; ~$0.09/run)
    │                     alt: bm25 if cost matters more than recall
    │
    └── NO, vendor is fine
        │
        Q: Does the question set need multi-hop reasoning across many sessions?
        │
        ├── YES         → mem0g (graph layer pays off) or graphiti
        │                 (mem0g actually wins on smoke; graphiti's
        │                 advantage is at scale that LongMemEval-S
        │                 doesn't yet exercise)
        │
        └── NO          → mem0 (top of leaderboard, simpler than mem0g)
```

## Use-case matrix

| You care about... | Pick | Avoid | Why |
|-------------------|------|-------|-----|
| **Lowest infra burden** | `naive_vector` | mem0g, graphiti, cognee | Just ChromaDB. No vendor account, no Neo4j. |
| **Lowest measurable cost** | `bm25` | full_context, karpathy_llm_wiki | $0.09/run, no embedding API. |
| **Top accuracy at any cost** | `mem0g` or `mem0` | recency_window, memori | 51.4 / 50.9% leads the table (tier-1 tie with naive_vector). |
| **Knowledge updates over time** | `mem0`, `cognee` | bm25, raptor | They model fact supersession explicitly (see Case 2 in case-studies). |
| **Single-fact recall** | `naive_vector`, `bm25` | persona_profile, memori | 7 of 12 strategies tie at 80% on Case 3; pick the cheapest. |
| **Lowest latency p50** | `memori`, `langmem` | hybrid_rrf, hyde | But beware accuracy floor, memori needs `MEMORI_API_KEY`. |
| **Most predictable answers** | `naive_vector` | hyde, raptor | Vector cosine is deterministic; HyDE / RAPTOR have an LLM in the retrieval path. |
| **Audit trail / explainability** | `karpathy_llm_wiki` | langmem, memori | Wiki is human-readable markdown with `[session=...]` citations. |
| **Long-context "stuff it all"** | `full_context` | (any retrieval) | Only if you have <30k tokens of history AND $5/run is fine. |

## What NOT to pick

| Don't use | When | Use instead |
|-----------|------|-------------|
| `recency_window` | Anything beyond a chat with <20 messages of state | Any retrieval-based strategy |
| `full_context` | History approaches token budget OR cost matters | `naive_vector` |
| `memori` (without `MEMORI_API_KEY`) | Production | Set the key, or use `mem0` |
| `hyde` | When latency matters | `naive_vector` (HyDE adds an LLM call to every recall) |
| `karpathy_llm_wiki` | Cost-sensitive workloads | $1.27/run is 14× naive_vector for ~half the accuracy |

## Calibration: the architecture isn't always the bottleneck

[Case 4](case-studies.md#case-4--118b2229-how-long-is-my-daily-commute-to-work)
shows what happens when every strategy fails at the same low score:
`Recall@5 = 1.0` across the board, accuracy = 0.08 across the board.
That's a **reading** failure (the LLM generator can't extract the
answer from the retrieved text), not a memory failure. If your
benchmark looks like Case 4 across many questions, no architecture
choice will help, the problem is the generator model or the chunking
strategy. Run [`scripts/cross_judge.py`](../scripts/cross_judge.py) to
see whether the issue is the judge, and the planned
[`scripts/robustness.py`](../scripts/robustness.py) to see whether
swapping the generator (Sonnet 4.6 → GPT-4o) closes the gap.

## When the table moves

The smoke set is 16 questions across 4 categories. The median 95%
bootstrap CI half-width is ~18 pp; the worst case (e.g. `raptor`
single-seed) is wider. The top 5 strategies are statistically tied
(see the pairwise heatmap in `docs/pairwise.png`). Don't read the
leaderboard ranks 1-5 as ordered, they aren't, on this corpus. The
substantive recommendation above is "tier 1 ~ tier 1; pick on infra
fit and cost." When v0.2 ships the full 500-question corpus the
within-tier ranks may separate; this guide will be updated against
those numbers.
