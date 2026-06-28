# Memory Arena: FAQ

The objections that come up most. Read this before opening an issue;
the ones that haven't been thought about will be the rare ones.

## On methodology

### Why 16 questions? That's tiny.

Because the smoke subset is the v0.1 unit, fast enough to re-run on every
PR, cheap enough that contributors can verify locally for under $5. The
full LongMemEval-S is 500 questions; running it once across all 19
strategies x 3 seeds costs ~$370 in API spend. v0.2 ships those numbers.

The smoke set is for the "did this PR break anything" loop. The README
table is honest about what 16 questions can statistically support: tier
1 (mem0 / mem0g / persona_profile / naive_vector / reflection) is a tie
within noise; tier 2 (raptor / hybrid_rrf / hyde) is below tier 1 but
within noise of each other; the gap to bm25 / recency_window is real
because the CIs don't overlap.

If you want to claim "X beats Y by 2 points" with confidence on this
corpus, you need the v0.2 numbers. Don't take the smoke ranks literally
inside a tier.

### Why only Opus 4.7 as judge? Anthropic-on-Anthropic seems sus.

It is sus. The right cross-judge sanity check is to re-grade every
strategy x every question under a different provider's judge and look
at Pearson correlation on per-question grades (not Spearman on
strategy-level ranks, which collapses N to the number of strategies).

Done at the rank level: re-grading 19 of the 20 strategies (sqr's results
lack a seed suffix, so the seed-based re-grade skips it) x 16 questions x
seeds under GPT-4o gives **Spearman = +0.967** on the strategy ranks
(`results/cross_judge_report.json`). GPT-4o grades ~10-25pp higher in
absolute terms but agrees on the order, and independently ranks every
vendor SDK below the pure-Python tier. The stronger per-question Pearson
check (above) is a v0.2 refinement; treat the published numbers as
Opus-judged with a strong rank-level GPT-4o sanity check. See Zheng et al.
2024 ("Judging LLM-as-a-Judge", NeurIPS) for the judge-bias framework.

Reproduction once the script targets all strategies:
`python scripts/cross_judge.py --strategies all --judge gpt-4o`.

v0.2 will add `gemini-1.5-pro` as a third judge and report the
three-way correlation across all strategies.

### Why didn't you tune the vendors?

Because "what does the vendor produce when I run their documented install"
is the question 90% of users actually have. Vendor-tuned configs are a
follow-up, not the leadoff. The README is explicit:

> The table answers the question "what does memory-arena measure when
> I run the documented install command?", not "what's the maximum each
> vendor SDK can achieve."

PRs that ship a tuned config, with a link to the vendor doc that
recommends those defaults, are the engagement loop. See
[`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md).
Every accepted PR re-runs the bench against the new config and updates
the table.

### Why is `memori` at 1.4%? Is the benchmark broken?

Memori 3.x routes its augmentation runtime through a cloud quota service
that 429s anonymous IPs after a few requests. We ran the bench without a
`MEMORI_API_KEY` set, so cloud augmentation was throttled to ~zero and
the SQL-only path scored barely above floor. Set the key and rerun for
Memori's intended throughput. Row is footnoted ‖ in the table.

### Why doesn't the leaderboard show p-values for pairwise differences?

Because the displayed CIs are seed-level (3-seed bootstrap), and
seed-level CIs don't underwrite pairwise tests cleanly: they conflate
judge-variance with retrieval-accuracy variance, and overlapping CIs
don't directly imply non-significance.

The right test for "strategy X beats strategy Y on these N questions"
is a paired bootstrap over per-question accuracy differences, then
check whether the 95% CI of the mean delta excludes 0. The pairwise
heatmap in `docs/pairwise.png` uses the seed-level approach as a
qualitative proxy; it's the right shape but not the right statistic.

A proper paired-bootstrap implementation is queued for v0.1.7 (no API
spend; aggregation-script change). When it lands, the README table
will gain a "vs naive_vector" column with p-values. Until then: trust
the qualitative clusters (tier 1 / tier 2 / floor); don't trust 2-pp
gaps within a cluster.

### Why is the cost cell different in the table vs the hero chart?

The table marks vendor SDK costs with ‡ (e.g. `mem0` at `$0.048‡`); the
chart plots the same number without the ‡. ‡ means: this is the cost
memory-arena's LLMClient saw, but the vendor SDK runs additional internal
LLM calls (mem0 extraction, graphiti entity extraction, langmem fact
extraction, cognee cognify) that don't go through our accounting. True
cost is "what the table shows + the unmeasured vendor-internal piece."
For mem0 the unmeasured piece is gpt-4o-mini calls (the Mem0 default
extraction LLM); for cognee/langmem it's the same.

We'll add proper interception of vendor-internal LLM calls in v0.2 so
those costs are countable.

## On the data

### Why are mem0 and mem0g so close to naive_vector?

Because at 16 questions across 4 categories (4 questions per category),
the CI half-width swallows ±5 pp differences. mem0g is at 51.4%, mem0
at 50.9% +/-5.5, naive_vector at 49.2% +/-3.7 are statistically
indistinguishable, and persona_profile (49.7%) and reflection (47.4%)
sit inside that same tie.

The substantive finding is "the cost of choosing wrong is small." If
you're shipping an agent today, picking between mem0 and naive_vector
buys you negligible accuracy delta on this corpus; pick on infra fit
(do you want to manage a vendor account or a ChromaDB volume?), on
cost ($0.05/run for mem0 vendor-internal-extras vs $0.09/run for
naive_vector all-in), and on tail behaviour you can probe in the recall
lab.

### Why does graphiti lose to naive_vector here?

Graph structure pays off most for questions that span multiple sessions
and require multi-hop reasoning ("what did the user do at company X
before they joined company Y?"). The smoke set has 4 multi-session
questions. With more multi-hop questions in the v0.2 full corpus,
graphiti's relative position should improve.

### Why does the Karpathy LLM Wiki score so low (~22%)?

The wiki pattern is information-dense per page but the LLM has to
produce a page selection step that's not as battle-tested as cosine
similarity in vector search. The smoke set is heavy on single-fact
extraction where the wiki's strength (cross-linked synthesis) doesn't
fire. It scored 57.8% Recall@5 (better than half the table), so the
right pages are being retrieved; the failure is in answering from
them.

This is one of the strategies most likely to climb on the v0.2 corpus,
which has more multi-session synthesis questions where cross-linked
pages would carry their weight.

### Why is `full_context` at 29.5% when it sees everything?

The LLM judge prefers concise, confident answers. Stuffing the whole
chat history into the prompt produces longer, hedged outputs that the
judge marks down. Long-context isn't a substitute for retrieval, even
when the model can hold it. This was the most surprising finding when
we built the eval.

### Why does `recency_window` score 5%?

Because most LongMemEval questions reference facts from sessions far
outside the last 20 messages. Recency_window is a floor baseline; its
value is showing how much retrieval matters. If a strategy doesn't beat
recency_window, it's broken.

### What's the difference between agent memory and RAG?

Same retrieval primitive, different shape of corpus and different
write path.

RAG retrieves over a (mostly) static corpus: your docs, a wiki, code,
papers. The corpus is ingested once, doesn't update with each user
interaction, and the user query is independent of the corpus.

Agent memory retrieves over the agent's *own conversation history with
this user*. The corpus is incremental (every turn adds new content),
mutable (the user's preferences change: "I moved from Chicago"
followed three months later by "I moved to Brooklyn"), and per-user
(Alice's memory must not leak into Bob's). The write path is the
interesting axis: vendors like mem0 actively rewrite stored facts when
newer ones contradict; pure-Python `naive_vector` just accumulates;
graph approaches like graphiti add temporal edges.

The retrieval primitives are mostly shared (vector search, BM25, graphs
all work for both), which is why pure-Python baselines sit inside tier 1
of this benchmark. memory-arena scores the *write path* through the
`knowledge_update` category specifically because that's where memory and
RAG diverge.

## On the project

### How do I know your numbers are real?

Every result JSON is stamped with the commit SHA, installed package
versions, model IDs, host info, and seed under `metadata`. Re-run
`scripts/aggregate_bootstrap.py` and `scripts/render_readme.py` from a
clean checkout: same numbers should fall out of the same per-seed JSONs.

If you want to verify from scratch (no bundled snapshot): see "Verify
our numbers in 5 minutes" in the README.

### How is this different from RAGAS / Promptfoo / DeepEval / TruLens?

Different cohort. RAGAS, Promptfoo, DeepEval, and TruLens are *eval
frameworks*: toolkits you reach for when you want to build evals for
your own RAG or LLM application against your own questions. They help
you measure *your* pipeline.

memory-arena is a *benchmark*: a fixed corpus, a fixed evaluator, a
fixed `setup -> ingest_session -> recall -> teardown` lifecycle, applied
across 19 different memory architectures. The unit of work is
"compare strategy A vs strategy B on the same conversations under the
same configs."

You'd reach for RAGAS or DeepEval to evaluate *your* RAG pipeline
against *your* questions. You'd reach for memory-arena to decide
*which* memory architecture to use in the first place. The input is
"I'm building an agent that needs to remember things; should I bring
in mem0 or roll vector search myself?"

There's overlap on technique (the LLM-as-judge pattern is now
standard everywhere, and the 7-axis scoring borrows from
G-Eval / RAGAS norms), but the deliverables differ. RAGAS doesn't ship
a leaderboard of memory vendors; we don't ship a framework you embed
in your CI.

### Why isn't `letta` in the table?

Letta's docker container sends 100k+ tokens of context per turn through
its gpt-4o-mini agent loop, which makes a 16-question / 82-session bench
take 2+ hours of wall time and consume vendor-internal tokens we can't
measure. Returning when we have a streaming-context Letta config,
documented in `docs/vendor-pins.md`.

### Why isn't [Zep / Pinecone / Weaviate / GraphRAG / LightRAG / OpenAI Memory] in the table?

None of these are bad. The omissions are about architectural
distinctness, methodology constraints, or both.

- **Zep is already in.** `graphiti` is Zep's open-source temporal-graph
  framework ([getzep/graphiti](https://github.com/getzep/graphiti)); the
  cloud product is a managed wrapper on top. A separate `zep_cloud`
  strategy would be Graphiti with a different backend, not a new
  architectural data point.
- **Pinecone, Weaviate, Qdrant, Chroma Cloud are redundant with
  `naive_vector`.** Same algorithm (chunk -> embed -> cosine top-k),
  different storage. The benchmark's question on that row is "is vector
  retrieval competitive?" and `naive_vector` already answers it. Adding
  more vector DBs multiplies rows without adding a new point unless the
  retrieval *algorithm* also varies. Pinecone's reranking and Weaviate's
  hybrid (sparse+dense) search are different algorithms and *do* deserve
  their own slots; that's a v0.2 entry.
- **GraphRAG (Microsoft) and LightRAG sit in the same bucket as
  `graphiti`/`cognee`/`mem0g`.** Three knowledge-graph vendors are
  already in the table and they cluster at 19-23%. Two more would
  tighten the cluster, not change the headline. Their ingest pipelines
  also aren't chat-session-shaped (GraphRAG ingests documents, not
  conversations), so a wrapper would do nontrivial massaging.
- **OpenAI Memory (Assistants API) and the Anthropic memory tool are
  provider-locked.** Including them means the generator becomes
  GPT-4o-or-Claude-with-tools rather than the constant Sonnet 4.6 every
  other strategy holds. That breaks the methodology constraint that
  makes the comparison clean. They belong in a separate
  "provider-managed memory" track with the generator-control caveat
  called out, not on the headline table.

If you want any of them measured anyway, the
[`MemoryStrategy`](../memory_arena/strategies/base.py) ABC is 4 methods,
and the smallest existing strategy (`bm25` or `recency_window`) is ~80
lines. PRs welcome.

### Is this an academic paper?

No. It's a tool for choosing a memory store. The paper-shaped thing
would be (a) the v0.2 full-corpus run with cross-judges and rank
correlations, plus (b) a methodology section justifying the 7-axis
evaluator weights. That's a v0.3 ambition.

### Are the vendors going to be mad?

Some will be. The fix is the same as the fix for "the LongMemEval
authors might be mad": stay accurate, run the methodology in the open,
publish the recipe, and merge their PR when they ship a tuned config
that beats the default. The PR template is structured to make that
loop easy.
