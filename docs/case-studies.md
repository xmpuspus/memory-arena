# Memory Arena: Failure-Mode Case Studies

Four questions chosen to span the design space. Each shows the exact
question, the ground-truth answer, and what every relevant strategy
actually returned, not summarized, not paraphrased, copied verbatim
from `results/longmemeval-s_<strategy>_seed0.json`. The interesting
finding in this benchmark isn't the leaderboard rank, it's the
*why*.

To reproduce any row below: `memory-arena recall-lab` with the
question_id shown, or read `results/longmemeval-s_<strategy>_seed0.json`
and grep for the question_id.

---

## Case 1: `6aeb4375` "How many Korean restaurants have I tried in my city?"

**Category:** `knowledge_update` · **Ground truth:** `four`

This is the question with the largest pure-Python-vs-vendor gap in the
corpus (+68 pp). It's also the case that most directly explains *why*
the gap exists.

| Strategy | Acc | R@5 | Answer (verbatim, truncated) |
|----------|----:|----:|------------------------------|
| mem0g | 0.04 | 1.00 | "I do not have any information about you trying Korean restaurants. The memories mention you *thinking about* serving Korean BBQ beef at home…" |
| mem0 | 0.04 | 1.00 | "There is no information about you having tried any Korean restaurants. The memories only reference your interest in **making Korean-style BBQ beef**…" |
| persona_profile | 0.80 | 1.00 | "You mentioned trying **four** different Korean restaurants…" |
| naive_vector | 0.80 | 1.00 | "You mentioned trying **four** different Korean restaurants…" |
| reflection | 0.80 | 1.00 | "You mentioned trying **four** different Korean restaurants…" |
| raptor | 0.80 | 1.00 | "You have tried **four** different Korean restaurants…" |
| bm25 | 0.80 | 0.50 | "You have tried **four** different Korean restaurants…" |
| langmem | 0.04 | 0.00 | "I do not have that information. The memories only indicate your interest in Korean BBQ…" |
| graphiti | 0.00 | 0.50 | "Tips about finding Korean restaurants…" |
| cognee | 0.04 | 0.00 | "Only contain details about basmati rice brands and Indian food…" |
| memori | 0.00 | 0.00 | "I don't have any information…" |

**What's happening.** Every vendor SDK that does write-time fact
extraction decided the salient fact in the source session was *"user
is interested in Korean BBQ recipes."* That extracted fact then
crowded out the actual fact in the raw conversation: *"I've tried four
different Korean restaurants in the city."* The pure-Python retrievers
don't extract, they store raw turns and search them at query time, so
the count survives. **Recall@5 = 1.00 for mem0/mem0g** confirms they
retrieved the right *session*, they just don't have the count
anywhere in the structured memory they built from it.

**The reading.** Aggressive write-time extraction is information loss.
If the questions you care about touch *attributes* of an entity that
the extractor didn't extract, you lose. Read-time retrieval over raw
text doesn't have this failure mode at the cost of more tokens per
recall. There's no free lunch.

---

## Case 2: `830ce83f` "Where did Rachel move to after her recent relocation?"

**Category:** `knowledge_update` · **Ground truth:** `the suburbs`

Two sessions in the corpus mention Rachel's location. The earlier one
says she moved to Chicago. The later one says she moved back to the
suburbs. The right answer is the *latest* fact, not the first.

| Strategy | Acc | R@5 | Answer (verbatim, truncated) |
|----------|----:|----:|------------------------------|
| mem0g | 0.76 | 1.00 | "Two potentially relevant pieces of information about a friend named Rachel…" |
| mem0 | 0.76 | 1.00 | "Rachel moved back to **the suburbs** … This represents an update from an earlier memory that had her moved to Chicago." |
| persona_profile | 0.04 | 1.00 | "Rachel moved to **Chicago**." |
| naive_vector | 0.32 | 1.00 | "Two different accounts: in one session she moved to Chicago. In another, she moved to the suburbs." |
| reflection | 0.68 | 1.00 | "Conflicting details. In one session: Chicago. In the other: the suburbs…" |
| raptor | 0.08 | 1.00 | "Rachel moved to **Chicago**." |
| bm25 | 0.00 | 0.50 | "No information about Rachel relocating." |
| langmem | 0.56 | 0.00 | "Two different accounts: Chicago / the suburbs." |
| graphiti | 0.08 | 0.50 | "Rachel moved to **Chicago**." |
| cognee | 0.80 | 0.00 | "Rachel moved to **the suburbs** after her recent relocation, having previously lived in the city." |
| memori | 0.04 | 0.00 | "I do not have that information." |

**What's happening.** This is the question vendor SDKs are *built for*.
Mem0 explicitly tracks memory updates ("this represents an update from
an earlier memory"). Cognee's graph supersession logic gets it right
even with Recall@5 = 0 (the right session wasn't returned but the
graph already knew the latest fact). naive_vector retrieves both
sessions but can't tell which is newer; raptor's hierarchical
clustering loses the temporal ordering entirely and confidently picks
the wrong one. reflection at least flags the conflict explicitly.

**The reading.** When the question is "what's the latest version of
fact X?", architectures that model time-of-write, fact stores with
update tracking, temporal graphs, pay back their write-time cost. The
vector-only retrievers can return the right sessions but don't know
which is current.

---

## Case 3: `e47becba` "What degree did I graduate with?"

**Category:** `information_extraction` · **Ground truth:** `Business Administration`

The boring case, single fact, mentioned once, no temporal structure.
The point of including it is to show what the chart looks like when
the question is *easy*.

| Strategy | Acc | R@5 | Answer (verbatim, truncated) |
|----------|----:|----:|------------------------------|
| mem0g | 0.80 | 1.00 | "Business Administration" |
| mem0 | 0.80 | 1.00 | "Business Administration" |
| naive_vector | 0.80 | 1.00 | "Business Administration" |
| reflection | 0.80 | 1.00 | "Business Administration" |
| raptor | 0.80 | 1.00 | "Business Administration" |
| bm25 | 0.80 | 1.00 | "Business Administration" |
| graphiti | 0.80 | 1.00 | "Business Administration" |
| persona_profile | 0.32 | 1.00 | "Your profile doesn't mention your educational background…" |
| langmem | 0.00 | 0.00 | "Memories do not mention anything about your degree…" |
| cognee | 0.00 | 0.00 | "No mention of your educational degree…" |
| memori | 0.00 | 0.00 | "No retrieved memories that mention your degree…" |
| recency_window | 0.04 | 0.00 | "Recent conversation turns do not contain…" |

**What's happening.** 7 of 12 strategies nail it identically. The
failures are diagnostic: `persona_profile` retrieved the right session
but its compressed bio omitted the education line, so the model
doesn't trust the retrieved snippet over the (silent) profile. langmem
/ cognee / memori never extracted "graduated with X" as a memorable
fact, same write-time-extraction failure as Case 1, on a different
attribute. recency_window is the floor: 20-message window doesn't
reach the session that mentioned it.

**The reading.** When retrieval works (high R@5) and the model has the
right session, single-fact extraction is solved across architectures.
The interesting failures are what the *write-time* layer chose not to
keep, not what the read-time retriever found.

---

## Case 4: `118b2229` "How long is my daily commute to work?"

**Category:** `information_extraction` · **Best score across all strategies: 0.08**

The benchmark's hardest question. Every strategy fails. The diagnostic
is *why* every strategy fails the same way.

| Strategy | Acc | R@5 | Answer (verbatim, truncated) |
|----------|----:|----:|------------------------------|
| mem0g | 0.04 | 1.00 | "I don't have specific information about the **duration** of your daily commute. I know you commute to Roppongi…" |
| mem0 | 0.08 | 1.00 | "I know you've been commuting to Roppongi for an English teaching job for about 4 months…" |
| naive_vector | 0.08 | 1.00 | "Only detail mentioned is that you commute to Roppongi for your English teaching job…" |
| raptor | 0.08 | 1.00 | "The only detail mentioned is that you commute to Roppongi…" |
| bm25 | 0.08 | 1.00 | "Memories mention that you commute daily to Roppongi … the duration is not specified." |
| graphiti | 0.04 | 0.00 | "I don't have specific information about the duration…" |
| persona_profile | 0.04 | 0.00 | "Mention of a daily commute to Roppongi … no specific details about the length…" |

(others omitted, same pattern.)

**What's happening.** Recall@5 is 1.00 for nearly every strategy -
the right session was retrieved. The reference answer "45 minutes
each way" is in the retrieved text. But the LLM reader can't find it,
or doesn't trust it, or the chunk it got didn't contain that exact
phrase.

This is a **reading** failure, not a **retrieval** failure, not a
**memory** failure. Every strategy here is being graded on the same
read step (Sonnet 4.6 generation), so they all hit the same ceiling.

**The reading.** When R@5 is 1.0 across the board and accuracy is
0.08 across the board, the architecture didn't fail, the reader did.
A larger generator model or a more aggressive chunking strategy would
move the floor here. This is one of the questions where a 2×2
generator-vs-judge robustness sweep (`scripts/robustness.py`) would
tell us how much of the gap is reader-shaped.

---

## What these four cases collectively say

1. **Write-time fact extraction is information loss.** If the question
   touches a property the extractor didn't extract, you lose. Vendors
   that do this (mem0, langmem, cognee, memori) win on the questions
   where their extractor happened to keep the right thing, lose
   spectacularly when it didn't.
2. **Update-aware architectures earn back their cost on
   `knowledge_update` questions.** Mem0 and Cognee shine on Case 2
   for exactly the reason they were built. Naive vector retrieves the
   right sessions but can't pick the latest fact.
3. **Single-fact retrieval is mostly solved.** Case 3 has 7-way
   agreement at 80% accuracy. Architectures don't differ much when the
   question matches what every retriever is good at.
4. **Some questions fail at the reader, not the retriever.** Case 4
   has perfect retrieval and 8% accuracy. The architecture isn't the
   bottleneck; the LLM read step is.

Together they map onto the taxonomy figure (`docs/taxonomy.png`):
read-time strategies bottom-left win Cases 1 and 3 because they don't
extract; graph/structured strategies top-right win Case 2 because they
*do* model time; nothing wins Case 4 because it's not a memory
problem.
