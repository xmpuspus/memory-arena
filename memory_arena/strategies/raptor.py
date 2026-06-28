"""RAPTOR-style hierarchical summary tree (Sarthi et al. 2024).

At ingest end, all turns become L0 nodes. Cluster L0 nodes by embedding
similarity; each cluster gets an LLM summary -> L1 node. Cluster L1, etc.
At recall, query against ALL levels at once - leaves give precise quotes,
roots give bird's-eye context.

Implementation differences from the paper:
- Cluster with k-means on embeddings (paper uses GMM); shouldn't matter on
  small corpora.
- Stop when the level has <= 4 nodes or 4 levels deep, whichever first.
- Use Anthropic Haiku for the summaries; gpt-4o-mini would also work.
"""

from __future__ import annotations

import time

import numpy as np

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

_SUMMARY_SYSTEM = (
    "You are summarizing a cluster of related chat snippets into a single short paragraph. "
    "Capture the durable facts and themes, drop redundant phrasing. "
    "Reply with only the summary, two to four sentences."
)

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Memories include both raw turns and higher-level summaries. Use only the retrieved content. "
    "Cite sessions by their id in square brackets. If the memories do not contain the answer, "
    'say "I do not have that information."'
)

_MAX_LEVELS = 4
_BRANCH = 4
_RECALL_PER_LEVEL = 4


class _Node:
    __slots__ = ("id", "level", "text", "session_id", "turn_id", "embedding", "children")

    def __init__(
        self,
        id: str,
        level: int,
        text: str,
        session_id: str,
        turn_id: str | None = None,
    ):
        self.id = id
        self.level = level
        self.text = text
        self.session_id = session_id
        self.turn_id = turn_id
        self.embedding: list[float] | None = None
        self.children: list[str] = []


class RaptorStrategy(MemoryStrategy):
    name = "raptor"
    # Each level retrieves _RECALL_PER_LEVEL items; after concatenation we
    # rank into a candidate pool of `top_k * CANDIDATE_POOL_MULTIPLIER` and
    # then return exactly top_k. Bumping this widens the rerank window
    # without changing what the generator sees.
    CANDIDATE_POOL_MULTIPLIER = 2

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, _Node] = {}
        self._levels: dict[int, list[str]] = {}
        self._sessions_buffer: list[Session] = []
        self._embedder = None
        self._llm: LLMClient | None = None
        self._tree_built = False

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._nodes = {}
        self._levels = {0: []}
        self._sessions_buffer = []
        self._tree_built = False
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        for turn in session.turns:
            nid = f"L0_{turn.id}"
            n = _Node(
                id=nid,
                level=0,
                text=f"{turn.role}: {turn.content}",
                session_id=session.id,
                turn_id=turn.id,
            )
            self._nodes[nid] = n
            self._levels[0].append(nid)
        self._sessions_buffer.append(session)
        self._tree_built = False
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            facts_extracted=len(session.turns),
        )

    def _get_embedder(self):
        if self._embedder is None:
            from memory_arena.strategies.embeddings import OpenAIEmbedding

            self._embedder = OpenAIEmbedding()
        return self._embedder

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        ef = self._get_embedder()
        return ef(texts)

    async def _build_tree(self) -> tuple[int, float]:
        from sklearn.cluster import KMeans

        if self._llm is None:
            self._llm = LLMClient()
        l0_ids = self._levels[0]
        if not l0_ids:
            return 0, 0.0
        l0_texts = [self._nodes[i].text for i in l0_ids]
        embs = self._embed_batch(l0_texts)
        for nid, emb in zip(l0_ids, embs, strict=False):
            self._nodes[nid].embedding = list(emb)

        cost = 0.0
        new_node_count = 0
        cur_level = 0
        while cur_level < _MAX_LEVELS - 1:
            cur_ids = self._levels[cur_level]
            if len(cur_ids) <= _BRANCH:
                break
            n_clusters = max(2, len(cur_ids) // _BRANCH)
            X = np.array([self._nodes[i].embedding for i in cur_ids], dtype=np.float32)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
            labels = kmeans.labels_

            next_level = cur_level + 1
            self._levels.setdefault(next_level, [])
            cluster_to_ids: dict[int, list[str]] = {}
            for nid, label in zip(cur_ids, labels, strict=False):
                cluster_to_ids.setdefault(int(label), []).append(nid)

            for cluster_id, member_ids in cluster_to_ids.items():
                cluster_texts = [self._nodes[i].text for i in member_ids]
                joined = "\n---\n".join(cluster_texts)[:8000]
                resp = await self._llm.generate(
                    query="", context=joined, system_prompt=_SUMMARY_SYSTEM
                )
                cost += resp.cost_usd
                summary_text = resp.text.strip()
                summary_id = f"L{next_level}_{self.run_id}_{cluster_id}"
                summary_node = _Node(
                    id=summary_id,
                    level=next_level,
                    text=summary_text,
                    session_id="(summary)",
                )
                summary_node.children = list(member_ids)
                summary_node.embedding = self._embed_batch([summary_text])[0]
                self._nodes[summary_id] = summary_node
                self._levels[next_level].append(summary_id)
                new_node_count += 1
            cur_level = next_level
        return new_node_count, cost

    async def recall(self, query: str, top_k: int = 5) -> RecallResult:
        if not self._tree_built:
            await self._build_tree()
            self._tree_built = True
        start = self._start_timer()
        retrieval_start = time.perf_counter()
        if not self._nodes:
            return RecallResult(answer="I do not have that information.", strategy=self.name)
        q_emb = np.array(self._embed_batch([query])[0], dtype=np.float32)

        # For each level, retrieve top-N
        memories: list[dict] = []
        session_ids: list[str] = []
        turn_ids: list[str] = []
        for level, level_ids in sorted(self._levels.items()):
            if not level_ids:
                continue
            X = np.array([self._nodes[i].embedding for i in level_ids], dtype=np.float32)
            # cosine similarity
            qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            sims = (Xn @ qn).tolist()
            ranked = sorted(range(len(level_ids)), key=lambda i: sims[i], reverse=True)
            for i in ranked[:_RECALL_PER_LEVEL]:
                n = self._nodes[level_ids[i]]
                memories.append(
                    {
                        "id": n.id,
                        "level": level,
                        "score": float(sims[i]),
                        "session_id": n.session_id,
                        "turn_id": n.turn_id,
                        "content": n.text,
                    }
                )
                if n.session_id and n.session_id != "(summary)" and n.session_id not in session_ids:
                    session_ids.append(n.session_id)
                if n.turn_id and n.turn_id not in turn_ids:
                    turn_ids.append(n.turn_id)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Candidate pool: rank a 2x window then trim to top_k so the
        # generator sees exactly top_k items, matching the constant the
        # benchmark holds across strategies.
        candidates = sorted(memories, key=lambda m: m["score"], reverse=True)[
            : top_k * self.CANDIDATE_POOL_MULTIPLIER
        ]
        memories = candidates[:top_k]
        context = "\n\n---\n\n".join(
            f"[L{m['level']} session={m['session_id']} score={m['score']:.2f}] {m['content']}"
            for m in memories
        )

        if self._llm is None:
            self._llm = LLMClient()
        gen_start = time.perf_counter()
        resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=resp.text,
            supporting_session_ids=session_ids,
            supporting_turn_ids=turn_ids,
            retrieved_memories=memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        self._nodes = {}
        self._levels = {0: []}
        self._sessions_buffer = []
        self._tree_built = False
        self._llm = None
        self._embedder = None
