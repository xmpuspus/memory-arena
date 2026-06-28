"""HippoRAG 2: open-IE knowledge graph + personalized PageRank over passages.

Paper: https://arxiv.org/abs/2502.14802 (Gutierrez et al., ICML 2025).
Reference repo: https://github.com/OSU-NLP-Group/HippoRAG.

Inspired by hippocampal indexing theory: dense passage embeddings act as the
neocortex (semantic gist), while a sparse open-IE graph plus PPR plays the role
of the hippocampus (associative pattern completion across episodes).

Pipeline per session:
    1. Concatenate turns into a passage; embed the passage and store in Chroma.
    2. Run open-IE on the passage with Haiku, emitting (subject, predicate,
       object) triples as JSON.
    3. Add subject/object as nodes in a networkx graph, with predicate-labeled
       directed edges between them. Each entity node tracks the set of passages
       it appears in (`belongs_to_passage` weights).

Synonym edges (deferred to first recall, cached after):
    Embed every node phrase once. For pairs above SYNONYM_THRESHOLD cosine
    similarity, add an undirected `synonym` edge weighted by similarity.

Pipeline per recall:
    1. Embed the query. Pick top-K seed nodes by query-vs-node cosine.
    2. Run personalized PageRank over the graph with the seed nodes as
       personalization weights (alpha = PAGERANK_DAMPING).
    3. Score each passage by summing PPR mass on its entity nodes.
    4. Optionally LLM-rerank the top-CANDIDATE_POOL_SIZE candidates.
    5. Generate the final answer with Sonnet over the top top_k passages.

Deviations from the reference repo:
    - Single-passage-per-session granularity, not per-paragraph chunks.
      LongMemEval items are conversational, not document corpora.
    - Triples are parsed with the same tolerant JSON parser as karpathy_llm_wiki
      (markdown-fence-safe). No DSPy. No spaCy fallback. Empty extractions are
      tolerated; the dense path still works.
    - LLM rerank is off by default (cost cap), matching `HippoRAG-lite`.
"""

from __future__ import annotations

import logging
import math
import time

import networkx as nx

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.karpathy_llm_wiki import _safe_parse_json_list

logger = logging.getLogger(__name__)

_OPENIE_SYSTEM = (
    "You are an open information extraction system. Read the passage and emit "
    "(subject, predicate, object) triples that capture its salient relational "
    "facts. Use short noun phrases for subjects and objects (1-4 words).\n\n"
    "Output ONLY a JSON array. No prose, no markdown fences.\n"
    '[{"s": "<subject>", "p": "<predicate>", "o": "<object>"}]\n\n'
    "Rules:\n"
    "- Lowercase entities. Singular form when possible.\n"
    "- Skip pronouns: use the referent (e.g. 'the user') instead of 'they'.\n"
    "- 3-12 triples per passage. Skip filler small-talk.\n"
    "- Empty array is fine if the passage has nothing factual."
)

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)

_RERANK_SYSTEM = (
    "You are re-ranking passages by their relevance to a question. "
    "Output ONLY a JSON array of passage ids ordered most-relevant first. "
    'No prose, no fences. Example: ["p_03", "p_07", "p_01"]'
)


def _cosine(a: list[float], b: list[float]) -> float:
    # Guard with len() rather than truthiness: ``not a`` raises
    # "truth value of an array is ambiguous" when a/b is a numpy array (the
    # embedder hands back arrays, while node embeddings are stored as lists).
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _normalize_entity(raw: str) -> str:
    """Lowercase, strip, collapse internal whitespace. Cap length."""
    if not isinstance(raw, str):
        return ""
    cleaned = " ".join(raw.strip().lower().split())
    return cleaned[:80]


class HippoRAG2Strategy(MemoryStrategy):
    """HippoRAG 2: open-IE graph + dense passage index + personalized PageRank."""

    name = "hipporag2"

    DEFAULT_TOP_K: int = 5
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENIE_MODEL: str = "haiku"  # cheap triple extraction (via LLMClient classify path)
    RECALL_GEN_MODEL: str = "sonnet"  # final answer (via LLMClient generate path)
    PAGERANK_DAMPING: float = 0.5
    PAGERANK_TOL: float = 1e-6
    PAGERANK_MAX_ITER: int = 50
    SYNONYM_THRESHOLD: float = 0.85
    CANDIDATE_POOL_SIZE: int = 30
    LLM_RERANK: bool = False
    SEED_NODES_PER_QUERY: int = 10

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._collection = None
        self._llm: LLMClient | None = None
        # Graph of entity nodes -> predicate-labeled edges.
        self._graph: nx.DiGraph = nx.DiGraph()
        # passage_id -> {session_id, text, entities: set[str]}
        self._passages: dict[str, dict] = {}
        # entity -> set[passage_id]
        self._entity_to_passages: dict[str, set[str]] = {}
        # Cached node embeddings (computed lazily at first recall, re-used after).
        self._node_embeddings: dict[str, list[float]] = {}
        # Tracks whether synonym edges need a refresh (new nodes added since last sync).
        self._synonym_edges_stale: bool = True

    # ------------------------------------------------------------------
    # ChromaDB plumbing (mirrors naive_vector pattern)
    # ------------------------------------------------------------------

    def _collection_name(self) -> str:
        return f"hipporag2_{self.run_id}" if self.run_id else "hipporag2_default"

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        from memory_arena.strategies.embeddings import OpenAIEmbedding

        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        ef = OpenAIEmbedding()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name(),
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _get_embedder(self):
        from memory_arena.strategies.embeddings import OpenAIEmbedding

        return OpenAIEmbedding()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._collection = None
        self._llm = LLMClient()
        self._graph = nx.DiGraph()
        self._passages = {}
        self._entity_to_passages = {}
        self._node_embeddings = {}
        self._synonym_edges_stale = True
        # Force collection creation so teardown works idempotently.
        self._get_collection()

    def _format_session(self, session: Session) -> str:
        lines = [f"Session {session.id} ({session.timestamp or 'unknown'}):"]
        for turn in session.turns:
            lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(lines)

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._llm is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        cost = 0.0
        tokens = 0
        err = ""
        triple_count = 0

        passage_text = self._format_session(session)
        passage_id = f"p_{session.id}"

        # Store the passage embedding in Chroma. The collection handles
        # the OpenAI embedding call internally.
        collection = self._get_collection()
        try:
            collection.upsert(
                ids=[passage_id],
                documents=[passage_text],
                metadatas=[
                    {
                        "session_id": session.id,
                        "timestamp": session.timestamp or "",
                        "user_id": session.user_id,
                    }
                ],
            )
        except Exception as exc:
            err = f"chroma_upsert: {exc}"
            logger.warning("strategy=%s ingest chroma upsert failed: %s", self.name, exc)

        # Run open-IE on the passage. Use the fast/Haiku model.
        triples: list[dict] = []
        if not err:
            try:
                resp = await self._llm._call(
                    "fast",
                    _OPENIE_SYSTEM,
                    passage_text,
                    max_tokens=1500,
                )
                cost += resp.cost_usd
                tokens += resp.total_tokens
                raw = _safe_parse_json_list(resp.text)
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    s = _normalize_entity(item.get("s", ""))
                    p = str(item.get("p", "")).strip().lower()[:60]
                    o = _normalize_entity(item.get("o", ""))
                    if s and o and s != o:
                        triples.append({"s": s, "p": p or "related_to", "o": o})
            except Exception as exc:
                err = f"openie: {exc}"
                logger.warning(
                    "strategy=%s ingest openie failed session=%s: %s",
                    self.name,
                    session.id,
                    exc,
                )

        # Record passage + add graph nodes/edges.
        entities: set[str] = set()
        for t in triples:
            entities.add(t["s"])
            entities.add(t["o"])

        self._passages[passage_id] = {
            "session_id": session.id,
            "text": passage_text,
            "entities": entities,
            "timestamp": session.timestamp or "",
        }

        for ent in entities:
            if not self._graph.has_node(ent):
                self._graph.add_node(ent)
            self._entity_to_passages.setdefault(ent, set()).add(passage_id)

        for t in triples:
            # Aggregate weight on repeated triples so PageRank prefers
            # frequently-co-mentioned entities.
            if self._graph.has_edge(t["s"], t["o"]):
                self._graph[t["s"]][t["o"]]["weight"] = (
                    self._graph[t["s"]][t["o"]].get("weight", 1.0) + 1.0
                )
            else:
                self._graph.add_edge(t["s"], t["o"], predicate=t["p"], weight=1.0)

        triple_count = len(triples)
        self._synonym_edges_stale = True

        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            tokens_used=tokens,
            cost_usd=cost,
            facts_extracted=triple_count,
            error=err,
        )

    # ------------------------------------------------------------------
    # Recall: seed selection -> personalized PageRank -> passage scoring
    # ------------------------------------------------------------------

    def _ensure_node_embeddings(self) -> None:
        """Embed all entity-node phrases that don't have a cached vector yet."""
        nodes = list(self._graph.nodes())
        missing = [n for n in nodes if n not in self._node_embeddings]
        if not missing:
            return
        ef = self._get_embedder()
        # Embed in batches; OpenAIEmbedding already batches inside the call.
        # Small per-batch caps keep memory reasonable on the laptop demo.
        batch = 64
        for s in range(0, len(missing), batch):
            chunk = missing[s : s + batch]
            try:
                vecs = ef(chunk)
            except Exception as exc:
                logger.warning("strategy=%s node embed batch failed: %s", self.name, exc)
                continue
            for name, vec in zip(chunk, vecs, strict=False):
                self._node_embeddings[name] = list(vec)

    def _refresh_synonym_edges(self) -> None:
        """Add undirected synonym edges between near-duplicate node names.

        Vectorized via numpy: builds a (n, d) matrix, L2-normalizes rows, then
        does one matmul to get all pairwise cosines. O(n^2 d) FLOPs run in BLAS,
        not Python — ~1000x faster than the nested-loop version for d=3072.
        """
        if not self._synonym_edges_stale:
            return
        import numpy as np

        names = [n for n in self._node_embeddings.keys() if self._node_embeddings.get(n)]
        if len(names) < 2:
            self._synonym_edges_stale = False
            return
        mat = np.asarray([self._node_embeddings[n] for n in names], dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat = mat / norms
        # Pairwise cosine = mat @ mat.T (upper triangle only, excluding diagonal)
        sim_matrix = mat @ mat.T
        rows, cols = np.where(np.triu(sim_matrix >= self.SYNONYM_THRESHOLD, k=1))
        for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
            sim = float(sim_matrix[i, j])
            self._graph.add_edge(names[i], names[j], predicate="synonym", weight=sim)
            self._graph.add_edge(names[j], names[i], predicate="synonym", weight=sim)
        self._synonym_edges_stale = False

    def _seed_personalization(self, query: str) -> dict[str, float]:
        """Pick top-K query-similar nodes; weight them for personalized PageRank."""
        if not self._node_embeddings:
            return {}
        ef = self._get_embedder()
        try:
            q_vec = ef([query])[0]
        except Exception as exc:
            logger.warning("strategy=%s query embed failed: %s", self.name, exc)
            return {}
        scored: list[tuple[str, float]] = []
        for node, vec in self._node_embeddings.items():
            sim = _cosine(q_vec, vec)
            if sim > 0.0:
                scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.SEED_NODES_PER_QUERY]
        if not top:
            return {}
        total = sum(s for _, s in top) or 1.0
        return {node: sim / total for node, sim in top}

    def _score_passages(self, ppr_scores: dict[str, float]) -> list[tuple[str, float]]:
        """For each passage, sum PPR mass over its entities."""
        out: list[tuple[str, float]] = []
        for pid, payload in self._passages.items():
            entities = payload.get("entities") or set()
            if not entities:
                out.append((pid, 0.0))
                continue
            score = sum(ppr_scores.get(e, 0.0) for e in entities)
            out.append((pid, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    async def _llm_rerank(
        self, query: str, candidates: list[tuple[str, float]]
    ) -> tuple[list[tuple[str, float]], float, int]:
        """Optional Sonnet rerank over the top-pool candidates."""
        if self._llm is None or not candidates:
            return candidates, 0.0, 0
        snippets = []
        for pid, _ in candidates:
            text = self._passages.get(pid, {}).get("text", "")[:600]
            snippets.append(f"[{pid}]\n{text}")
        body = "\n\n---\n\n".join(snippets)
        try:
            resp = await self._llm.generate(query, body, _RERANK_SYSTEM)
            order = _safe_parse_json_list(resp.text)
            seen: dict[str, float] = {pid: score for pid, score in candidates}
            ranked = [pid for pid in order if isinstance(pid, str) and pid in seen]
            # Append any candidates the LLM dropped so we never lose recall.
            for pid in seen:
                if pid not in ranked:
                    ranked.append(pid)
            return [(pid, seen[pid]) for pid in ranked], resp.cost_usd, resp.total_tokens
        except Exception as exc:
            logger.warning("strategy=%s rerank failed: %s", self.name, exc)
            return candidates, 0.0, 0

    async def recall(self, query: str, top_k: int = DEFAULT_TOP_K) -> RecallResult:
        if self._llm is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        cost = 0.0
        tokens = 0

        retrieval_start = time.perf_counter()
        # Lazily build node embeddings + synonym edges the first time recall
        # runs (or whenever new entities have been added since last refresh).
        self._ensure_node_embeddings()
        self._refresh_synonym_edges()

        personalization = self._seed_personalization(query)
        # Run personalized PageRank over the directed graph.
        ppr_scores: dict[str, float] = {}
        if personalization and self._graph.number_of_nodes() > 0:
            try:
                ppr_scores = nx.pagerank(
                    self._graph,
                    alpha=self.PAGERANK_DAMPING,
                    personalization=personalization,
                    max_iter=self.PAGERANK_MAX_ITER,
                    tol=self.PAGERANK_TOL,
                    weight="weight",
                )
            except Exception as exc:
                logger.warning("strategy=%s pagerank failed: %s", self.name, exc)
                ppr_scores = {}

        ranked = self._score_passages(ppr_scores)
        # If the graph is empty or PPR produced no signal, fall back to
        # dense retrieval over the Chroma passage index so we never return
        # nothing.
        if not ppr_scores or all(s == 0.0 for _, s in ranked):
            try:
                collection = self._get_collection()
                result = collection.query(
                    query_texts=[query],
                    n_results=max(top_k, self.CANDIDATE_POOL_SIZE),
                    include=["documents", "metadatas", "distances"],
                )
                ids = result["ids"][0] if result.get("ids") else []
                distances = result["distances"][0] if result.get("distances") else []
                ranked = [
                    (pid, 1.0 - (distances[i] if i < len(distances) else 0.0))
                    for i, pid in enumerate(ids)
                ]
            except Exception as exc:
                logger.warning("strategy=%s dense fallback failed: %s", self.name, exc)

        candidates = ranked[: self.CANDIDATE_POOL_SIZE]
        if self.LLM_RERANK and candidates:
            candidates, rerank_cost, rerank_tokens = await self._llm_rerank(query, candidates)
            cost += rerank_cost
            tokens += rerank_tokens

        top = candidates[:top_k]
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Build context + supporting ids for the generator.
        memories: list[dict] = []
        session_ids: list[str] = []
        for pid, score in top:
            payload = self._passages.get(pid)
            if payload is None:
                # Came from the dense fallback; pull metadata from Chroma.
                sid = pid[2:] if pid.startswith("p_") else pid
                text = ""
                try:
                    res = self._get_collection().get(ids=[pid], include=["documents"])
                    docs = res.get("documents") or []
                    if docs:
                        text = docs[0]
                except Exception:
                    pass
                payload = {"session_id": sid, "text": text}
            sid = payload.get("session_id", "")
            memories.append(
                {
                    "content": payload.get("text", ""),
                    "session_id": sid,
                    "score": float(score),
                }
            )
            if sid and sid not in session_ids:
                session_ids.append(sid)

        context = "\n\n---\n\n".join(
            f"[session={m['session_id']} score={m['score']:.4f}] {m['content']}" for m in memories
        )

        gen_start = time.perf_counter()
        try:
            gen_resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
            cost += gen_resp.cost_usd
            tokens += gen_resp.total_tokens
            answer = gen_resp.text
        except Exception as exc:
            logger.warning("strategy=%s recall generation failed: %s", self.name, exc)
            answer = f"[ERROR] generation failed: {exc}"
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency = (self._start_timer() - start) * 1000
        return RecallResult(
            answer=answer,
            supporting_session_ids=session_ids,
            supporting_turn_ids=[],
            retrieved_memories=memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=tokens,
            cost_usd=cost,
        )

    async def teardown(self) -> None:
        if self._client is not None and self.run_id:
            try:
                self._client.delete_collection(name=self._collection_name())
            except Exception:
                pass
        self._collection = None
        self._llm = None
        self._graph = nx.DiGraph()
        self._passages = {}
        self._entity_to_passages = {}
        self._node_embeddings = {}
        self._synonym_edges_stale = True


__strategy__ = HippoRAG2Strategy
