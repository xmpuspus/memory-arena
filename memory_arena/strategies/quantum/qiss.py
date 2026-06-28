"""QISS: Quantum-Inspired Semantic Similarity reranker (pure NumPy).

Coarse-retrieves through naive_vector's ChromaDB collection, then reranks the
candidates by *quantum fidelity* between the query and each document treated as
pure quantum states.

The fidelity of two pure states |q> and |d> is

    F = Tr(rho_q . rho_d) = |<q|d>|^2 = cos^2(theta)

where rho = |v><v| is the density matrix (outer product) of the unit-normalized
embedding and theta is the angle between the embeddings. Because ChromaDB's
cosine space returns distance = 1 - cos(theta), the single-query fidelity needs
no raw embeddings at all:

    F = (1 - distance)^2

so QISS's single-query reranking is, by construction, cos^2 of naive_vector's
own cosine similarity over the *same* OpenAI embeddings (apples-to-apples). This
is the invariant the unit tests pin.

The research contribution is **multi-query superposition fusion** (opt-in, off
by default so the single-query invariant holds). A multi-session question is
decomposed into sub-queries via the cheap LLM path, each sub-query is embedded,
and they are combined into a single superposition state

    |Q> = a1|q1> + a2|q2> + ...

The reranking score Tr(rho_Q . rho_d) = |<Q|d>|^2 then expands to

    |sum_i a_i <q_i|d>|^2 = sum_ij a_i a_j <q_i|d><d|q_j>

whose i != j terms are genuine quantum *interference* cross-terms. That is
mathematically distinct from classical Reciprocal Rank Fusion, which combines
per-sub-query rankings additively and never produces cross-terms.
"""

from __future__ import annotations

import time

import numpy as np

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.naive_vector import NaiveVectorStrategy

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)

# Strict search-query rewriter. The earlier "decompose into sub-questions" prompt
# on the cheap classify path made the model REFUSE first-person questions ("I
# don't have access to your shopping history...") and emit advice instead of
# queries, which poisoned the superposition. This prompt + the refusal filter
# below keep only real search-query fragments. NOTE: the v0.1.7 retrieval
# experiments (scripts/quantum_experiments.py) showed multi-query superposition
# fusion is INERT here even with clean subqueries (coherent == incoherent, both
# <= single-query cosine), so this mode stays off by default; it is kept for
# demonstrating the interference math, not for a measured retrieval gain.
_DECOMPOSE_SYSTEM = (
    "You rewrite a user's question into standalone SEARCH QUERIES that retrieve the user's "
    "own past chat messages. Output 2 to 4 short keyword-style search queries, one per line. "
    "Each query is a noun phrase or topic, NOT a sentence addressed to anyone. "
    "Do NOT answer the question. Do NOT refuse. Do NOT give advice. "
    "Do NOT say 'I', 'you', or 'your'. If the question is atomic, output a single query."
)

_REFUSAL_MARKERS = (
    "i don't",
    "i do not",
    "i can't",
    "i cannot",
    "i'm happy",
    "could you",
    "if you'd",
    "please ",
    "check your",
    "log into",
    "to answer",
    "here are",
    "sorry",
)


def unit_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm. Zero vectors pass through unchanged."""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def quantum_fidelity(q: np.ndarray, d: np.ndarray) -> float:
    """Fidelity Tr(rho_q . rho_d) = |<q|d>|^2 = cos^2(theta) for two embeddings.

    Forming the full density matrices rho = |v><v| is wasteful (3072x3072 per
    vector); for pure states the trace collapses to the squared inner product
    of the unit vectors, which is what we compute.
    """
    qn = unit_normalize(q)
    dn = unit_normalize(d)
    overlap = float(np.dot(qn, dn))
    return float(min(1.0, max(0.0, overlap * overlap)))


def fidelity_from_cosine_distance(distance: float) -> float:
    """Single-query fidelity from a ChromaDB cosine distance: (1 - distance)^2.

    Clamped to [0, 1]; a probability/metric by construction.
    """
    cos_sim = 1.0 - float(distance)
    fid = cos_sim * cos_sim
    return float(min(1.0, max(0.0, fid)))


def superpose(vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Build the unit superposition state |Q> = sum_i a_i |q_i>.

    ``vectors`` is (k, dim); each row is unit-normalized first so no single
    sub-query dominates by raw magnitude, then summed (optionally weighted) and
    re-normalized to a proper pure state.
    """
    mat = np.asarray(vectors, dtype=float)
    if mat.ndim == 1:
        mat = mat[None, :]
    normed = np.vstack([unit_normalize(row) for row in mat])
    if weights is None:
        combined = normed.sum(axis=0)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        combined = (normed * w).sum(axis=0)
    return unit_normalize(combined)


def fidelity_scores(query_state: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Vectorized fidelity of one query state against many docs.

    ``docs`` is (n, dim). Returns (n,) of |<Q|d>|^2 with each doc unit-normalized.
    The einsum computes the overlaps in one pass; squaring keeps interference
    cross-terms that already live inside the superposed ``query_state``.
    """
    q = unit_normalize(query_state)
    mat = np.asarray(docs, dtype=float)
    if mat.size == 0:
        return np.zeros(0, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_docs = mat / norms
    overlaps = np.einsum("ij,j->i", unit_docs, q)
    return np.clip(overlaps * overlaps, 0.0, 1.0)


class QISSStrategy(MemoryStrategy):
    name = "qiss"

    def __init__(self) -> None:
        super().__init__()
        self._vector = NaiveVectorStrategy()
        self._llm: LLMClient | None = None
        # Over-fetch factor: pull top_k * fanout candidates, rerank, keep top_k.
        self._fanout: int = settings.qiss_fanout
        # Multi-query superposition fusion. Off by default so the single-query
        # fidelity invariant (score == cos^2 of naive_vector's cosine) holds.
        self._decompose: bool = settings.qiss_decompose

    def _collection_name(self) -> str:
        return f"qiss_{self.run_id}" if self.run_id else "qiss_default"

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        # Prefix the inner Chroma collection so QISS shares naive_vector's
        # retrieval substrate without colliding with a concurrent naive_vector.
        await self._vector.setup(f"qiss_{run_id}")
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        # Reranking only changes recall; ingest is identical to naive_vector.
        return await self._vector.ingest_session(session)

    async def _decompose_query(self, query: str) -> list[str]:
        """Rewrite a question into clean search-query fragments via the generate model.

        Uses the strict rewriter prompt + a refusal filter so first-person
        questions don't degrade into refusals/advice that would poison the
        superposition state.
        """
        if self._llm is None:
            self._llm = LLMClient()
        resp = await self._llm.generate(query, "", _DECOMPOSE_SYSTEM, max_tokens=200)
        subs: list[str] = []
        for line in resp.text.splitlines():
            s = line.strip("-*0123456789. \t").strip()
            if not s or len(s) > 90:
                continue
            if any(m in s.lower() for m in _REFUSAL_MARKERS):
                continue
            subs.append(s)
        if query not in subs:
            subs.append(query)
        return subs or [query]

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        start = self._start_timer()
        collection = self._vector._get_collection()
        n_candidates = max(top_k, top_k * max(1, self._fanout))

        include = ["documents", "metadatas", "distances"]
        if self._decompose:
            include = ["embeddings", *include]

        retrieval_start = time.perf_counter()
        results = collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=include,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        if self._decompose and results.get("embeddings"):
            # Multi-query superposition fusion with interference cross-terms.
            from memory_arena.strategies.embeddings import OpenAIEmbedding

            subqueries = await self._decompose_query(query)
            ef = OpenAIEmbedding()
            sub_vecs = np.asarray(ef(subqueries), dtype=float)
            query_state = superpose(sub_vecs)
            doc_embeddings = np.asarray(results["embeddings"][0], dtype=float)
            scores = fidelity_scores(query_state, doc_embeddings).tolist()
        else:
            # Single-query shortcut: fidelity = (1 - cosine_distance)^2 over the
            # same embeddings naive_vector ranked with. No raw vectors needed.
            scores = [fidelity_from_cosine_distance(d) for d in distances]

        ranked = sorted(
            range(len(docs)),
            key=lambda i: scores[i] if i < len(scores) else 0.0,
            reverse=True,
        )[:top_k]

        memories: list[dict] = []
        session_ids: list[str] = []
        turn_ids: list[str] = []
        for i in ranked:
            meta = metas[i] if i < len(metas) else {}
            score = scores[i] if i < len(scores) else 0.0
            memories.append(
                {
                    "content": docs[i],
                    "session_id": meta.get("session_id", ""),
                    "turn_id": meta.get("turn_id", ""),
                    "score": score,
                }
            )
            sid = meta.get("session_id", "")
            tid = meta.get("turn_id", "")
            if sid and sid not in session_ids:
                session_ids.append(sid)
            if tid and tid not in turn_ids:
                turn_ids.append(tid)

        context = "\n\n---\n\n".join(
            f"[session={m['session_id']} turn={m['turn_id']} score={m['score']:.2f}] {m['content']}"
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
        await self._vector.teardown()
        self._llm = None
