"""SQR: Simulated Quantum Reranker (Qiskit Aer SWAP test, simulator only).

Same coarse phase as qiss: retrieve top_k * fanout candidates from
naive_vector's ChromaDB index (NEVER a parallel index). Then, instead of the
closed-form cosine^2, run an actual SWAP-test quantum circuit on the Aer
simulator to estimate each query-document state overlap |<q|d>|^2.

Because text-embedding-3-large is 3072-d and a 4-qubit register holds only 16
amplitudes, embeddings are PCA-projected to 2^n_qubits dims before amplitude
encoding. The variance that projection discards is recorded
(``pca_variance_explained``), the honest headline cost of the quantum encoding.

Statevector mode is the exact, noiseless default (correct for benchmarking).
``settings.sqr_shots > 0`` switches to the sampled estimator for the
accuracy-vs-speed curve.

qiskit / qiskit_aer are imported at module top so the strategy registry's
_try_import gate drops sqr cleanly when the optional [quantum] extra is absent,
keeping the core install and CI light.
"""

from __future__ import annotations

import time

import numpy as np
import qiskit  # noqa: F401  (module-top import gates the registry's _try_import)
import qiskit_aer  # noqa: F401

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.naive_vector import NaiveVectorStrategy
from memory_arena.strategies.quantum.circuits import swap_test_fidelities
from memory_arena.strategies.quantum.utils import PCAReducer, amplitude_encode, state_dim

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)

# Cap the corpus sample the PCA basis is fit on. A few hundred turns is plenty to
# estimate the principal directions and keeps setup-time fit cheap.
_PCA_FIT_SAMPLE = 512


class SQRStrategy(MemoryStrategy):
    name = "sqr"

    def __init__(self) -> None:
        super().__init__()
        self._vector = NaiveVectorStrategy()
        self._llm: LLMClient | None = None
        self._fanout: int = settings.sqr_fanout
        self._n_qubits: int = settings.sqr_n_qubits
        self._shots: int = settings.sqr_shots  # 0 = exact statevector
        self._target_dims: int = state_dim(self._n_qubits)
        self._reducer: PCAReducer | None = None
        self.pca_variance_explained: float | None = None

    def _collection_name(self) -> str:
        return f"sqr_{self.run_id}" if self.run_id else "sqr_default"

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        # Share naive_vector's retrieval substrate via a prefixed collection.
        await self._vector.setup(f"sqr_{run_id}")
        self._llm = LLMClient()

    async def ingest_session(self, session: Session) -> IngestRecord:
        # Reranking only changes recall; ingest is identical to naive_vector.
        return await self._vector.ingest_session(session)

    def _ensure_reducer(self, collection) -> None:
        """Fit the PCA basis once, on a sample of the ingested corpus embeddings.

        Fitting on the corpus (not the per-query candidates) makes the variance
        figure an honest 3072 -> 2^n projection cost rather than an artifact of a
        20-point candidate cloud.
        """
        if self._reducer is not None:
            return
        sample = collection.get(include=["embeddings"], limit=_PCA_FIT_SAMPLE)
        embeddings = sample.get("embeddings") if sample else None
        if embeddings is None or len(embeddings) == 0:
            return
        mat = np.asarray(embeddings, dtype=float)
        self._reducer = PCAReducer(self._target_dims).fit(mat)
        self.pca_variance_explained = self._reducer.variance_explained

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        from memory_arena.strategies.embeddings import OpenAIEmbedding

        start = self._start_timer()
        collection = self._vector._get_collection()
        self._ensure_reducer(collection)
        n_candidates = max(top_k, top_k * max(1, self._fanout))

        retrieval_start = time.perf_counter()
        results = collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []
        embeddings = results["embeddings"][0] if results.get("embeddings") else []

        quantum_ms = 0.0
        if self._reducer is not None and len(embeddings):
            q_vec = np.asarray(OpenAIEmbedding()([query])[0], dtype=float)
            q_red = self._reducer.transform([q_vec])[0]
            d_red = self._reducer.transform(np.asarray(embeddings, dtype=float))
            q_amp = amplitude_encode(q_red, self._n_qubits)
            d_amps = [amplitude_encode(d, self._n_qubits) for d in d_red]
            q_start = time.perf_counter()
            scores = swap_test_fidelities(q_amp, d_amps, self._n_qubits, shots=self._shots)
            quantum_ms = (time.perf_counter() - q_start) * 1000
        else:
            # Degenerate fallback (empty store / unfit reducer): cosine^2 from
            # the distances so recall still returns something sane.
            scores = [max(0.0, min(1.0, (1.0 - d) ** 2)) for d in distances]

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
        # Quantum overhead (circuit build + simulate) is logged for the honest
        # "overhead vs naive_vector" metric; recall_latency stays comparable.
        self.last_quantum_ms = quantum_ms
        return RecallResult(
            answer=resp.text,
            supporting_session_ids=session_ids,
            supporting_turn_ids=turn_ids,
            retrieved_memories=memories,
            strategy=self.name,
            latency_ms=latency,
            retrieval_latency_ms=retrieval_ms + quantum_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def teardown(self) -> None:
        await self._vector.teardown()
        self._llm = None
        self._reducer = None
