"""Naive vector recall: embed every turn, top-k cosine, then generate.

Each turn becomes one document. Metadata pins the session id and turn id so we
can return supporting_session_ids cleanly. Run-id namespacing keeps concurrent
benchmarks from contaminating each other.
"""

from __future__ import annotations

import time

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult

_RECALL_SYSTEM = (
    "You are an assistant answering a question using retrieved memories from a chat history. "
    "Use only information that appears in the retrieved memories. Cite sessions by their id "
    "in square brackets. If the memories do not contain the answer, say "
    '"I do not have that information."'
)


class NaiveVectorStrategy(MemoryStrategy):
    name = "naive_vector"

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._collection = None
        self._llm: LLMClient | None = None

    def _collection_name(self) -> str:
        return f"naive_vector_{self.run_id}" if self.run_id else "naive_vector_default"

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

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._collection = None
        self._llm = LLMClient()
        # Force collection creation so teardown works idempotently.
        self._get_collection()

    async def ingest_session(self, session: Session) -> IngestRecord:
        start = self._start_timer()
        collection = self._get_collection()
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        for turn in session.turns:
            ids.append(turn.id)
            texts.append(f"{turn.role}: {turn.content}")
            metadatas.append(
                {
                    "session_id": session.id,
                    "turn_id": turn.id,
                    "user_id": session.user_id,
                    "role": turn.role,
                    "timestamp": turn.timestamp or session.timestamp or "",
                }
            )
        if ids:
            batch = 500
            for s in range(0, len(ids), batch):
                collection.upsert(
                    ids=ids[s : s + batch],
                    documents=texts[s : s + batch],
                    metadatas=metadatas[s : s + batch],
                )
        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            tokens_used=0,
            cost_usd=0.0,
            facts_extracted=len(ids),
        )

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        start = self._start_timer()
        collection = self._get_collection()
        retrieval_start = time.perf_counter()
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        chunks = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results.get("distances") else []

        memories: list[dict] = []
        session_ids: list[str] = []
        turn_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            meta = metas[i] if i < len(metas) else {}
            score = 1.0 - (distances[i] if i < len(distances) else 0.0)
            memories.append(
                {
                    "content": chunk,
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
        if self._client is not None and self.run_id:
            try:
                self._client.delete_collection(name=self._collection_name())
            except Exception:
                pass
        self._collection = None
        self._llm = None
