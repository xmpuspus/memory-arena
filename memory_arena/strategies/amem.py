"""A-MEM: Agentic Memory (NeurIPS 2025).

Paper: https://arxiv.org/abs/2502.12110
Reference repo: https://github.com/WujiangXu/A-mem
(prompts adapted from agentic_memory/memory_layer.py)

A-MEM is a self-organizing memory system. Each session is turned into 1-3
"notes" by an LLM. Notes carry structured fields (keywords, context, tags) on
top of free-text content. The notes are embedded into a vector store. Every
LINK_EVOLUTION_EVERY sessions, the LLM walks the most recently added notes,
inspects their vector neighbors, and proposes new directed links between them.
The graph of notes the LLM maintains is the "agentic" part.

Retrieval is a two-stage walk:
    1. vector search returns top_k notes
    2. each retrieved note expands to include up to MAX_LINKED_NEIGHBORS
       of its previously-linked notes (deduplicated against the seed set)
The merged set goes to the generator as context.

Deviations from the reference repo:
    - We use OpenAI text-embedding-3-large (not all-MiniLM-L6-v2) to match the
      rest of memory-arena.
    - We collapse the reference's per-content add_note + analyze_content into
      one Haiku call per session that emits a JSON array of notes. This keeps
      ingest cost in line with karpathy_llm_wiki and matches our LLMClient
      generate() surface.
    - Link evolution runs every N sessions instead of per-note (paper uses a
      threshold but at small corpus sizes that means it never fires).
    - We store note.linked_note_ids as a flat list of ids; the reference uses
      a dict keyed by relation type but the eval doesn't differentiate.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from memory_arena.llm.client import LLMClient
from memory_arena.sessions.schema import Session
from memory_arena.settings import settings
from memory_arena.strategies.base import IngestRecord, MemoryStrategy, RecallResult
from memory_arena.strategies.karpathy_llm_wiki import _safe_parse_json_list

logger = logging.getLogger(__name__)


_NOTE_GEN_SYSTEM = (
    "You are a memory note generator. Given a chat session, emit 1-3 durable "
    "memory notes that capture facts worth remembering about the user. Skip "
    "social pleasantries.\n\n"
    "Output ONLY a JSON array. No prose, no markdown fences.\n"
    "[\n"
    '  {"content": "<one-sentence durable fact about the user>", '
    '"keywords": ["<noun>", "<verb>", "<concept>"], '
    '"context": "<one-sentence motivation / domain>", '
    '"tags": ["<broad category>", "<topic>"]}\n'
    "]\n\n"
    "Rules:\n"
    "- 1-3 notes per session. Pick the most durable facts.\n"
    "- keywords: 3+ salient nouns/verbs/concepts, most-to-least important. "
    "Skip speaker names and timestamps.\n"
    "- context: one sentence on the domain / why this matters.\n"
    "- tags: 2+ broad categories for classification.\n"
    "- Skip the session entirely (return []) if nothing is worth remembering."
)

_LINK_EVOLUTION_SYSTEM = (
    "You are a memory evolution agent. You manage a knowledge base of memory "
    "notes. Given a batch of recently added notes and each one's nearest "
    "neighbors, decide which new directed links should be created.\n\n"
    "Output ONLY a JSON array. No prose, no markdown fences.\n"
    "[\n"
    '  {"from_note_id": "<id>", "to_note_id": "<id>", '
    '"reason": "<short why these connect>"}\n'
    "]\n\n"
    "Rules:\n"
    "- Only link notes that share a concrete topic, entity, or causal link.\n"
    "- Skip pairs that are merely tangentially related.\n"
    "- Return [] if no links are warranted.\n"
    "- Use the exact note ids provided. Do not invent ids."
)

_RECALL_SYSTEM = (
    "You are an assistant answering using retrieved memory notes from a chat "
    "history. Each note is rendered as [note=<id> session=<id>] <content>. "
    "Use only information that appears in the retrieved notes. Cite sessions "
    "by their id in square brackets. If the notes do not contain the answer, "
    'say "I do not have that information."'
)


def _embed_payload(content: str, keywords: list[str], context: str) -> str:
    """Build the text the embedding model sees for a note.

    Concatenate content + keywords + context so that semantically related
    notes cluster on more than just the literal sentence.
    """
    kw = " ".join(keywords or [])
    ctx = context or ""
    return f"{content} || {kw} || {ctx}"


class AMEMStrategy(MemoryStrategy):
    """Agentic memory: LLM-structured notes with self-evolving link graph."""

    name = "amem"

    DEFAULT_TOP_K = 5
    NOTE_EMBEDDING_MODEL = "text-embedding-3-large"
    NOTE_GEN_MODEL = "haiku"  # cheap structuring; runs every ingest
    LINK_EVOLUTION_MODEL = "haiku"  # cheap link evolution
    RECALL_GEN_MODEL = "sonnet"  # final answer generation handled by LLMClient.generate()
    LINK_EVOLUTION_EVERY = 5
    MAX_LINKED_NEIGHBORS = 3
    LINK_EVOLUTION_K_NEIGHBORS = 5  # vector neighbors fetched per recent note

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._collection = None
        self._llm: LLMClient | None = None
        # In-memory note graph. ChromaDB stores embeddings + metadata but the
        # graph (id -> linked_note_ids) lives here so link evolution can mutate
        # it cheaply without round-tripping through the vector store.
        self._notes: dict[str, dict[str, Any]] = {}
        self._recent_note_ids: list[str] = []  # ids added since last evolution
        self._sessions_since_evolution = 0
        self._errors: list[dict] = []

    def _collection_name(self) -> str:
        return f"amem_{self.run_id}" if self.run_id else "amem_default"

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        from memory_arena.strategies.embeddings import OpenAIEmbedding

        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        ef = OpenAIEmbedding(model=self.NOTE_EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name(),
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    async def setup(self, run_id: str) -> None:
        self.run_id = run_id
        self._collection = None
        self._notes = {}
        self._recent_note_ids = []
        self._sessions_since_evolution = 0
        self._errors = []
        self._llm = LLMClient()
        self._get_collection()  # force creation so teardown is idempotent

    def _format_session(self, session: Session) -> str:
        body_lines = [f"Session {session.id} ({session.timestamp or 'unknown'}):"]
        for turn in session.turns:
            body_lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(body_lines)

    def _new_note_id(self, session_id: str) -> str:
        # Short uuid suffix keeps ids readable in logs while staying unique.
        return f"{session_id}_note_{uuid.uuid4().hex[:8]}"

    async def ingest_session(self, session: Session) -> IngestRecord:
        if self._llm is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        cost = 0.0
        tokens = 0
        err = ""
        notes_created = 0

        body = self._format_session(session)
        prompt = f"{body}\n\nReturn the JSON array of memory notes for this session."

        try:
            resp = await self._llm.generate(prompt, "", _NOTE_GEN_SYSTEM)
            cost += resp.cost_usd
            tokens += resp.total_tokens
            parsed = _safe_parse_json_list(resp.text)
            collection = self._get_collection()

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for raw in parsed:
                if not isinstance(raw, dict):
                    continue
                content = (raw.get("content") or "").strip()
                if not content:
                    continue
                keywords = [str(k) for k in raw.get("keywords", []) if k]
                context = str(raw.get("context") or "").strip()
                tags = [str(t) for t in raw.get("tags", []) if t]

                note_id = self._new_note_id(session.id)
                note = {
                    "id": note_id,
                    "session_id": session.id,
                    "content": content,
                    "keywords": keywords,
                    "context": context,
                    "tags": tags,
                    "linked_note_ids": [],
                    "created_at": session.timestamp or "",
                }
                self._notes[note_id] = note
                self._recent_note_ids.append(note_id)

                ids.append(note_id)
                documents.append(_embed_payload(content, keywords, context))
                # ChromaDB metadata must be flat scalars; serialize lists.
                metadatas.append(
                    {
                        "session_id": session.id,
                        "note_id": note_id,
                        "user_id": session.user_id,
                        "keywords": ",".join(keywords),
                        "tags": ",".join(tags),
                        "context": context,
                        "timestamp": session.timestamp or "",
                    }
                )

            if ids:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            notes_created = len(ids)
        except Exception as exc:
            err = f"ingest: {exc}"
            logger.warning(
                "strategy=%s ingest session=%s failed: %s",
                self.name,
                session.id,
                exc,
            )
            self._errors.append(
                {
                    "phase": "ingest",
                    "session_id": session.id,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )

        # Periodic link-evolution pass.
        self._sessions_since_evolution += 1
        if not err and self._sessions_since_evolution >= self.LINK_EVOLUTION_EVERY:
            try:
                evo_cost, evo_tokens = await self._evolve_links()
                cost += evo_cost
                tokens += evo_tokens
            except Exception as evo_exc:
                logger.warning(
                    "strategy=%s link-evolution after session=%s failed: %s",
                    self.name,
                    session.id,
                    evo_exc,
                )
                self._errors.append(
                    {
                        "phase": "ingest",
                        "step": "link_evolution",
                        "session_id": session.id,
                        "error": str(evo_exc),
                        "type": type(evo_exc).__name__,
                    }
                )
            self._sessions_since_evolution = 0
            self._recent_note_ids = []

        elapsed = (self._start_timer() - start) * 1000
        return IngestRecord(
            session_id=session.id,
            latency_ms=elapsed,
            tokens_used=tokens,
            cost_usd=cost,
            facts_extracted=notes_created,
            error=err,
        )

    async def _evolve_links(self) -> tuple[float, int]:
        """Have the LLM propose new links between recent notes and their neighbors.

        Returns (cost, tokens) consumed by the evolution call.
        """
        if self._llm is None or not self._recent_note_ids:
            return 0.0, 0
        if len(self._notes) < 2:
            return 0.0, 0

        collection = self._get_collection()
        # For each recent note, fetch its K nearest neighbors (other notes).
        rendered_blocks: list[str] = []
        for nid in self._recent_note_ids:
            note = self._notes.get(nid)
            if note is None:
                continue
            query_text = _embed_payload(note["content"], note["keywords"], note["context"])
            try:
                hits = collection.query(
                    query_texts=[query_text],
                    n_results=self.LINK_EVOLUTION_K_NEIGHBORS + 1,  # +1 for self
                    include=["metadatas", "documents"],
                )
            except Exception as q_exc:
                logger.warning(
                    "strategy=%s evolve_links neighbor query failed for %s: %s",
                    self.name,
                    nid,
                    q_exc,
                )
                continue
            metas = hits["metadatas"][0] if hits.get("metadatas") else []
            neighbor_ids = [m.get("note_id", "") for m in metas if m.get("note_id") != nid][
                : self.LINK_EVOLUTION_K_NEIGHBORS
            ]
            neighbor_lines: list[str] = []
            for n_id in neighbor_ids:
                n = self._notes.get(n_id)
                if n is None:
                    continue
                neighbor_lines.append(f"  - [{n_id}] {n['content']} (tags: {', '.join(n['tags'])})")
            rendered_blocks.append(
                f"Recent note [{nid}]: {note['content']}\n"
                f"  tags: {', '.join(note['tags'])}\n"
                f"  keywords: {', '.join(note['keywords'])}\n"
                f"  context: {note['context']}\n"
                "  neighbors:\n" + ("\n".join(neighbor_lines) or "  (none)")
            )

        if not rendered_blocks:
            return 0.0, 0
        body = "\n\n".join(rendered_blocks)
        resp = await self._llm.generate(
            f"Recent notes and neighbors:\n{body}\n\n"
            "Return the JSON array of new directed links to add.",
            "",
            _LINK_EVOLUTION_SYSTEM,
        )
        proposed = _safe_parse_json_list(resp.text)
        for link in proposed:
            if not isinstance(link, dict):
                continue
            src = link.get("from_note_id", "")
            dst = link.get("to_note_id", "")
            if not src or not dst or src == dst:
                continue
            if src not in self._notes or dst not in self._notes:
                continue
            linked = self._notes[src]["linked_note_ids"]
            if dst not in linked:
                linked.append(dst)
        return resp.cost_usd, resp.total_tokens

    def _expand_with_links(self, seed_note_ids: list[str]) -> list[str]:
        """Take a seed list and add up to MAX_LINKED_NEIGHBORS linked notes per seed."""
        seen = set(seed_note_ids)
        ordered = list(seed_note_ids)
        for nid in seed_note_ids:
            note = self._notes.get(nid)
            if note is None:
                continue
            for linked_id in note.get("linked_note_ids", [])[: self.MAX_LINKED_NEIGHBORS]:
                if linked_id in seen:
                    continue
                if linked_id not in self._notes:
                    continue
                ordered.append(linked_id)
                seen.add(linked_id)
        return ordered

    async def recall(self, query: str, top_k: int = 10) -> RecallResult:
        if self._llm is None:
            raise RuntimeError("setup() not called")
        start = self._start_timer()
        # Cap at DEFAULT_TOP_K to stay paper-faithful, but honor a tighter
        # budget if the runner asks for fewer.
        k = min(top_k, self.DEFAULT_TOP_K) if top_k else self.DEFAULT_TOP_K

        collection = self._get_collection()
        retrieval_start = time.perf_counter()
        try:
            results = collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            logger.warning("strategy=%s vector query failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "vector_query",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
            latency = (self._start_timer() - start) * 1000
            return RecallResult(
                answer="[ERROR] vector query failed",
                supporting_session_ids=[],
                supporting_turn_ids=[],
                retrieved_memories=[],
                strategy=self.name,
                latency_ms=latency,
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=0.0,
                tokens_used=0,
                cost_usd=0.0,
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        metas = results["metadatas"][0] if results.get("metadatas") else []
        documents = results["documents"][0] if results.get("documents") else []
        distances = results["distances"][0] if results.get("distances") else []

        seed_ids: list[str] = []
        score_by_id: dict[str, float] = {}
        for i, meta in enumerate(metas):
            nid = meta.get("note_id", "")
            if not nid:
                continue
            seed_ids.append(nid)
            score_by_id[nid] = 1.0 - (distances[i] if i < len(distances) else 0.0)

        expanded_ids = self._expand_with_links(seed_ids)

        memories: list[dict] = []
        session_ids: list[str] = []
        for nid in expanded_ids:
            note = self._notes.get(nid)
            if note is None:
                continue
            sid = note["session_id"]
            memories.append(
                {
                    "note_id": nid,
                    "content": note["content"],
                    "keywords": note["keywords"],
                    "tags": note["tags"],
                    "session_id": sid,
                    "score": score_by_id.get(nid, 0.0),
                    "linked": nid not in seed_ids,
                }
            )
            if sid and sid not in session_ids:
                session_ids.append(sid)

        context = "\n\n---\n\n".join(
            f"[note={m['note_id']} session={m['session_id']} score={m['score']:.2f}] "
            f"{m['content']}"
            for m in memories
        )
        if not context and documents:
            # Defensive fallback: documents arrived but no metadata matched.
            context = "\n\n---\n\n".join(documents)

        gen_start = time.perf_counter()
        try:
            gen_resp = await self._llm.generate(query, context, _RECALL_SYSTEM)
            answer = gen_resp.text
            gen_cost = gen_resp.cost_usd
            gen_tokens = gen_resp.total_tokens
        except Exception as exc:
            logger.warning("strategy=%s recall generation failed: %s", self.name, exc)
            self._errors.append(
                {
                    "phase": "recall",
                    "step": "generation",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
            answer = f"[ERROR] generation failed: {exc}"
            gen_cost = 0.0
            gen_tokens = 0
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
            tokens_used=gen_tokens,
            cost_usd=gen_cost,
        )

    async def teardown(self) -> None:
        if self._client is not None and self.run_id:
            try:
                self._client.delete_collection(name=self._collection_name())
            except Exception:
                pass
        self._collection = None
        self._notes = {}
        self._recent_note_ids = []
        self._llm = None
