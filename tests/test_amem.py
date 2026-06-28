"""Tests for AMEMStrategy (A-MEM: Agentic Memory).

LLM and ChromaDB are both mocked; no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_arena.llm.client import LLMResponse
from memory_arena.strategies.amem import AMEMStrategy
from memory_arena.strategies.base import IngestRecord, RecallResult

# A typical note-generation response — emits two structured notes for one session.
_INGEST_RESPONSE = LLMResponse(
    text=(
        "["
        '{"content": "User works as a software engineer at a fintech startup.",'
        ' "keywords": ["software engineer", "fintech", "startup"],'
        ' "context": "Career fact about the user.",'
        ' "tags": ["career", "tech"]},'
        '{"content": "User is building a fraud detection pipeline using transformer embeddings.",'
        ' "keywords": ["fraud detection", "transformer", "embeddings"],'
        ' "context": "Current work focus.",'
        ' "tags": ["project", "ml"]}'
        "]"
    ),
    input_tokens=120,
    output_tokens=80,
    cost_usd=0.001,
)

_LINK_RESPONSE = LLMResponse(
    text=(
        '[{"from_note_id": "PLACEHOLDER_A", "to_note_id": "PLACEHOLDER_B",'
        ' "reason": "Both about the user\'s work at fintech startup"}]'
    ),
    input_tokens=80,
    output_tokens=30,
    cost_usd=0.0005,
)

_RECALL_RESPONSE = LLMResponse(
    text="User is a software engineer at a fintech startup [session_03].",
    input_tokens=60,
    output_tokens=20,
    cost_usd=0.0003,
)


def _make_collection_mock():
    """Build a chroma collection mock that echoes back the metadata it was upserted with.

    Stores upserted ids+metadatas so query() can return real note ids — A-MEM
    looks them up in self._notes, so query must yield ids that the strategy
    actually added.
    """
    collection = MagicMock()
    state = {"ids": [], "metas": [], "documents": []}

    def fake_upsert(ids=None, documents=None, metadatas=None, **kw):
        state["ids"].extend(ids or [])
        state["metas"].extend(metadatas or [])
        state["documents"].extend(documents or [])

    def fake_query(query_texts=None, n_results=5, **kw):
        # Return all stored entries up to n_results, in insertion order.
        # That's fine for tests — the fact that we return real note ids
        # is what matters.
        n = min(n_results, len(state["ids"]))
        return {
            "ids": [state["ids"][:n]],
            "documents": [state["documents"][:n]],
            "metadatas": [state["metas"][:n]],
            "distances": [[0.1] * n],
        }

    collection.upsert.side_effect = fake_upsert
    collection.query.side_effect = fake_query
    collection._state = state
    return collection


def _make_chroma_client(collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.delete_collection.return_value = None
    return client


def _patch_llm_factory(monkeypatch, fake_llm):
    monkeypatch.setattr("memory_arena.strategies.amem.LLMClient", lambda: fake_llm)


class TestAMEMSetupTeardown:
    @pytest.mark.asyncio
    async def test_setup_creates_collection_scoped_to_run_id(self, monkeypatch):
        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(return_value=_INGEST_RESPONSE)
        _patch_llm_factory(monkeypatch, fake_llm)

        collection = _make_collection_mock()
        client = _make_chroma_client(collection)

        with patch("chromadb.PersistentClient", return_value=client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = AMEMStrategy()
                await s.setup("run-amem-test")
                assert s.run_id == "run-amem-test"
                assert s._collection is not None
                # Collection name must be namespaced by run_id.
                called_kwargs = client.get_or_create_collection.call_args.kwargs
                assert called_kwargs["name"] == "amem_run-amem-test"
                await s.teardown()
                # delete_collection was invoked with the right name.
                client.delete_collection.assert_called_with(name="amem_run-amem-test")
                assert s._notes == {}
                assert s._llm is None


class TestAMEMIngest:
    @pytest.mark.asyncio
    async def test_ingest_session_emits_one_to_three_notes(self, monkeypatch, sample_session):
        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(return_value=_INGEST_RESPONSE)
        _patch_llm_factory(monkeypatch, fake_llm)

        collection = _make_collection_mock()
        client = _make_chroma_client(collection)

        with patch("chromadb.PersistentClient", return_value=client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = AMEMStrategy()
                await s.setup("run-amem-ingest")
                rec = await s.ingest_session(sample_session)

                assert isinstance(rec, IngestRecord)
                assert rec.session_id == sample_session.id
                assert rec.error == ""
                # Two notes in our canned response.
                assert rec.facts_extracted == 2
                assert 1 <= len(s._notes) <= 3
                # Each note carries the session id and structured fields.
                for note in s._notes.values():
                    assert note["session_id"] == sample_session.id
                    assert note["content"]
                    assert isinstance(note["keywords"], list)
                    assert isinstance(note["tags"], list)
                    assert note["linked_note_ids"] == []
                # upsert was called once with the 2 notes.
                assert collection.upsert.call_count == 1

    @pytest.mark.asyncio
    async def test_ingest_with_empty_llm_response_records_zero_notes(
        self, monkeypatch, sample_session
    ):
        empty = LLMResponse(text="[]", input_tokens=10, output_tokens=5, cost_usd=0.00001)
        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(return_value=empty)
        _patch_llm_factory(monkeypatch, fake_llm)

        collection = _make_collection_mock()
        client = _make_chroma_client(collection)
        with patch("chromadb.PersistentClient", return_value=client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = AMEMStrategy()
                await s.setup("run-amem-empty")
                rec = await s.ingest_session(sample_session)
                assert rec.facts_extracted == 0
                # No notes created; collection.upsert not called.
                collection.upsert.assert_not_called()
                await s.teardown()


class TestAMEMRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_top_k_notes_with_session_ids(self, monkeypatch, sample_session):
        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(side_effect=[_INGEST_RESPONSE, _RECALL_RESPONSE])
        _patch_llm_factory(monkeypatch, fake_llm)

        collection = _make_collection_mock()
        client = _make_chroma_client(collection)
        with patch("chromadb.PersistentClient", return_value=client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = AMEMStrategy()
                await s.setup("run-amem-recall")
                await s.ingest_session(sample_session)

                result = await s.recall("What does the user do?", top_k=5)
                assert isinstance(result, RecallResult)
                assert result.strategy == "amem"
                assert "fintech" in result.answer
                assert sample_session.id in result.supporting_session_ids
                # Cap at DEFAULT_TOP_K (5).
                assert len(result.retrieved_memories) <= s.DEFAULT_TOP_K
                # Each retrieved memory carries a note_id.
                for m in result.retrieved_memories:
                    assert m["note_id"] in s._notes

    @pytest.mark.asyncio
    async def test_recall_expands_with_linked_neighbors(self, monkeypatch, sample_session):
        """If a note has linked_note_ids set, recall should fan out to include them."""
        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(side_effect=[_INGEST_RESPONSE, _RECALL_RESPONSE])
        _patch_llm_factory(monkeypatch, fake_llm)

        collection = _make_collection_mock()
        client = _make_chroma_client(collection)
        with patch("chromadb.PersistentClient", return_value=client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = AMEMStrategy()
                await s.setup("run-amem-links")
                await s.ingest_session(sample_session)

                # Manually wire one note to link to the other so we can verify
                # the fan-out logic without depending on link evolution timing.
                note_ids = list(s._notes.keys())
                assert len(note_ids) >= 2
                # Make the query return ONLY the first note, then link evolution
                # should fan out to the second.
                first_id = note_ids[0]
                second_id = note_ids[1]
                s._notes[first_id]["linked_note_ids"] = [second_id]

                collection._state["ids"] = [first_id]
                collection._state["metas"] = [
                    {"note_id": first_id, "session_id": sample_session.id}
                ]
                collection._state["documents"] = ["content of first note"]

                result = await s.recall("?", top_k=5)
                retrieved_ids = [m["note_id"] for m in result.retrieved_memories]
                assert first_id in retrieved_ids
                assert second_id in retrieved_ids
                # The second one is marked as linked (not a vector seed).
                linked = next(m for m in result.retrieved_memories if m["note_id"] == second_id)
                assert linked["linked"] is True


class TestAMEMLinkEvolution:
    @pytest.mark.asyncio
    async def test_link_evolution_fires_every_n_sessions(self, monkeypatch, sample_session):
        """Run LINK_EVOLUTION_EVERY ingests; the LLM should be called with the
        link-evolution prompt and at least one link should be added."""
        fake_llm = AsyncMock()

        ingest_calls = 0
        link_calls = 0
        # The link prompt uses the actual note ids that exist in the strategy.
        # We capture them at call time by inspecting the strategy state.
        # To keep the mock simple, return an evolution response that ALWAYS
        # connects the first two notes in the strategy.

        strategy_ref: dict = {}

        async def fake_generate(*args, **kwargs):
            nonlocal ingest_calls, link_calls
            # The link-evolution system prompt mentions "memory evolution agent".
            # Distinguish by inspecting the system arg.
            system = args[2] if len(args) >= 3 else kwargs.get("system_prompt", "")
            if "memory evolution agent" in system:
                link_calls += 1
                s = strategy_ref.get("s")
                if s and len(s._notes) >= 2:
                    ids = list(s._notes.keys())
                    return LLMResponse(
                        text=(
                            f'[{{"from_note_id": "{ids[0]}", "to_note_id": "{ids[1]}",'
                            ' "reason": "test link"}]'
                        ),
                        input_tokens=50,
                        output_tokens=20,
                        cost_usd=0.0005,
                    )
                return LLMResponse(text="[]", input_tokens=10, output_tokens=2, cost_usd=0.0)
            ingest_calls += 1
            return _INGEST_RESPONSE

        fake_llm.generate = AsyncMock(side_effect=fake_generate)
        _patch_llm_factory(monkeypatch, fake_llm)

        collection = _make_collection_mock()
        client = _make_chroma_client(collection)
        with patch("chromadb.PersistentClient", return_value=client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = AMEMStrategy()
                strategy_ref["s"] = s
                await s.setup("run-amem-evo")

                # Ingest LINK_EVOLUTION_EVERY sessions to trigger one evolution pass.
                for i in range(s.LINK_EVOLUTION_EVERY):
                    # Reuse sample_session but mutate the id so each is distinct.
                    sess = sample_session.model_copy(deep=True)
                    sess.id = f"session_evo_{i}"
                    for t in sess.turns:
                        t.session_id = sess.id
                        t.id = f"{sess.id}_turn"
                    await s.ingest_session(sess)

                assert ingest_calls == s.LINK_EVOLUTION_EVERY
                assert link_calls == 1
                # At least one note should now have a linked neighbor.
                any_linked = any(n["linked_note_ids"] for n in s._notes.values())
                assert any_linked, "expected link evolution to add at least one link"
                # The recent-notes buffer was reset after evolution.
                assert s._recent_note_ids == []
