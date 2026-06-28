"""Mock-based lifecycle smoke tests for vendor and pure-Python strategies.

Each test exercises ``setup -> ingest_session -> recall -> teardown`` against a
mocked vendor SDK / LLM client and asserts the strategy returns the expected
types without crashing. These are the entry-level coverage tests the v0.1.5
audit flagged as missing.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_arena.llm.client import LLMResponse
from memory_arena.strategies.base import IngestRecord, RecallResult


def _mock_llm_client():
    fake = AsyncMock()
    fake.generate = AsyncMock(
        return_value=LLMResponse(
            text="The user works as a software engineer [session_03].",
            input_tokens=80,
            output_tokens=20,
            cost_usd=0.0003,
        )
    )
    fake.classify = AsyncMock(return_value="answer")
    return fake


# ---------------------------------------------------------------------------
# Mem0Strategy — lifecycle with mocked mem0.Memory
# ---------------------------------------------------------------------------


class TestMem0Lifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session):
        from memory_arena.strategies.mem0 import Mem0Strategy

        # Mock the mem0.Memory client
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": [{"id": "m1", "memory": "extracted fact"}]}
        mock_memory.search.return_value = {
            "results": [
                {
                    "id": "m1",
                    "memory": "User is a software engineer at a fintech startup.",
                    "metadata": {"session_id": "session_03"},
                }
            ]
        }
        mock_memory.delete_all.return_value = None

        mock_memory_cls = MagicMock()
        mock_memory_cls.from_config.return_value = mock_memory

        # Patch the import so `from mem0 import Memory` returns our mock
        fake_mem0 = SimpleNamespace(Memory=mock_memory_cls)
        monkeypatch.setitem(sys.modules, "mem0", fake_mem0)

        # Patch the LLM client used inside recall()
        fake_llm = _mock_llm_client()
        monkeypatch.setattr("memory_arena.strategies.mem0.LLMClient", lambda: fake_llm)

        s = Mem0Strategy()
        await s.setup("run-mem0-test")
        assert s.run_id == "run-mem0-test"
        assert s._client is mock_memory

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id
        assert rec.error == ""
        mock_memory.add.assert_called()

        result = await s.recall("What does the user do?")
        assert isinstance(result, RecallResult)
        assert result.strategy == "mem0"
        assert "session_03" in result.supporting_session_ids
        # v2 API contract: search() takes top_k + filters, NOT limit + user_id.
        mock_memory.search.assert_called()
        search_kwargs = mock_memory.search.call_args.kwargs
        assert "top_k" in search_kwargs
        assert search_kwargs.get("filters", {}).get("user_id")

        await s.teardown()
        assert s._client is None
        mock_memory.delete_all.assert_called()


# ---------------------------------------------------------------------------
# Mem0GraphStrategy — same lifecycle but with graph_store wired up
# ---------------------------------------------------------------------------


class TestMem0GraphLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_graph(self, monkeypatch, sample_session):
        from memory_arena.strategies.mem0g import Mem0GraphStrategy

        captured_config: dict = {}

        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": []}
        mock_memory.search.return_value = {"results": []}
        mock_memory.delete_all.return_value = None

        def _from_config(cfg):
            captured_config.update(cfg)
            return mock_memory

        mock_memory_cls = MagicMock()
        mock_memory_cls.from_config.side_effect = _from_config

        fake_mem0 = SimpleNamespace(Memory=mock_memory_cls)
        monkeypatch.setitem(sys.modules, "mem0", fake_mem0)

        fake_llm = _mock_llm_client()
        # mem0g.py is now its own module (was previously a subclass of Mem0Strategy).
        # v2 OSS removed graph_store, so mem0g pins mem0ai v1 and uses the v1 client surface.
        monkeypatch.setattr("memory_arena.strategies.mem0g.LLMClient", lambda: fake_llm)

        s = Mem0GraphStrategy()
        assert s.name == "mem0g"
        assert s._graph_enabled is True

        await s.setup("run-mem0g-test")
        assert "graph_store" in captured_config
        assert captured_config["graph_store"]["provider"] == "neo4j"

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)

        result = await s.recall("test query")
        assert isinstance(result, RecallResult)
        assert result.strategy == "mem0g"

        await s.teardown()


# ---------------------------------------------------------------------------
# GraphitiStrategy — lifecycle with mocked graphiti_core
# ---------------------------------------------------------------------------


class TestGraphitiLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session):
        from memory_arena.strategies.graphiti import GraphitiStrategy

        # Mock the graphiti client
        mock_client = MagicMock()
        mock_client.build_indices_and_constraints = AsyncMock(return_value=None)
        mock_client.add_episode = AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[]))

        # Mock search results: items with a session_id-like attribute
        fake_edge = SimpleNamespace(
            uuid="e1",
            fact="User works at fintech",
            valid_at=None,
            invalid_at=None,
            source_node_uuid="n1",
            target_node_uuid="n2",
            episodes=["session_03"],
        )
        mock_client.search = AsyncMock(return_value=[fake_edge])
        mock_client.close = AsyncMock(return_value=None)

        mock_graphiti_cls = MagicMock(return_value=mock_client)
        mock_openai_client_cls = MagicMock()
        mock_llm_config_cls = MagicMock()
        mock_episode_type = SimpleNamespace(message="message")

        fake_graphiti_core = SimpleNamespace(Graphiti=mock_graphiti_cls)
        fake_llm_client = SimpleNamespace(
            OpenAIClient=mock_openai_client_cls,
            config=SimpleNamespace(LLMConfig=mock_llm_config_cls),
        )
        fake_nodes = SimpleNamespace(EpisodeType=mock_episode_type)

        monkeypatch.setitem(sys.modules, "graphiti_core", fake_graphiti_core)
        monkeypatch.setitem(sys.modules, "graphiti_core.llm_client", fake_llm_client)
        monkeypatch.setitem(
            sys.modules,
            "graphiti_core.llm_client.config",
            SimpleNamespace(LLMConfig=mock_llm_config_cls),
        )
        monkeypatch.setitem(sys.modules, "graphiti_core.nodes", fake_nodes)

        fake_llm = _mock_llm_client()
        monkeypatch.setattr("memory_arena.strategies.graphiti.LLMClient", lambda: fake_llm)

        s = GraphitiStrategy()
        await s.setup("run-graphiti-test")
        assert s.run_id == "run-graphiti-test"

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id

        result = await s.recall("test query")
        assert isinstance(result, RecallResult)
        assert result.strategy == "graphiti"

        await s.teardown()


# ---------------------------------------------------------------------------
# GraphitiFalkorStrategy — same algorithm, FalkorDB engine
# ---------------------------------------------------------------------------


class TestGraphitiFalkorLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session):
        from memory_arena.strategies.graphiti_falkor import GraphitiFalkorStrategy

        mock_client = MagicMock()
        mock_client.build_indices_and_constraints = AsyncMock(return_value=None)
        mock_client.add_episode = AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[]))
        fake_edge = SimpleNamespace(
            uuid="e1",
            fact="User works at fintech",
            valid_at=None,
            invalid_at=None,
            source_node_uuid="n1",
            target_node_uuid="n2",
            episodes=["session_03"],
        )
        mock_client.search = AsyncMock(return_value=[fake_edge])
        mock_client.close = AsyncMock(return_value=None)

        mock_graphiti_cls = MagicMock(return_value=mock_client)
        mock_falkor_driver_cls = MagicMock()
        mock_openai_client_cls = MagicMock()
        mock_llm_config_cls = MagicMock()
        mock_episode_type = SimpleNamespace(message="message")

        # Native falkordb client used in teardown: FalkorDB(...).select_graph(name).delete()
        mock_graph = MagicMock()
        mock_falkordb_instance = MagicMock()
        mock_falkordb_instance.select_graph.return_value = mock_graph
        mock_falkordb_cls = MagicMock(return_value=mock_falkordb_instance)

        monkeypatch.setitem(
            sys.modules, "graphiti_core", SimpleNamespace(Graphiti=mock_graphiti_cls)
        )
        monkeypatch.setitem(sys.modules, "graphiti_core.driver", SimpleNamespace())
        monkeypatch.setitem(
            sys.modules,
            "graphiti_core.driver.falkordb_driver",
            SimpleNamespace(FalkorDriver=mock_falkor_driver_cls),
        )
        monkeypatch.setitem(
            sys.modules,
            "graphiti_core.llm_client",
            SimpleNamespace(OpenAIClient=mock_openai_client_cls),
        )
        monkeypatch.setitem(
            sys.modules,
            "graphiti_core.llm_client.config",
            SimpleNamespace(LLMConfig=mock_llm_config_cls),
        )
        monkeypatch.setitem(
            sys.modules, "graphiti_core.nodes", SimpleNamespace(EpisodeType=mock_episode_type)
        )
        monkeypatch.setitem(sys.modules, "falkordb", SimpleNamespace(FalkorDB=mock_falkordb_cls))

        fake_llm = _mock_llm_client()
        monkeypatch.setattr("memory_arena.strategies.graphiti_falkor.LLMClient", lambda: fake_llm)

        s = GraphitiFalkorStrategy()
        assert s.name == "graphiti_falkor"

        await s.setup("run-falkor-test")
        assert s.run_id == "run-falkor-test"
        # The graph driver, not the Neo4j uri/user/password path, is wired up.
        assert mock_falkor_driver_cls.called
        assert "graph_driver" in mock_graphiti_cls.call_args.kwargs

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id

        result = await s.recall("test query")
        assert isinstance(result, RecallResult)
        assert result.strategy == "graphiti_falkor"

        await s.teardown()
        assert s._client is None
        # Per-run graph dropped via the native client.
        mock_graph.delete.assert_called()


# ---------------------------------------------------------------------------
# CogneeStrategy — lifecycle with mocked cognee module
# ---------------------------------------------------------------------------


class TestCogneeLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session):
        # Build a minimal fake cognee module that satisfies the strategy's
        # interface. cognee 1.x exposes add/cognify/search and a config helper.
        fake_cognee = MagicMock()
        fake_cognee.add = AsyncMock(return_value=None)
        fake_cognee.cognify = AsyncMock(return_value=None)
        fake_cognee.search = AsyncMock(return_value=[{"text": "user is engineer"}])
        fake_cognee.prune = SimpleNamespace(
            prune_data=AsyncMock(return_value=None),
            prune_system=AsyncMock(return_value=None),
        )
        fake_cognee.config = SimpleNamespace(
            set_llm_provider=MagicMock(),
            set_llm_model=MagicMock(),
            set_llm_api_key=MagicMock(),
            set_embedding_dimensions=MagicMock(),
        )

        # Some cognee paths import SearchType from cognee.api.v1.search.types
        fake_search_types = SimpleNamespace(
            SearchType=SimpleNamespace(INSIGHTS="INSIGHTS", GRAPH_COMPLETION="GRAPH_COMPLETION")
        )

        monkeypatch.setitem(sys.modules, "cognee", fake_cognee)
        monkeypatch.setitem(sys.modules, "cognee.api", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "cognee.api.v1", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "cognee.api.v1.search", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "cognee.api.v1.search.types", fake_search_types)

        fake_llm = _mock_llm_client()
        from memory_arena.strategies.cognee import CogneeStrategy

        monkeypatch.setattr("memory_arena.strategies.cognee.LLMClient", lambda: fake_llm)

        s = CogneeStrategy()
        await s.setup("run-cognee-test")
        assert s.run_id == "run-cognee-test"

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id

        result = await s.recall("test query")
        assert isinstance(result, RecallResult)
        assert result.strategy == "cognee"

        await s.teardown()


# ---------------------------------------------------------------------------
# LangMemStrategy — lifecycle with mocked langmem + langgraph
# ---------------------------------------------------------------------------


class TestLangMemLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session):
        # Set the openai key so the strategy's gating check passes
        from memory_arena.settings import settings as _s

        old_key = _s.openai_api_key
        _s.openai_api_key = "test-key"

        try:
            mock_store = MagicMock()
            mock_store.asearch = AsyncMock(
                return_value=[
                    SimpleNamespace(
                        value={"content": "User is a software engineer."},
                        key="m1",
                    )
                ]
            )
            mock_store_cls = MagicMock(return_value=mock_store)

            mock_manager = MagicMock()
            mock_manager.ainvoke = AsyncMock(return_value=None)
            mock_create_manager = MagicMock(return_value=mock_manager)

            fake_langgraph_store_memory = SimpleNamespace(InMemoryStore=mock_store_cls)
            fake_langmem = SimpleNamespace(create_memory_store_manager=mock_create_manager)

            monkeypatch.setitem(sys.modules, "langgraph", SimpleNamespace())
            monkeypatch.setitem(sys.modules, "langgraph.store", SimpleNamespace())
            monkeypatch.setitem(sys.modules, "langgraph.store.memory", fake_langgraph_store_memory)
            monkeypatch.setitem(sys.modules, "langmem", fake_langmem)

            fake_llm = _mock_llm_client()
            from memory_arena.strategies.langmem import LangMemStrategy

            monkeypatch.setattr("memory_arena.strategies.langmem.LLMClient", lambda: fake_llm)

            s = LangMemStrategy()
            assert s.recall_at_k_measurable is False

            await s.setup("run-langmem-test")
            assert s.run_id == "run-langmem-test"

            rec = await s.ingest_session(sample_session)
            assert isinstance(rec, IngestRecord)

            result = await s.recall("test query")
            assert isinstance(result, RecallResult)
            assert result.strategy == "langmem"
            # LangMem doesn't carry session ids — supporting list should be empty
            assert result.supporting_session_ids == []

            await s.teardown()
        finally:
            _s.openai_api_key = old_key


# ---------------------------------------------------------------------------
# MemoriStrategy — lifecycle with mocked memori module
# ---------------------------------------------------------------------------


class TestMemoriLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session):
        # Build a fake memori module that satisfies the strategy
        mock_memori_instance = MagicMock()
        mock_memori_instance.new_session.return_value = None
        mock_memori_instance.set_session.return_value = None
        mock_memori_instance.augmentation = MagicMock()
        mock_memori_instance.augmentation.enqueue.return_value = None
        mock_memori_instance.recall = MagicMock(
            return_value=[{"content": "User is a software engineer."}]
        )
        mock_memori_instance.anthropic = MagicMock()
        mock_memori_instance.anthropic.register = MagicMock()

        mock_memori_cls = MagicMock(return_value=mock_memori_instance)

        # The strategy constructs ConversationMessage and AugmentationInput
        # inside ingest_session; provide simple stand-ins.
        fake_msg_mod = SimpleNamespace(
            ConversationMessage=lambda role, content: SimpleNamespace(role=role, content=content)
        )
        fake_input_mod = SimpleNamespace(AugmentationInput=lambda **kw: SimpleNamespace(**kw))

        monkeypatch.setitem(sys.modules, "memori", SimpleNamespace(Memori=mock_memori_cls))
        monkeypatch.setitem(sys.modules, "memori.memory", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "memori.memory.augmentation", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "memori.memory.augmentation._message", fake_msg_mod)
        monkeypatch.setitem(sys.modules, "memori.memory.augmentation.input", fake_input_mod)

        # Mock psycopg so the conn factory doesn't try a real connection
        fake_psycopg = SimpleNamespace(connect=MagicMock(return_value=MagicMock()))
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

        fake_llm = _mock_llm_client()
        from memory_arena.strategies.memori import MemoriStrategy

        monkeypatch.setattr("memory_arena.strategies.memori.LLMClient", lambda: fake_llm)

        s = MemoriStrategy()
        assert s.recall_at_k_measurable is False

        await s.setup("run-memori-test")
        assert s.run_id == "run-memori-test"

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)

        result = await s.recall("test query")
        assert isinstance(result, RecallResult)
        assert result.strategy == "memori"

        await s.teardown()


# ---------------------------------------------------------------------------
# KarpathyLlmWikiStrategy — pure-Python, only LLM is mocked
# ---------------------------------------------------------------------------


class TestQISSLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, mock_chroma_client, sample_session):
        from unittest.mock import patch

        from memory_arena.strategies.quantum.qiss import QISSStrategy

        # Make the wrapped naive_vector's recall return the canned chroma rows.
        mock_chroma_client.get_or_create_collection.return_value.query.return_value = {
            "documents": [["user: I work as a software engineer at a fintech startup"]],
            "metadatas": [[{"session_id": "session_03", "turn_id": "session_03_turn_001"}]],
            "distances": [[0.12]],
        }
        fake_llm = _mock_llm_client()
        monkeypatch.setattr("memory_arena.strategies.naive_vector.LLMClient", lambda: fake_llm)
        monkeypatch.setattr("memory_arena.strategies.quantum.qiss.LLMClient", lambda: fake_llm)

        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch(
                "memory_arena.strategies.embeddings.OpenAIEmbedding",
                return_value=MagicMock(),
            ):
                s = QISSStrategy()
                assert s.name == "qiss"
                await s.setup("run-qiss-test")
                assert s.run_id == "run-qiss-test"

                rec = await s.ingest_session(sample_session)
                assert isinstance(rec, IngestRecord)
                assert rec.session_id == sample_session.id

                result = await s.recall("What does the user do?")
                assert isinstance(result, RecallResult)
                assert result.strategy == "qiss"
                assert "session_03" in result.supporting_session_ids
                # Single-query fidelity == (1 - 0.12)^2.
                assert result.retrieved_memories[0]["score"] == pytest.approx(0.7744)

                await s.teardown()


class TestSQRLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, mock_chroma_client, sample_session):
        pytest.importorskip("qiskit_aer")
        from unittest.mock import patch

        import numpy as np

        from memory_arena.strategies.quantum.sqr import SQRStrategy

        rng = np.random.default_rng(0)
        collection = mock_chroma_client.get_or_create_collection.return_value
        # PCA basis is fit from collection.get; supply a small corpus sample.
        collection.get.return_value = {"embeddings": rng.standard_normal((24, 8)).tolist()}
        collection.query.return_value = {
            "documents": [["user: I work as a software engineer at a fintech startup"]],
            "metadatas": [[{"session_id": "session_03", "turn_id": "session_03_turn_001"}]],
            "distances": [[0.12]],
            "embeddings": [rng.standard_normal((1, 8)).tolist()],
        }
        fake_llm = _mock_llm_client()
        monkeypatch.setattr("memory_arena.strategies.naive_vector.LLMClient", lambda: fake_llm)
        monkeypatch.setattr("memory_arena.strategies.quantum.sqr.LLMClient", lambda: fake_llm)

        class _FakeEF:
            def __call__(self, inputs):
                return [rng.standard_normal(8).tolist() for _ in inputs]

        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch("memory_arena.strategies.embeddings.OpenAIEmbedding", _FakeEF):
                s = SQRStrategy()
                assert s.name == "sqr"
                s._n_qubits = 2
                s._target_dims = 4
                await s.setup("run-sqr-test")
                assert s.run_id == "run-sqr-test"

                rec = await s.ingest_session(sample_session)
                assert isinstance(rec, IngestRecord)
                assert rec.session_id == sample_session.id

                result = await s.recall("What does the user do?")
                assert isinstance(result, RecallResult)
                assert result.strategy == "sqr"
                assert "session_03" in result.supporting_session_ids
                assert s.pca_variance_explained is not None

                await s.teardown()


class TestKarpathyLlmWikiLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, monkeypatch, sample_session, tmp_path):
        from memory_arena.strategies.karpathy_llm_wiki import KarpathyLlmWikiStrategy

        # The ingest LLM call expects a JSON array of operations.
        ingest_response = LLMResponse(
            text=(
                '[{"op": "create", "page": "user-job", '
                '"content": "User is a software engineer at a fintech startup '
                '[session=session_03]"}]'
            ),
            input_tokens=120,
            output_tokens=40,
            cost_usd=0.001,
        )
        # The select stage returns a JSON list of page names.
        select_response = LLMResponse(
            text='["user-job"]', input_tokens=30, output_tokens=10, cost_usd=0.0001
        )
        # The answer stage returns an answer string.
        answer_response = LLMResponse(
            text="Software engineer at a fintech startup [session_03].",
            input_tokens=60,
            output_tokens=20,
            cost_usd=0.0003,
        )

        fake_llm = AsyncMock()
        fake_llm.generate = AsyncMock(
            side_effect=[ingest_response, select_response, answer_response]
        )

        monkeypatch.setattr("memory_arena.strategies.karpathy_llm_wiki.LLMClient", lambda: fake_llm)

        s = KarpathyLlmWikiStrategy()
        await s.setup("run-karpathy-test")
        assert s.run_id == "run-karpathy-test"
        assert s._wiki_dir is not None
        assert s._wiki_dir.exists()

        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id

        # After ingest, at least one page should exist
        existing_pages = list((s._wiki_dir / "pages").glob("*.md"))
        assert len(existing_pages) >= 1

        result = await s.recall("What does the user do?")
        assert isinstance(result, RecallResult)
        assert result.strategy == "karpathy_llm_wiki"

        await s.teardown()
