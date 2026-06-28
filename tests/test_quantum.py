"""Tests for the quantum / quantum-inspired rerankers (qiss, sqr).

qiss is pure NumPy and always tested. sqr needs the optional ``[quantum]``
extra (qiskit + qiskit-aer); its tests are guarded with
``pytest.importorskip("qiskit_aer")`` so the core suite stays green without it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from memory_arena.llm.client import LLMResponse
from memory_arena.strategies.base import IngestRecord, RecallResult
from memory_arena.strategies.quantum.qiss import (
    QISSStrategy,
    fidelity_from_cosine_distance,
    fidelity_scores,
    quantum_fidelity,
    superpose,
    unit_normalize,
)


def _fake_llm():
    fake = AsyncMock()
    fake.generate = AsyncMock(
        return_value=LLMResponse(
            text="The user is a software engineer [session_03].",
            input_tokens=80,
            output_tokens=20,
            cost_usd=0.0003,
        )
    )
    fake.classify = AsyncMock(return_value="sub one\nsub two")
    return fake


# ---------------------------------------------------------------------------
# Pure-function math: the contract is fidelity == cos^2.
# ---------------------------------------------------------------------------


class TestQISSMath:
    def test_unit_normalize(self):
        v = unit_normalize(np.array([3.0, 4.0]))
        assert pytest.approx(float(np.linalg.norm(v))) == 1.0

    def test_unit_normalize_zero_passthrough(self):
        v = unit_normalize(np.zeros(3))
        assert float(np.linalg.norm(v)) == 0.0

    def test_fidelity_equals_cosine_squared(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.standard_normal(8)
            d = rng.standard_normal(8)
            cos = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d)))
            assert quantum_fidelity(q, d) == pytest.approx(cos * cos, abs=1e-9)

    def test_fidelity_in_unit_interval(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            q = rng.standard_normal(16)
            d = rng.standard_normal(16)
            f = quantum_fidelity(q, d)
            assert 0.0 <= f <= 1.0

    def test_identical_vectors_full_fidelity(self):
        v = np.array([1.0, 2.0, 3.0])
        assert quantum_fidelity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_zero_fidelity(self):
        assert quantum_fidelity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_distance_shortcut_matches_squared_cosine(self):
        # ChromaDB cosine distance = 1 - cos. Shortcut fidelity = (1 - dist)^2.
        for cos in (0.9, 0.5, 0.0, -0.3):
            dist = 1.0 - cos
            assert fidelity_from_cosine_distance(dist) == pytest.approx(cos * cos, abs=1e-9)

    def test_superpose_is_unit_state(self):
        q = superpose(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        assert pytest.approx(float(np.linalg.norm(q))) == 1.0

    def test_fidelity_scores_vectorized_matches_loop(self):
        rng = np.random.default_rng(2)
        query = rng.standard_normal(12)
        docs = rng.standard_normal((6, 12))
        vec = fidelity_scores(query, docs)
        loop = np.array([quantum_fidelity(query, d) for d in docs])
        assert np.allclose(vec, loop, atol=1e-9)
        assert vec.shape == (6,)
        assert vec.min() >= 0.0 and vec.max() <= 1.0

    def test_interference_differs_from_classical_average(self):
        # Superposition fusion carries i!=j interference cross-terms, so the
        # fused score is NOT the average of the per-sub-query fidelities. This
        # is what distinguishes it from rank-additive RRF.
        q1 = np.array([1.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0])
        d = unit_normalize(np.array([1.0, 1.0, 0.0]))
        fused = quantum_fidelity(superpose(np.vstack([q1, q2])), d)
        classical = 0.5 * (quantum_fidelity(q1, d) + quantum_fidelity(q2, d))
        assert abs(fused - classical) > 0.1


# ---------------------------------------------------------------------------
# recall() invariant: single-query QISS score == (1 - cosine_distance)^2 and
# the reranking preserves naive_vector's cosine ordering.
# ---------------------------------------------------------------------------


def _mock_collection(documents, metadatas, distances):
    col = MagicMock()
    col.query.return_value = {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
    return col


class TestQISSRecallInvariant:
    @pytest.mark.asyncio
    async def test_single_query_scores_are_cosine_squared(self):
        docs = ["a", "b", "c"]
        metas = [
            {"session_id": "s1", "turn_id": "t1"},
            {"session_id": "s2", "turn_id": "t2"},
            {"session_id": "s3", "turn_id": "t3"},
        ]
        distances = [0.1, 0.3, 0.2]  # cosine sims 0.9, 0.7, 0.8

        s = QISSStrategy()
        s.run_id = "test"
        s._vector._collection = _mock_collection(docs, metas, distances)
        s._llm = _fake_llm()
        s._decompose = False

        result = await s.recall("q", top_k=3)
        assert isinstance(result, RecallResult)
        assert result.strategy == "qiss"

        # Scores equal (1 - distance)^2 == cos^2.
        scored = {m["turn_id"]: m["score"] for m in result.retrieved_memories}
        assert scored["t1"] == pytest.approx(0.81)
        assert scored["t2"] == pytest.approx(0.49)
        assert scored["t3"] == pytest.approx(0.64)

        # Reranked order matches naive_vector's cosine order: t1 > t3 > t2.
        order = [m["turn_id"] for m in result.retrieved_memories]
        assert order == ["t1", "t3", "t2"]
        # Same order naive_vector (score = 1 - distance) would produce.
        naive_order = [
            metas[i]["turn_id"]
            for i in sorted(range(3), key=lambda i: 1.0 - distances[i], reverse=True)
        ]
        assert order == naive_order

    @pytest.mark.asyncio
    async def test_recall_top_k_truncates(self):
        docs = ["a", "b", "c", "d"]
        metas = [{"session_id": f"s{i}", "turn_id": f"t{i}"} for i in range(4)]
        distances = [0.4, 0.1, 0.3, 0.2]

        s = QISSStrategy()
        s.run_id = "test"
        s._vector._collection = _mock_collection(docs, metas, distances)
        s._llm = _fake_llm()
        result = await s.recall("q", top_k=2)
        assert len(result.retrieved_memories) == 2
        # Best two by cosine: t1 (0.1), t3 (0.2).
        assert [m["turn_id"] for m in result.retrieved_memories] == ["t1", "t3"]


class TestQISSDecompose:
    @pytest.mark.asyncio
    async def test_filters_refusals_and_advice_keeps_queries(self):
        # Haiku used to refuse first-person questions; the rewriter + filter must
        # drop refusal/advice lines and keep only real search-query fragments.
        s = QISSStrategy()
        fake = AsyncMock()
        fake.generate = AsyncMock(
            return_value=LLMResponse(
                text=(
                    "I don't have access to your project history\n"
                    "projects led\n"
                    "- project leadership role\n"
                    "check your email for the list\n"
                    "current projects managing\n"
                ),
                input_tokens=10,
                output_tokens=10,
                cost_usd=0.0,
            )
        )
        s._llm = fake
        subs = await s._decompose_query("How many projects have I led?")
        assert "projects led" in subs
        assert "project leadership role" in subs  # leading "- " stripped
        assert "current projects managing" in subs
        assert all("i don't" not in x.lower() for x in subs)
        assert all("check your" not in x.lower() for x in subs)
        assert "How many projects have I led?" in subs  # original always appended

    @pytest.mark.asyncio
    async def test_all_refusals_falls_back_to_query(self):
        s = QISSStrategy()
        fake = AsyncMock()
        fake.generate = AsyncMock(
            return_value=LLMResponse(text="I cannot help with that\nSorry", cost_usd=0.0)
        )
        s._llm = fake
        subs = await s._decompose_query("atomic question")
        assert subs == ["atomic question"]


class TestQISSLifecycle:
    @pytest.mark.asyncio
    async def test_ingest_delegates_to_vector(self, sample_session):
        s = QISSStrategy()
        s.run_id = "test"
        s._llm = _fake_llm()
        # Mock the wrapped vector so we don't hit chroma.
        s._vector.ingest_session = AsyncMock(
            return_value=IngestRecord(session_id=sample_session.id, facts_extracted=3)
        )
        rec = await s.ingest_session(sample_session)
        assert isinstance(rec, IngestRecord)
        assert rec.session_id == sample_session.id
        s._vector.ingest_session.assert_awaited_once()


# ---------------------------------------------------------------------------
# SQR utils: pure NumPy / sklearn, no qiskit needed.
# ---------------------------------------------------------------------------


class TestSQRUtils:
    def test_state_dim(self):
        from memory_arena.strategies.quantum.utils import state_dim

        assert state_dim(1) == 2
        assert state_dim(4) == 16

    def test_required_qubits(self):
        from memory_arena.strategies.quantum.utils import required_qubits

        assert required_qubits(16) == 4
        assert required_qubits(3072) == 12
        assert required_qubits(1) == 1

    def test_amplitude_encode_pads_and_normalizes(self):
        from memory_arena.strategies.quantum.utils import amplitude_encode

        amp = amplitude_encode(np.array([1.0, 1.0]), n_qubits=2)  # target dim 4
        assert amp.shape == (4,)
        assert pytest.approx(float(np.linalg.norm(amp))) == 1.0
        assert amp[2] == 0.0 and amp[3] == 0.0

    def test_amplitude_encode_truncates(self):
        from memory_arena.strategies.quantum.utils import amplitude_encode

        amp = amplitude_encode(np.arange(8.0), n_qubits=2)  # target dim 4
        assert amp.shape == (4,)
        assert pytest.approx(float(np.linalg.norm(amp))) == 1.0

    def test_amplitude_encode_zero_vector_is_basis_state(self):
        from memory_arena.strategies.quantum.utils import amplitude_encode

        amp = amplitude_encode(np.zeros(4), n_qubits=2)
        assert amp[0] == 1.0
        assert pytest.approx(float(np.linalg.norm(amp))) == 1.0

    def test_swap_test_fidelity_from_p0(self):
        from memory_arena.strategies.quantum.utils import swap_test_fidelity_from_p0

        assert swap_test_fidelity_from_p0(1.0) == pytest.approx(1.0)  # identical states
        assert swap_test_fidelity_from_p0(0.5) == pytest.approx(0.0)  # orthogonal states
        assert swap_test_fidelity_from_p0(0.4) == 0.0  # clamped below zero

    def test_pca_reducer_variance_and_shape(self):
        from memory_arena.strategies.quantum.utils import PCAReducer

        rng = np.random.default_rng(0)
        data = rng.standard_normal((40, 64))
        r = PCAReducer(16).fit(data)
        assert r.fitted_dim == 16
        assert r.input_dim == 64
        assert 0.0 < r.variance_explained <= 1.0
        assert r.transform(data[:3]).shape == (3, 16)


# ---------------------------------------------------------------------------
# SQR SWAP-test circuit: needs the [quantum] extra (qiskit + qiskit-aer).
# ---------------------------------------------------------------------------

qiskit_aer = pytest.importorskip("qiskit_aer")


class TestSQRSwapTest:
    def test_statevector_fidelity_matches_inner_product_squared(self):
        from memory_arena.strategies.quantum.circuits import swap_test_fidelities
        from memory_arena.strategies.quantum.utils import amplitude_encode

        rng = np.random.default_rng(7)
        for _ in range(8):
            a = amplitude_encode(rng.standard_normal(4), n_qubits=2)
            b = amplitude_encode(rng.standard_normal(4), n_qubits=2)
            expected = float(np.dot(a, b)) ** 2
            (fid,) = swap_test_fidelities(a, [b], n_qubits=2, shots=0)
            assert fid == pytest.approx(expected, abs=1e-6)

    def test_identical_states_fidelity_one(self):
        from memory_arena.strategies.quantum.circuits import swap_test_fidelities
        from memory_arena.strategies.quantum.utils import amplitude_encode

        a = amplitude_encode(np.array([0.3, 0.7, 0.1, 0.2]), n_qubits=2)
        (fid,) = swap_test_fidelities(a, [a], n_qubits=2, shots=0)
        assert fid == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_states_fidelity_zero(self):
        from memory_arena.strategies.quantum.circuits import swap_test_fidelities

        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0, 0.0])
        (fid,) = swap_test_fidelities(a, [b], n_qubits=2, shots=0)
        assert fid == pytest.approx(0.0, abs=1e-6)

    def test_batch_returns_one_score_per_doc(self):
        from memory_arena.strategies.quantum.circuits import swap_test_fidelities
        from memory_arena.strategies.quantum.utils import amplitude_encode

        rng = np.random.default_rng(11)
        q = amplitude_encode(rng.standard_normal(4), n_qubits=2)
        docs = [amplitude_encode(rng.standard_normal(4), n_qubits=2) for _ in range(5)]
        fids = swap_test_fidelities(q, docs, n_qubits=2, shots=0)
        assert len(fids) == 5
        assert all(0.0 <= f <= 1.0 for f in fids)

    def test_shots_mode_approximates_statevector(self):
        from memory_arena.strategies.quantum.circuits import swap_test_fidelities
        from memory_arena.strategies.quantum.utils import amplitude_encode

        rng = np.random.default_rng(3)
        a = amplitude_encode(rng.standard_normal(4), n_qubits=2)
        b = amplitude_encode(rng.standard_normal(4), n_qubits=2)
        (exact,) = swap_test_fidelities(a, [b], n_qubits=2, shots=0)
        (sampled,) = swap_test_fidelities(a, [b], n_qubits=2, shots=8192)
        assert abs(exact - sampled) < 0.1


class TestSQRRecall:
    @pytest.mark.asyncio
    async def test_recall_reranks_by_swap_fidelity(self):
        from memory_arena.strategies.quantum.sqr import SQRStrategy

        rng = np.random.default_rng(5)
        # Small corpus + small qubit count to keep the circuit tiny.
        corpus = rng.standard_normal((24, 8)).tolist()
        cand_embeddings = rng.standard_normal((4, 8)).tolist()
        docs = ["a", "b", "c", "d"]
        metas = [{"session_id": f"s{i}", "turn_id": f"t{i}"} for i in range(4)]
        distances = [0.2, 0.1, 0.4, 0.3]

        col = MagicMock()
        col.get.return_value = {"embeddings": corpus}
        col.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
            "embeddings": [cand_embeddings],
        }

        s = SQRStrategy()
        s.run_id = "test"
        s._n_qubits = 2
        s._target_dims = 4
        s._llm = _fake_llm()
        s._vector._collection = col

        # Patch the query embedding so no OpenAI call happens.
        import memory_arena.strategies.embeddings as emb_mod

        class _FakeEF:
            def __call__(self, inputs):
                return [rng.standard_normal(8).tolist() for _ in inputs]

        orig = emb_mod.OpenAIEmbedding
        emb_mod.OpenAIEmbedding = _FakeEF
        try:
            result = await s.recall("q", top_k=3)
        finally:
            emb_mod.OpenAIEmbedding = orig

        assert isinstance(result, RecallResult)
        assert result.strategy == "sqr"
        assert len(result.retrieved_memories) == 3
        # PCA variance got recorded as a fraction.
        assert s.pca_variance_explained is not None
        assert 0.0 < s.pca_variance_explained <= 1.0
        # Scores are valid fidelities.
        for m in result.retrieved_memories:
            assert 0.0 <= m["score"] <= 1.0
