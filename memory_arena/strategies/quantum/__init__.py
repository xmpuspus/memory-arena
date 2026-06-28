"""Quantum and quantum-inspired rerankers over the naive_vector store.

Two strategies live here, both coarse-retrieve through NaiveVectorStrategy's
ChromaDB collection and then rerank the candidates by a quantum-state fidelity:

  qiss  Quantum-Inspired Semantic Similarity. Pure NumPy, zero new deps.
        Reranks by quantum fidelity Tr(rho_q . rho_d) = |<q|d>|^2 = cos^2 over
        the same OpenAI embeddings naive_vector uses. Optional multi-query
        superposition fusion (off by default) adds interference cross-terms.

  sqr   Simulated Quantum Reranker. Runs an actual SWAP-test circuit on the
        Qiskit Aer simulator (statevector by default, shots optional). Requires
        the optional ``[quantum]`` extra (qiskit + qiskit-aer).

Both share naive_vector's Chroma index (prefixed collection name) rather than a
parallel index, so the retrieval substrate is identical and only the reranking
math differs.
"""

from __future__ import annotations

from memory_arena.strategies.quantum.qiss import QISSStrategy

__all__ = ["QISSStrategy"]
