"""Honest diagnostics for the quantum rerankers.

Computes and persists the numbers the README/CHANGELOG quote, so no figure is
ever narrated from memory:

  1. PCA variance-explained vs n_qubits. The real cost of squeezing 3072-d
     text-embedding-3-large vectors into a 2^n-amplitude register, measured on a
     sample of the actual corpus embeddings.
  2. SWAP-test similarity error vs shots. Mean absolute error of the sampled
     SWAP-test estimator against the exact statevector value, by shot count.

Writes results/<corpus>_quantum_diagnostics.json.

Usage:
    python scripts/quantum_diagnostics.py [corpus]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

_SAMPLE_TURNS = 256
_N_QUBITS_GRID = [2, 3, 4, 5, 6, 7]
_SHOTS_GRID = [64, 256, 1024, 4096, 16384]
_SWAP_TRIALS = 64


class PCAVariancePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_qubits: int
    dims: int
    variance_explained: float = Field(ge=0.0, le=1.0)
    variance_lost: float = Field(ge=0.0, le=1.0)


class SwapErrorPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shots: int
    mean_abs_error: float = Field(ge=0.0, le=1.0)
    max_abs_error: float = Field(ge=0.0, le=1.0)


def _sample_corpus_embeddings(corpus: str) -> np.ndarray:
    from memory_arena.sessions.loaders import load_sessions
    from memory_arena.strategies.embeddings import OpenAIEmbedding

    sessions = load_sessions(corpus)
    texts: list[str] = []
    for sess in sessions:
        for turn in sess.turns:
            texts.append(f"{turn.role}: {turn.content}")
    # Deterministic stride sample so the figure is reproducible.
    if len(texts) > _SAMPLE_TURNS:
        idx = np.linspace(0, len(texts) - 1, _SAMPLE_TURNS).astype(int)
        texts = [texts[i] for i in idx]
    ef = OpenAIEmbedding()
    return np.asarray(ef(texts), dtype=float)


def pca_variance_curve(embeddings: np.ndarray) -> list[PCAVariancePoint]:
    from memory_arena.strategies.quantum.utils import PCAReducer, state_dim

    out: list[PCAVariancePoint] = []
    for n_qubits in _N_QUBITS_GRID:
        dims = state_dim(n_qubits)
        ve = float(PCAReducer(dims).fit(embeddings).variance_explained)
        out.append(
            PCAVariancePoint(
                n_qubits=n_qubits,
                dims=dims,
                variance_explained=round(ve, 4),
                variance_lost=round(1.0 - ve, 4),
            )
        )
    return out


def swap_error_curve() -> list[SwapErrorPoint]:
    from memory_arena.strategies.quantum.circuits import swap_test_fidelities
    from memory_arena.strategies.quantum.utils import amplitude_encode

    rng = np.random.default_rng(0)
    n_qubits = 4
    pairs = [
        (
            amplitude_encode(rng.standard_normal(16), n_qubits),
            amplitude_encode(rng.standard_normal(16), n_qubits),
        )
        for _ in range(_SWAP_TRIALS)
    ]
    exact = np.array([swap_test_fidelities(a, [b], n_qubits, shots=0)[0] for a, b in pairs])
    out: list[SwapErrorPoint] = []
    for shots in _SHOTS_GRID:
        sampled = np.array(
            [swap_test_fidelities(a, [b], n_qubits, shots=shots)[0] for a, b in pairs]
        )
        err = np.abs(sampled - exact)
        out.append(
            SwapErrorPoint(
                shots=shots,
                mean_abs_error=round(float(err.mean()), 4),
                max_abs_error=round(float(err.max()), 4),
            )
        )
    return out


def main(corpus: str) -> Path:
    print(f"Sampling corpus embeddings for {corpus} ...")
    embeddings = _sample_corpus_embeddings(corpus)
    print(f"  fit on {embeddings.shape[0]} turns x {embeddings.shape[1]} dims")

    pca = pca_variance_curve(embeddings)
    print("\nPCA variance-explained vs n_qubits:")
    for p in pca:
        print(
            f"  n_qubits={p.n_qubits} ({p.dims:>3} dims): "
            f"keeps {p.variance_explained:.1%}, loses {p.variance_lost:.1%}"
        )

    swap = swap_error_curve()
    print("\nSWAP-test error vs shots (4 qubits, statevector reference):")
    for s in swap:
        print(
            f"  shots={s.shots:>6}: mean|err|={s.mean_abs_error:.4f} max|err|={s.max_abs_error:.4f}"
        )

    out_path = RESULTS_DIR / f"{corpus}_quantum_diagnostics.json"
    payload = {
        "corpus": corpus,
        "embedding_model": "text-embedding-3-large",
        "embedding_dims": int(embeddings.shape[1]),
        "sample_turns": int(embeddings.shape[0]),
        "pca_variance_curve": [p.model_dump() for p in pca],
        "swap_error_vs_shots": [s.model_dump() for s in swap],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return out_path


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "longmemeval-s"
    main(corpus)
