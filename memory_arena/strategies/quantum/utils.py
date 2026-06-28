"""Helpers for the SWAP-test reranker (sqr): PCA reduction, unit-normalization,
amplitude encoding, and qubit-count arithmetic.

text-embedding-3-large is 3072-d. A SWAP test on n qubits per register can only
hold 2^n amplitudes, so the embeddings must be projected down first. At
n_qubits=4 that is 3072 -> 16 dims, a large reduction whose variance loss is
tracked honestly via :attr:`PCAReducer.variance_explained` (it is bigger than
the kb-arena 768-d case for the same target dim).
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA


def state_dim(n_qubits: int) -> int:
    """Amplitude count a register of ``n_qubits`` can encode: 2^n_qubits."""
    return 2**n_qubits


def required_qubits(dim: int) -> int:
    """Minimum qubits whose 2^n state space holds ``dim`` amplitudes."""
    if dim <= 1:
        return 1
    return max(1, math.ceil(math.log2(dim)))


def unit_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale ``vec`` to unit L2 norm. A zero vector passes through unchanged."""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def amplitude_encode(vec: np.ndarray, n_qubits: int) -> np.ndarray:
    """Project ``vec`` to a normalized 2^n_qubits amplitude vector for initialize().

    Pads with zeros (or truncates) to exactly 2^n_qubits entries, then
    unit-normalizes so it is a valid quantum statevector. If the input is all
    zeros, returns a basis state |0...0> so qc.initialize() never sees a
    zero-norm vector.
    """
    target = state_dim(n_qubits)
    arr = np.asarray(vec, dtype=float).ravel()
    if arr.shape[0] < target:
        arr = np.pad(arr, (0, target - arr.shape[0]))
    else:
        arr = arr[:target]
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        arr = arr.copy()
        arr[0] = 1.0
        return arr
    return arr / norm


def swap_test_fidelity_from_p0(p0: float) -> float:
    """Recover |<q|d>|^2 from the SWAP-test ancilla P(0): fidelity = 2*P0 - 1.

    Clamped to [0, 1]; sampling noise can push the raw estimate slightly out of
    range.
    """
    fid = 2.0 * float(p0) - 1.0
    return float(min(1.0, max(0.0, fid)))


class PCAReducer:
    """Fit-once PCA projector from embedding space to 2^n_qubits dims.

    ``variance_explained`` is the honest fraction of variance the projection
    keeps, the headline cost of squeezing 3072-d embeddings into a few qubits.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self._pca: PCA | None = None
        self.variance_explained: float | None = None
        self.fitted_dim: int | None = None
        self.input_dim: int | None = None

    def fit(self, x: np.ndarray) -> PCAReducer:
        mat = np.asarray(x, dtype=float)
        if mat.ndim == 1:
            mat = mat[None, :]
        self.input_dim = int(mat.shape[1])
        # PCA can't ask for more components than samples or features.
        n = min(self.n_components, mat.shape[0], mat.shape[1])
        n = max(1, n)
        self._pca = PCA(n_components=n)
        self._pca.fit(mat)
        self.variance_explained = float(self._pca.explained_variance_ratio_.sum())
        self.fitted_dim = n
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAReducer.transform called before fit")
        mat = np.asarray(x, dtype=float)
        if mat.ndim == 1:
            mat = mat[None, :]
        return self._pca.transform(mat)
