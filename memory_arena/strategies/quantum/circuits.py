"""SWAP-test circuit construction + batched execution on the Qiskit Aer simulator.

The SWAP test estimates the state overlap |<q|d>|^2 between two amplitude-encoded
registers. With an ancilla prepared in (|0> + |1>)/sqrt(2), n controlled-SWAPs,
and a final Hadamard, the ancilla measures 0 with probability

    P(0) = (1 + |<q|d>|^2) / 2   =>   |<q|d>|^2 = 2 * P(0) - 1

Statevector mode reads P(0) exactly from the simulated state (the benchmark
default, noiseless and reproducible). Shots mode samples it, trading accuracy
for speed; that is the knob behind the error-vs-shots calibration curve.

This module imports qiskit / qiskit_aer at top level on purpose: the strategy
registry's _try_import drops sqr cleanly when the optional [quantum] extra is
absent, exactly like the vendor SDK gates.
"""

from __future__ import annotations

import numpy as np
import qiskit_aer  # noqa: F401  (registers save_statevector on QuantumCircuit)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from memory_arena.strategies.quantum.utils import swap_test_fidelity_from_p0


def build_swap_test_circuit(
    q_amp: np.ndarray,
    d_amp: np.ndarray,
    n_qubits: int,
    measure: bool = False,
) -> QuantumCircuit:
    """Build a SWAP-test circuit comparing two amplitude-encoded registers.

    Layout: ancilla = qubit 0, query register = qubits [1..n], doc register =
    qubits [n+1..2n]. The ancilla-first ordering means its marginal is
    ``probabilities([0])`` in the statevector path.

    measure=False appends save_statevector() (statevector/exact path).
    measure=True measures the ancilla into a 1-bit classical register (shots path).
    """
    anc = QuantumRegister(1, "anc")
    qr = QuantumRegister(n_qubits, "q")
    dr = QuantumRegister(n_qubits, "d")
    if measure:
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(anc, qr, dr, cr)
    else:
        qc = QuantumCircuit(anc, qr, dr)

    qc.initialize(list(q_amp), qr)
    qc.initialize(list(d_amp), dr)
    qc.h(anc[0])
    for i in range(n_qubits):
        qc.cswap(anc[0], qr[i], dr[i])
    qc.h(anc[0])

    if measure:
        qc.measure(anc[0], cr[0])
    else:
        qc.save_statevector()
    return qc


def _p0_from_statevector(state) -> float:
    """Marginal P(ancilla=0) from a simulated statevector (ancilla is qubit 0)."""
    probs = Statevector(state).probabilities([0])
    return float(probs[0])


def swap_test_fidelities(
    q_amp: np.ndarray,
    doc_amps: list[np.ndarray],
    n_qubits: int,
    shots: int = 0,
) -> list[float]:
    """Fidelity |<q|d>|^2 for one query vs many docs, all in a single Aer job.

    shots=0 -> exact statevector (default). shots>0 -> sampled estimate.
    Circuits are built once, transpiled once, and submitted as a single batch
    (per-circuit run() calls are far too slow for benchmark scale).
    """
    if not doc_amps:
        return []

    measure = shots > 0
    raw = [build_swap_test_circuit(q_amp, d, n_qubits, measure=measure) for d in doc_amps]

    if measure:
        sim = AerSimulator()
        circuits = transpile(raw, sim)
        result = sim.run(circuits, shots=shots).result()
        out: list[float] = []
        for i in range(len(circuits)):
            counts = result.get_counts(i)
            total = sum(counts.values()) or 1
            p0 = counts.get("0", 0) / total
            out.append(swap_test_fidelity_from_p0(p0))
        return out

    sim = AerSimulator(method="statevector")
    circuits = transpile(raw, sim)
    result = sim.run(circuits).result()
    out = []
    for i in range(len(circuits)):
        state = result.data(i)["statevector"]
        out.append(swap_test_fidelity_from_p0(_p0_from_statevector(state)))
    return out
