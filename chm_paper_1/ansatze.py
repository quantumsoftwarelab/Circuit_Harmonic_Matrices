# ansatze.py
#
# Core circuit library for quantum harmonic analysis experiments.
#
# This module defines all variational quantum circuit ansatzes studied in the paper,
# following Strobl et al. Appendix A (Fig. 6). It also provides the data structure
# (CircuitSpec) and factory function (build_qnode) for constructing JAX-compatible
# PennyLane QNodes.
#
# Circuit structure:
#   f(x; θ) = <O>_{U(x,θ)|0>}
#   where each layer applies: encoder(x) -> trainer_block(θ_l,d)  (repeated n_layers times)
#   and the trainer block is repeated train_depth times per layer.
#
# Conventions:
#   - n_layers L:    repetitions of [encoder + trainer]
#   - train_depth d: repetitions of the trainer block within each layer
#   - theta is a flat vector of length m = L * d * p_block
#     where p_block depends on the chosen trainer ansatz (see params_per_train_block)
#   - Encoder rotation angle = encoder_scale * x, applied independently to every qubit
#   - Observable O = (1/n) sum_i sigma_i^{x/y/z}
#
# Supported ansatzes (AnsatzName):
#   YZY              — per-qubit RY-RZ-RY, no entanglement
#   YZY_ENTANGLING   — YZY followed by all-to-all (triangular) CNOT layer
#   HEA              — YZY followed by hardware-efficient ring CNOT layer
#   CIRCUIT_15       — two RY layers separated by ring entanglers (Strobl Fig. 6d)
#   CIRCUIT_16/17    — RX+RZ base + CZ entangler + sparse extras (Strobl Fig. 6e/f)
#   CIRCUIT_18/19    — RX+RZ base + CZ entangler + dense extras (Strobl Fig. 6g/h)
#
# Requirements: pennylane, jax

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
import pennylane as qml


# -------------------------
# Type aliases
# -------------------------

Axis = Literal["RX", "RY", "RZ"]
ObsKind = Literal["OX", "OY", "OZ"]
AnsatzName = Literal[
    "YZY",
    "YZY_ENTANGLING",
    "HEA",
    "CIRCUIT_15",
    "CIRCUIT_16",
    "CIRCUIT_17",
    "CIRCUIT_18",
    "CIRCUIT_19",
]


# -------------------------
# Low-level primitives
# -------------------------

def _rot(axis: Axis, angle, wire: int):
    if axis == "RX":
        qml.RX(angle, wires=wire)
    elif axis == "RY":
        qml.RY(angle, wires=wire)
    elif axis == "RZ":
        qml.RZ(angle, wires=wire)
    else:
        raise ValueError(f"Unknown axis: {axis}")


def apply_encoder(x, n_qubits: int, encoder_axis: Axis, encoder_scale: float):
    """Applies identical single-qubit encoder rotations on all qubits."""
    angle = encoder_scale * x
    for w in range(n_qubits):
        _rot(encoder_axis, angle, w)


def observable_operator(n_qubits: int, obs_kind: ObsKind):
    """Returns O = (1/n) sum_i sigma_i^{x/y/z} as a PennyLane Observable."""
    if obs_kind == "OX":
        terms = [qml.PauliX(w) for w in range(n_qubits)]
    elif obs_kind == "OY":
        terms = [qml.PauliY(w) for w in range(n_qubits)]
    elif obs_kind == "OZ":
        terms = [qml.PauliZ(w) for w in range(n_qubits)]
    else:
        raise ValueError(f"Unknown obs_kind: {obs_kind}")

    # PennyLane supports scalar multiplication of observables.
    return (1.0 / float(n_qubits)) * qml.sum(*terms)


def entangle_ring_two_sublayers(n_qubits: int):
    """
    Hardware-efficient (circular) entangler as in Strobl Fig. 6(c):
      even edges: (0->1), (2->3), ...
      odd  edges: (1->2), (3->4), ... plus wrap (n-1 -> 0) when n is even/odd accordingly.
    """
    # Even pairs
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

    # Odd pairs + wrap
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

    if n_qubits > 2:
        qml.CNOT(wires=[n_qubits - 1, 0])


def entangle_all_to_all_upper(n_qubits: int):
    """
    All-to-all CNOT pattern shown in Strobl_toggle Fig. 6(b) (triangular):
      for i<j: CNOT(i -> j)
    """
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])


def entangle_cz_even_odd_chain(n_qubits: int):
    """
    CZ entangler used for Circuits 16-19 (scalable interpretation consistent with Fig. 6):
      even CZ: (0-1), (2-3), ...
      odd  CZ: (1-2), (3-4), ... plus wrap (n-1 - 0)
    """
    for i in range(0, n_qubits - 1, 2):
        qml.CZ(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CZ(wires=[i, i + 1])
    if n_qubits > 2:
        qml.CZ(wires=[n_qubits - 1, 0])


# -------------------------
# Parameter counting
# -------------------------

def params_per_train_block(ansatz: AnsatzName, n_qubits: int) -> int:
    """
    p_block: number of trainable parameters consumed by ONE trainer block application.
    """
    if ansatz in ("YZY", "YZY_ENTANGLING", "HEA"):
        # per-qubit: RY, RZ, RY
        return 3 * n_qubits

    if ansatz == "CIRCUIT_15":
        # two RY layers (each on all qubits), with ring entanglers in between and at end
        return 2 * n_qubits

    if ansatz in ("CIRCUIT_16", "CIRCUIT_17"):
        # base: per-qubit RX+RZ = 2n
        # plus "sparse" extra rotations on alternating qubits: count = n-1 (matches n=4 -> 3 in Fig. 6)
        return 2 * n_qubits + (n_qubits - 1)

    if ansatz in ("CIRCUIT_18", "CIRCUIT_19"):
        # base 2n plus extra on all qubits => 3n (matches n=4 -> 12 in Fig. 6)
        return 3 * n_qubits

    raise ValueError(f"Unknown ansatz: {ansatz}")


# -------------------------
# Trainer blocks (Appendix A family)
# -------------------------

def apply_trainer_block(ansatz: AnsatzName, theta_block, n_qubits: int):
    """
    Apply ONE trainer block, consuming theta_block with shape (p_block,).
    """
    p = params_per_train_block(ansatz, n_qubits)
    if int(theta_block.shape[0]) != int(p):
        raise ValueError(f"theta_block has length {theta_block.shape[0]} but expected {p} for {ansatz}.")

    t = theta_block  # alias

    if ansatz == "YZY":
        # per qubit: RY(a), RZ(b), RY(c)
        idx = 0
        for w in range(n_qubits):
            qml.RY(t[idx + 0], wires=w)
            qml.RZ(t[idx + 1], wires=w)
            qml.RY(t[idx + 2], wires=w)
            idx += 3
        return

    if ansatz == "YZY_ENTANGLING":
        idx = 0
        for w in range(n_qubits):
            qml.RY(t[idx + 0], wires=w)
            qml.RZ(t[idx + 1], wires=w)
            qml.RY(t[idx + 2], wires=w)
            idx += 3
        # all-to-all (triangular) CNOT pattern
        entangle_all_to_all_upper(n_qubits)
        return

    if ansatz == "HEA":
        idx = 0
        for w in range(n_qubits):
            qml.RY(t[idx + 0], wires=w)
            qml.RZ(t[idx + 1], wires=w)
            qml.RY(t[idx + 2], wires=w)
            idx += 3
        entangle_ring_two_sublayers(n_qubits)
        return

    if ansatz == "CIRCUIT_15":
        # RY layer -> ring entangle -> RY layer -> ring entangle
        idx = 0
        for w in range(n_qubits):
            qml.RY(t[idx], wires=w)
            idx += 1
        entangle_ring_two_sublayers(n_qubits)

        for w in range(n_qubits):
            qml.RY(t[idx], wires=w)
            idx += 1
        entangle_ring_two_sublayers(n_qubits)
        return

    if ansatz in ("CIRCUIT_16", "CIRCUIT_17", "CIRCUIT_18", "CIRCUIT_19"):
        # Base: per-qubit RX + RZ with independent params
        idx = 0
        for w in range(n_qubits):
            qml.RX(t[idx + 0], wires=w)
            qml.RZ(t[idx + 1], wires=w)
            idx += 2

        # Entangle with CZ pattern (scalable interpretation)
        entangle_cz_even_odd_chain(n_qubits)

        # Extra single-qubit rotations:
        #   - 16/18 use RZ extras
        #   - 17/19 use RX extras
        extra_axis: Axis = "RZ" if ansatz in ("CIRCUIT_16", "CIRCUIT_18") else "RX"

        if ansatz in ("CIRCUIT_16", "CIRCUIT_17"):
            # Sparse extras on alternating wires excluding one endpoint (count n-1),
            # implemented as: even wires (0,2,4,...) then odd wires (1,3,5,...) excluding last.
            # This reproduces the n=4 parameter count and staggered layout in Fig. 6(e,f).
            # Count bookkeeping is handled by params_per_train_block.
            # Even indices
            for w in range(0, n_qubits, 2):
                if idx >= p:
                    break
                _rot(extra_axis, t[idx], w)
                idx += 1
            # Odd indices (skip the last wire to keep total = n-1)
            for w in range(1, n_qubits - 1, 2):
                if idx >= p:
                    break
                _rot(extra_axis, t[idx], w)
                idx += 1

        else:
            # 18/19: dense extras on all wires (count n), matching Fig. 6(g,h) total = 3n
            for w in range(n_qubits):
                _rot(extra_axis, t[idx], w)
                idx += 1

        return

    raise ValueError(f"Unhandled ansatz: {ansatz}")


# -------------------------
# QNode factory
# -------------------------

@dataclass(frozen=True)
class CircuitSpec:
    ansatz: AnsatzName
    n_qubits: int
    n_layers: int
    train_depth: int
    encoder_axis: Axis = "RX"
    encoder_scale: float = 1.0
    obs_kind: ObsKind = "OZ"
    device_name: str = "default.qubit"
    diff_method: str = "parameter-shift"
    jit: bool = True


def build_qnode(spec: CircuitSpec) -> Tuple[Callable, int]:
    """
    Returns:
      qnode(x, theta) with theta flat of length m
      m: total number of trainable parameters
    """
    p_block = params_per_train_block(spec.ansatz, spec.n_qubits)
    m = spec.n_layers * spec.train_depth * p_block

    dev = qml.device(spec.device_name, wires=spec.n_qubits, shots=None)
    obs = observable_operator(spec.n_qubits, spec.obs_kind)

    def circuit(x, theta):
        theta = theta.reshape(spec.n_layers, spec.train_depth, p_block)

        for l in range(spec.n_layers):
            apply_encoder(x, spec.n_qubits, spec.encoder_axis, spec.encoder_scale)

            for d in range(spec.train_depth):
                apply_trainer_block(spec.ansatz, theta[l, d], spec.n_qubits)

        return qml.expval(obs)

    qnode = qml.QNode(circuit, dev, interface="jax", diff_method=spec.diff_method)
    return (jax.jit(qnode) if spec.jit else qnode), m