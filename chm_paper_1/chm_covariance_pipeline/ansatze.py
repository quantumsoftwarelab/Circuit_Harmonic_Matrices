"""PennyLane circuit definitions for the CHM covariance pipeline.

The module provides the JAX-compatible QNode used for direct Monte Carlo
validation of the exact two-copy twirling calculation.  The exact twirling
backend has its own Pauli-propagation implementation; this file is intentionally
kept as a small, public-facing circuit library rather than a general experiment
registry.

Circuit convention
------------------
For scalar input x and trainable parameters theta,

    f(x, theta) = <0| U(x, theta)^† O U(x, theta) |0>.

Each layer applies a data encoder followed by ``train_depth`` repetitions of the
chosen trainable block.  The flat parameter vector has length

    m = n_layers * train_depth * params_per_train_block(ansatz, n_qubits).

Supported ansatzes match the exact twirling backend in
``exact_covariance_twirled.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import jax
import pennylane as qml

Axis = Literal["RX", "RY", "RZ"]
ObsKind = Literal["OX", "OY", "OZ", "OZZ"]
AnsatzName = Literal[
    "YZY",
    "YZY_ENTANGLING",
    "HEA",
    "CIRCUIT_15",
    "CIRCUIT_16",
    "CIRCUIT_17",
    "CIRCUIT_18",
    "CIRCUIT_19",
    "CIRCUIT_ZZ",
    "CIRCUIT_MULTIOBS",
    "CIRCUIT_ENTFIRST",
]


# ---------------------------------------------------------------------------
# Elementary gates and observables
# ---------------------------------------------------------------------------


def _rot(axis: Axis, angle, wire: int) -> None:
    """Apply a single-qubit Pauli rotation."""
    if axis == "RX":
        qml.RX(angle, wires=wire)
    elif axis == "RY":
        qml.RY(angle, wires=wire)
    elif axis == "RZ":
        qml.RZ(angle, wires=wire)
    else:  # pragma: no cover - guarded by Literal annotations and CLI choices
        raise ValueError(f"Unknown rotation axis: {axis}")


def apply_encoder(x, n_qubits: int, encoder_axis: Axis, encoder_scale: float) -> None:
    """Apply the scalar data encoder exp[-i encoder_scale*x*P/2] to each qubit."""
    angle = encoder_scale * x
    for wire in range(n_qubits):
        _rot(encoder_axis, angle, wire)


def observable_zz(n_qubits: int):
    """Return the normalised two-body observable over all unordered Z_i Z_j pairs."""
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if not pairs:
        return qml.PauliZ(0)
    scale = 2.0 / float(n_qubits * (n_qubits - 1))
    terms = [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in pairs]
    return scale * qml.sum(*terms)


def observable_operator(n_qubits: int, obs_kind: ObsKind):
    """Return the normalised observable specified by ``obs_kind``."""
    if obs_kind == "OX":
        terms = [qml.PauliX(wire) for wire in range(n_qubits)]
    elif obs_kind == "OY":
        terms = [qml.PauliY(wire) for wire in range(n_qubits)]
    elif obs_kind == "OZ":
        terms = [qml.PauliZ(wire) for wire in range(n_qubits)]
    elif obs_kind == "OZZ":
        return observable_zz(n_qubits)
    else:  # pragma: no cover
        raise ValueError(f"Unknown observable kind: {obs_kind}")
    return (1.0 / float(n_qubits)) * qml.sum(*terms)


# ---------------------------------------------------------------------------
# Entangler patterns
# ---------------------------------------------------------------------------


def entangle_ring_two_sublayers(n_qubits: int) -> None:
    """Apply the two-sublayer ring CNOT pattern used by the HEA family."""
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    if n_qubits > 2:
        qml.CNOT(wires=[n_qubits - 1, 0])


def entangle_all_to_all_upper(n_qubits: int) -> None:
    """Apply the triangular all-to-all CNOT pattern CNOT(i -> j), i < j."""
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])


def _strobl_controlled_pairs(ansatz: str, n_qubits: int) -> list[tuple[int, int]]:
    """Return (control, target) pairs for circuits 16--19.

    The order is matched to the exact twirling backend.  For n=4 this gives
    C16/C17: (1,0), (3,2), (2,1), and C18/C19: (3,0), (2,3), (1,2), (0,1).
    """
    if n_qubits < 2:
        raise ValueError("CIRCUIT_16--19 require at least two qubits.")
    if ansatz in ("CIRCUIT_16", "CIRCUIT_17"):
        targets = [0] + list(range(n_qubits - 2, 0, -1))
        return [(target + 1, target) for target in targets]
    if ansatz in ("CIRCUIT_18", "CIRCUIT_19"):
        targets = [0] + list(range(n_qubits - 1, 0, -1))
        return [((target - 1) % n_qubits, target) for target in targets]
    raise ValueError(f"Unexpected ansatz for controlled rotations: {ansatz}")


def _apply_controlled_rotation(axis: Axis, angle, control: int, target: int) -> None:
    """Apply a trainable controlled RX or RZ rotation."""
    if axis == "RX":
        qml.CRX(angle, wires=[control, target])
    elif axis == "RZ":
        qml.CRZ(angle, wires=[control, target])
    else:  # pragma: no cover
        raise ValueError(f"Unsupported controlled-rotation axis: {axis}")


# ---------------------------------------------------------------------------
# Trainable blocks
# ---------------------------------------------------------------------------


def params_per_train_block(ansatz: AnsatzName, n_qubits: int) -> int:
    """Number of trainable parameters in one block of ``ansatz``."""
    if ansatz in ("YZY", "YZY_ENTANGLING", "HEA", "CIRCUIT_MULTIOBS", "CIRCUIT_ENTFIRST"):
        return 3 * n_qubits
    if ansatz == "CIRCUIT_15":
        return 2 * n_qubits
    if ansatz in ("CIRCUIT_16", "CIRCUIT_17"):
        return 2 * n_qubits + (n_qubits - 1)
    if ansatz in ("CIRCUIT_18", "CIRCUIT_19"):
        return 3 * n_qubits
    if ansatz == "CIRCUIT_ZZ":
        return 2 * n_qubits + n_qubits // 2
    raise ValueError(f"Unknown ansatz: {ansatz}")


def _apply_yzy(theta_block, n_qubits: int) -> None:
    idx = 0
    for wire in range(n_qubits):
        qml.RY(theta_block[idx + 0], wires=wire)
        qml.RZ(theta_block[idx + 1], wires=wire)
        qml.RY(theta_block[idx + 2], wires=wire)
        idx += 3


def apply_trainer_block(ansatz: AnsatzName, theta_block, n_qubits: int) -> None:
    """Apply one trainable block to the active PennyLane tape."""
    expected = params_per_train_block(ansatz, n_qubits)
    if int(theta_block.shape[0]) != expected:
        raise ValueError(
            f"theta_block has length {theta_block.shape[0]}, expected {expected} for {ansatz}."
        )

    if ansatz == "YZY":
        _apply_yzy(theta_block, n_qubits)
        return

    if ansatz == "YZY_ENTANGLING":
        _apply_yzy(theta_block, n_qubits)
        entangle_all_to_all_upper(n_qubits)
        return

    if ansatz == "HEA":
        _apply_yzy(theta_block, n_qubits)
        entangle_ring_two_sublayers(n_qubits)
        return

    if ansatz == "CIRCUIT_15":
        idx = 0
        for wire in range(n_qubits):
            qml.RY(theta_block[idx], wires=wire)
            idx += 1
        entangle_ring_two_sublayers(n_qubits)
        for wire in range(n_qubits):
            qml.RY(theta_block[idx], wires=wire)
            idx += 1
        entangle_ring_two_sublayers(n_qubits)
        return

    if ansatz in ("CIRCUIT_16", "CIRCUIT_17", "CIRCUIT_18", "CIRCUIT_19"):
        idx = 0
        for wire in range(n_qubits):
            qml.RX(theta_block[idx + 0], wires=wire)
            qml.RZ(theta_block[idx + 1], wires=wire)
            idx += 2
        ctrl_axis: Axis = "RZ" if ansatz in ("CIRCUIT_16", "CIRCUIT_18") else "RX"
        for control, target in _strobl_controlled_pairs(ansatz, n_qubits):
            _apply_controlled_rotation(ctrl_axis, theta_block[idx], control, target)
            idx += 1
        if idx != expected:
            raise RuntimeError(f"Internal parameter count mismatch for {ansatz}: {idx} != {expected}.")
        return

    if ansatz == "CIRCUIT_ZZ":
        idx = 0
        for wire in range(n_qubits):
            qml.RY(theta_block[idx + wire], wires=wire)
        idx += n_qubits
        for pair in range(n_qubits // 2):
            qml.IsingZZ(theta_block[idx + pair], wires=[2 * pair, 2 * pair + 1])
        idx += n_qubits // 2
        for wire in range(n_qubits):
            qml.RY(theta_block[idx + wire], wires=wire)
        return

    if ansatz == "CIRCUIT_MULTIOBS":
        _apply_yzy(theta_block, n_qubits)
        return

    if ansatz == "CIRCUIT_ENTFIRST":
        entangle_ring_two_sublayers(n_qubits)
        _apply_yzy(theta_block, n_qubits)
        return

    raise ValueError(f"Unhandled ansatz: {ansatz}")


# ---------------------------------------------------------------------------
# QNode factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitSpec:
    """Configuration for a scalar-input quantum model."""

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
    """Build ``qnode(x, theta)`` and return it with the flat parameter count."""
    p_block = params_per_train_block(spec.ansatz, spec.n_qubits)
    n_parameters = spec.n_layers * spec.train_depth * p_block
    device = qml.device(spec.device_name, wires=spec.n_qubits, shots=None)
    obs_kind = "OZZ" if spec.ansatz == "CIRCUIT_MULTIOBS" else spec.obs_kind
    observable = observable_operator(spec.n_qubits, obs_kind)

    def circuit(x, theta):
        theta = theta.reshape(spec.n_layers, spec.train_depth, p_block)
        for layer in range(spec.n_layers):
            apply_encoder(x, spec.n_qubits, spec.encoder_axis, spec.encoder_scale)
            for depth in range(spec.train_depth):
                apply_trainer_block(spec.ansatz, theta[layer, depth], spec.n_qubits)
        return qml.expval(observable)

    qnode = qml.QNode(circuit, device, interface="jax", diff_method=spec.diff_method)
    return (jax.jit(qnode) if spec.jit else qnode), n_parameters
