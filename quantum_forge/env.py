"""Gymnasium environment for hardware-aware quantum circuit synthesis.

An agent builds a quantum circuit one gate at a time. The reward is the
fidelity between the circuit's output state and a target state, minus a
weighted cost for gate count, depth, two-qubit gates, hardware violations
and estimated execution time.

Two simulation backends are supported: ``qiskit`` (default) and ``cirq``.
Both expose the same observation, action and reward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

#: Operation index -> gate name. Indices 4..8 are two-qubit gates.
OPERATIONS: Tuple[str, ...] = ("x", "z", "h", "ry", "cx", "cz", "rxx", "rzz", "cp", "u")
TWO_QUBIT_OPS: frozenset = frozenset({4, 5, 6, 7, 8})

TargetSpec = Union[str, np.ndarray]


@dataclass
class HardwareConstraints:
    """Description of the target device.

    Attributes:
        connectivity_map: adjacency list of qubits. Treated as undirected.
        gate_times: gate name -> duration (arbitrary units, e.g. ns).
        error_rates: (q1, q2) -> two-qubit gate error rate.
        max_parallel_gates: maximum number of gates in one layer. ``None``
            means unlimited.
    """

    connectivity_map: Dict[int, List[int]]
    gate_times: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[Tuple[int, int], float] = field(default_factory=dict)
    max_parallel_gates: Optional[int] = None

    def connected(self, q1: int, q2: int) -> bool:
        return q2 in self.connectivity_map.get(q1, ()) or q1 in self.connectivity_map.get(q2, ())


class _QiskitBackend:
    """Thin wrapper around a qiskit ``QuantumCircuit``."""

    name = "qiskit"

    def __init__(self, num_qubits: int, noise_level: float):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        self._QuantumCircuit = QuantumCircuit
        self._Statevector = Statevector
        self.num_qubits = num_qubits
        self.noise_level = noise_level
        self.circuit = QuantumCircuit(num_qubits)
        self._noisy_sim = None
        if noise_level > 0:
            self._noisy_sim = self._build_noisy_simulator(noise_level)

    @staticmethod
    def _build_noisy_simulator(noise_level: float):
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "noise_level > 0 with the qiskit backend requires qiskit-aer. "
                "Install it with `pip install quantum-forge[aer]`."
            ) from exc
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(noise_level, 1), ["x", "z", "h", "ry", "u"]
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(min(1.0, 10 * noise_level), 2), ["cx", "cz", "rxx", "rzz", "cp"]
        )
        return AerSimulator(method="density_matrix", noise_model=noise_model)

    def reset(self) -> None:
        self.circuit = self._QuantumCircuit(self.num_qubits)

    def apply(self, op: int, q1: int, q2: int, p1: float, p2: float) -> None:
        c = self.circuit
        if op == 0:
            c.x(q1)
        elif op == 1:
            c.z(q1)
        elif op == 2:
            c.h(q1)
        elif op == 3:
            c.ry(p1, q1)
        elif op == 4:
            c.cx(q1, q2)
        elif op == 5:
            c.cz(q1, q2)
        elif op == 6:
            c.rxx(p1, q1, q2)
        elif op == 7:
            c.rzz(p1, q1, q2)
        elif op == 8:
            c.cp(p1, q1, q2)
        elif op == 9:
            c.u(p1, p2, 0.0, q1)

    def statevector(self) -> np.ndarray:
        return np.asarray(self._Statevector(self.circuit).data, dtype=np.complex128)

    def fidelity(self, target: np.ndarray) -> float:
        if self._noisy_sim is None:
            return float(abs(np.vdot(target, self.statevector())) ** 2)
        from qiskit import transpile
        from qiskit.quantum_info import DensityMatrix, state_fidelity

        circ = self.circuit.copy()
        circ.save_density_matrix()
        result = self._noisy_sim.run(transpile(circ, self._noisy_sim)).result()
        dm = DensityMatrix(result.data(0)["density_matrix"])
        return float(state_fidelity(dm, self._Statevector(target), validate=False))

    def gates(self) -> List[Tuple[str, Tuple[int, ...]]]:
        c = self.circuit
        return [
            (inst.operation.name, tuple(c.find_bit(q).index for q in inst.qubits))
            for inst in c.data
        ]

    def text(self) -> str:
        return str(self.circuit.draw(output="text"))


class _CirqBackend:
    """Thin wrapper around a ``cirq.Circuit`` on ``LineQubit``s."""

    name = "cirq"

    def __init__(self, num_qubits: int, noise_level: float):
        try:
            import cirq
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "backend='cirq' requires cirq. Install it with `pip install quantum-forge[cirq]`."
            ) from exc
        self._cirq = cirq
        self.num_qubits = num_qubits
        self.noise_level = noise_level
        self.qubits = cirq.LineQubit.range(num_qubits)
        # cirq is big-endian (first qubit in qubit_order is most significant); qiskit is
        # little-endian. Simulating with reversed order gives qiskit-compatible vectors.
        self._order = list(reversed(self.qubits))
        self.circuit = cirq.Circuit()
        self._sim = cirq.Simulator()
        self._noisy_sim = (
            cirq.DensityMatrixSimulator(noise=cirq.depolarize(noise_level)) if noise_level > 0 else None
        )

    def reset(self) -> None:
        self.circuit = self._cirq.Circuit()

    def apply(self, op: int, q1: int, q2: int, p1: float, p2: float) -> None:
        cirq = self._cirq
        a, b = self.qubits[q1], self.qubits[q2]
        if op == 0:
            gate = cirq.X(a)
        elif op == 1:
            gate = cirq.Z(a)
        elif op == 2:
            gate = cirq.H(a)
        elif op == 3:
            gate = cirq.ry(p1).on(a)
        elif op == 4:
            gate = cirq.CNOT(a, b)
        elif op == 5:
            gate = cirq.CZ(a, b)
        elif op == 6:
            gate = cirq.XXPowGate(exponent=p1 / np.pi, global_shift=-0.5).on(a, b)
        elif op == 7:
            gate = cirq.ZZPowGate(exponent=p1 / np.pi, global_shift=-0.5).on(a, b)
        elif op == 8:
            gate = cirq.CZPowGate(exponent=p1 / np.pi).on(a, b)
        elif op == 9:
            gate = cirq.MatrixGate(_u3_matrix(p1, p2, 0.0), name="u").on(a)
        else:
            return
        self.circuit.append(gate)

    def statevector(self) -> np.ndarray:
        result = self._sim.simulate(self.circuit, qubit_order=self._order)
        return np.asarray(result.final_state_vector, dtype=np.complex128)

    def fidelity(self, target: np.ndarray) -> float:
        if self._noisy_sim is None:
            return float(abs(np.vdot(target, self.statevector())) ** 2)
        result = self._noisy_sim.simulate(self.circuit, qubit_order=self._order)
        dm = result.final_density_matrix
        return float(
            np.real(self._cirq.fidelity(dm, target, qid_shape=(2,) * self.num_qubits))
        )

    def gates(self) -> List[Tuple[str, Tuple[int, ...]]]:
        out = []
        for op in self.circuit.all_operations():
            name = _cirq_gate_name(self._cirq, op.gate)
            out.append((name, tuple(q.x for q in op.qubits)))
        return out

    def text(self) -> str:
        return str(self.circuit)


def _u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Qiskit's U(theta, phi, lambda) gate as a 2x2 unitary."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=np.complex128,
    )


def _cirq_gate_name(cirq, gate) -> str:
    """Map a cirq gate onto the qiskit-style names used in ``OPERATIONS``."""
    if isinstance(gate, cirq.MatrixGate):
        return "u"  # the only MatrixGate this backend emits
    if gate == cirq.X:
        return "x"
    if gate == cirq.Z:
        return "z"
    if gate == cirq.H:
        return "h"
    if gate == cirq.CNOT:
        return "cx"
    if gate == cirq.CZ:
        return "cz"
    if isinstance(gate, cirq.Ry):
        return "ry"
    if isinstance(gate, cirq.Rz):
        return "rz"
    if isinstance(gate, cirq.XXPowGate):
        return "rxx"
    if isinstance(gate, cirq.ZZPowGate):
        return "rzz"
    if isinstance(gate, cirq.CZPowGate):
        return "cp"
    return str(gate).lower()


class QuantumForgeEnv(gym.Env):
    """Build a circuit gate by gate to prepare a target state.

    Args:
        num_qubits: number of qubits in the circuit.
        backend: ``"qiskit"`` or ``"cirq"``.
        noise_level: depolarizing probability per gate. ``0`` disables noise
            and uses exact statevector simulation. Non-zero values require
            ``qiskit-aer`` for the qiskit backend.
        hardware_constraints: optional device description. Two-qubit gates on
            non-adjacent qubits are rejected and penalised.
        target: ``"random"`` (Haar-random state drawn on each reset),
            ``"ghz"`` or an explicit normalised statevector.
        max_steps: episode truncation length.
        success_threshold: fidelity at which the episode terminates.
        max_violations: number of rejected hardware-violating actions after
            which the episode terminates.
        cost_weights: per-metric weights applied in the reward's cost term.
        render_mode: ``"human"`` prints the circuit, ``"rgb_array"`` returns
            a bar chart of basis-state probabilities (requires matplotlib).

    Action: ``Box(5)`` = ``(operation, qubit1, qubit2, param1, param2)``.
    Continuous entries are rounded and clipped into range, so the environment
    can be driven by any continuous-control agent.

    Observation: real and imaginary parts of the current statevector, followed
    (when hardware constraints are given) by one busy flag per qubit for the
    most recent circuit layer.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    DEFAULT_COST_WEIGHTS = {
        "gate_count": 1.0,
        "circuit_depth": 0.5,
        "two_qubit_gate_count": 2.0,
        "hardware_violations": 1.5,
        "execution_time": 1.0,
    }

    def __init__(
        self,
        num_qubits: int = 3,
        backend: str = "qiskit",
        noise_level: float = 0.0,
        hardware_constraints: Optional[HardwareConstraints] = None,
        target: TargetSpec = "random",
        max_steps: int = 30,
        success_threshold: float = 0.99,
        max_violations: int = 5,
        cost_weights: Optional[Dict[str, float]] = None,
        cost_scale: float = 0.01,
        render_mode: Optional[str] = None,
        optimization_metric: str = "weighted",
    ):
        super().__init__()
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        backend = backend.lower()
        if backend not in ("qiskit", "cirq"):
            raise ValueError("backend must be either 'qiskit' or 'cirq'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.num_qubits = num_qubits
        self.backend_name = backend
        self.noise_level = float(noise_level)
        self.hardware_constraints = hardware_constraints
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.max_violations = max_violations
        self.cost_weights = {**self.DEFAULT_COST_WEIGHTS, **(cost_weights or {})}
        self.cost_scale = cost_scale
        self.render_mode = render_mode
        self.optimization_metric = optimization_metric
        self._target_spec = target
        self._validate_target_spec()

        self.dim = 2**num_qubits
        n = num_qubits
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, -np.pi, -np.pi], dtype=np.float32),
            high=np.array([len(OPERATIONS) - 1, n - 1, n - 1, np.pi, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 2 * self.dim + (n if hardware_constraints else 0)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._backend = (
            _QiskitBackend(n, self.noise_level) if backend == "qiskit" else _CirqBackend(n, self.noise_level)
        )

        self.target_state: Optional[np.ndarray] = None
        self._layers: List[List[Set[int]]] = []  # each layer is a list of qubit-sets
        self._step_count = 0
        self._violations = 0
        self._execution_time = 0.0
        self._last_fidelity = 0.0

    # ------------------------------------------------------------------ setup

    def _validate_target_spec(self) -> None:
        spec = self._target_spec
        if isinstance(spec, str):
            if spec not in ("random", "ghz"):
                raise ValueError("target must be 'random', 'ghz' or a statevector")
            return
        arr = np.asarray(spec, dtype=np.complex128)
        if arr.shape != (2**self.num_qubits,):
            raise ValueError(f"target statevector must have shape ({2**self.num_qubits},)")
        if not np.isclose(np.linalg.norm(arr), 1.0):
            raise ValueError("target statevector must be normalised")
        self._target_spec = arr

    def _make_target(self) -> np.ndarray:
        spec = self._target_spec
        if isinstance(spec, np.ndarray):
            return spec.copy()
        if spec == "ghz":
            state = np.zeros(self.dim, dtype=np.complex128)
            state[0] = state[-1] = 1 / np.sqrt(2)
            return state
        z = self.np_random.normal(size=self.dim) + 1j * self.np_random.normal(size=self.dim)
        return z / np.linalg.norm(z)

    @property
    def circuit(self):
        """The underlying backend circuit (``QuantumCircuit`` or ``cirq.Circuit``)."""
        return self._backend.circuit

    # -------------------------------------------------------------- gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._backend.reset()
        self.target_state = self._make_target()
        self._layers = []
        self._step_count = 0
        self._violations = 0
        self._execution_time = 0.0
        self._last_fidelity = self._backend.fidelity(self.target_state)
        return self._get_observation(), self._get_info()

    def step(self, action):
        op, q1, q2, p1, p2 = self._decode_action(action)
        self._step_count += 1

        if not self._hardware_allows(op, q1, q2):
            self._violations += 1
            terminated = self._violations > self.max_violations
            truncated = not terminated and self._step_count >= self.max_steps
            info = self._get_info()
            info["error"] = "hardware constraints violated"
            return self._get_observation(), -1.0, terminated, truncated, info

        self._backend.apply(op, q1, q2, p1, p2)
        self._schedule(op, q1, q2)
        if self.hardware_constraints is not None:
            self._execution_time += self.hardware_constraints.gate_times.get(OPERATIONS[op], 0.0)

        self._last_fidelity = self._backend.fidelity(self.target_state)
        reward = self._last_fidelity - self.cost_scale * self._cost()
        terminated = self._last_fidelity >= self.success_threshold
        truncated = not terminated and self._step_count >= self.max_steps
        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            print(self._backend.text())
            return None
        if self.render_mode == "rgb_array":
            return self._render_state()
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------- internals

    def _decode_action(self, action) -> Tuple[int, int, int, float, float]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != 5:
            raise ValueError("action must have 5 components")
        n = self.num_qubits
        op = int(np.clip(np.rint(a[0]), 0, len(OPERATIONS) - 1))
        q1 = int(np.clip(np.rint(a[1]), 0, n - 1))
        q2 = int(np.clip(np.rint(a[2]), 0, n - 1))
        if op in TWO_QUBIT_OPS and q1 == q2:
            q2 = (q1 + 1) % n if n > 1 else q1
        p1 = float(np.clip(a[3], -np.pi, np.pi))
        p2 = float(np.clip(a[4], -np.pi, np.pi))
        return op, q1, q2, p1, p2

    def _hardware_allows(self, op: int, q1: int, q2: int) -> bool:
        hc = self.hardware_constraints
        if hc is None:
            return True
        if op in TWO_QUBIT_OPS:
            if self.num_qubits < 2:
                return False
            return hc.connected(q1, q2)
        return True

    def _schedule(self, op: int, q1: int, q2: int) -> None:
        """As-soon-as-possible layering that respects ``max_parallel_gates``."""
        qubits = {q1, q2} if op in TWO_QUBIT_OPS else {q1}
        last_touch = -1
        for i, layer in enumerate(self._layers):
            if any(qubits & g for g in layer):
                last_touch = i
        limit = None
        if self.hardware_constraints is not None:
            limit = self.hardware_constraints.max_parallel_gates
        for j in range(last_touch + 1, len(self._layers)):
            if limit is None or len(self._layers[j]) < limit:
                self._layers[j].append(qubits)
                return
        self._layers.append([qubits])

    def _gate_stats(self) -> Tuple[int, int]:
        gates = self._backend.gates()
        two_q = sum(1 for _, qs in gates if len(qs) == 2)
        return len(gates), two_q

    def _cost(self) -> float:
        if self.optimization_metric != "weighted":
            return 0.0
        gate_count, two_q = self._gate_stats()
        w = self.cost_weights
        return (
            w["gate_count"] * gate_count
            + w["circuit_depth"] * len(self._layers)
            + w["two_qubit_gate_count"] * two_q
            + w["hardware_violations"] * self._violations
            + w["execution_time"] * self._execution_time
        )

    def _busy_flags(self) -> np.ndarray:
        flags = np.zeros(self.num_qubits, dtype=np.float32)
        if self._layers:
            for qs in self._layers[-1]:
                for q in qs:
                    flags[q] = 1.0
        return flags

    def _get_observation(self) -> np.ndarray:
        sv = self._backend.statevector()
        obs = np.concatenate([sv.real, sv.imag]).astype(np.float32)
        if self.hardware_constraints is not None:
            obs = np.concatenate([obs, self._busy_flags()])
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        gate_count, two_q = self._gate_stats()
        return {
            "fidelity": self._last_fidelity,
            "gate_count": gate_count,
            "circuit_depth": len(self._layers),
            "two_qubit_gate_count": two_q,
            "hardware_violations": self._violations,
            "estimated_execution_time": self._execution_time,
            "steps": self._step_count,
        }

    def _render_state(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        probs = np.abs(self._backend.statevector()) ** 2
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(probs)), probs)
        ax.set_xlabel("Basis state")
        ax.set_ylabel("Probability")
        ax.set_title(f"Fidelity to target: {self._last_fidelity:.3f}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
