"""Cut a random circuit into segments and execute each with qiskit-aer.

Requires ``pip install quantum-forge[aer,viz]``.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from quantum_forge import QuantumCircuitCutter


def create_large_circuit(num_qubits: int = 8, depth: int = 100, seed: int = 0) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            if rng.random() < 0.3:
                gate = rng.choice(["h", "x", "ry"])
                if gate == "ry":
                    circuit.ry(rng.random() * np.pi, qubit)
                else:
                    getattr(circuit, gate)(qubit)
        for _ in range(num_qubits // 2):
            if rng.random() < 0.3:
                q1, q2 = rng.choice(num_qubits, size=2, replace=False)
                getattr(circuit, rng.choice(["cx", "cz"]))(int(q1), int(q2))
    return circuit


def execute_subcircuits(subcircuits, shots: int = 1000):
    sim = AerSimulator()
    results = []
    for sub in subcircuits:
        meas = sub.circuit.copy()
        meas.measure_all()
        counts = sim.run(transpile(meas, sim), shots=shots).result().get_counts()
        results.append(counts)
    return results


def main() -> None:
    circuit = create_large_circuit()
    print(f"Original: {len(circuit.data)} gates, depth {circuit.depth()}, {circuit.num_qubits} qubits")

    cutter = QuantumCircuitCutter(max_subcircuit_width=5, max_subcircuit_depth=50)
    subcircuits = cutter.cut_circuit(circuit)
    print(f"Generated {len(subcircuits)} subcircuits at gate indices {[c.position for c in cutter.cut_points]}")
    for i, sc in enumerate(subcircuits, 1):
        print(
            f"  {i}: {sc.circuit.num_qubits} qubits, depth {sc.circuit.depth()}, "
            f"{len(sc.circuit.data)} gates, in={sorted(sc.input_qubits)} out={sorted(sc.output_qubits)}"
        )

    cutter.visualize_cuts(circuit)
    print("Saved circuit_cuts.png")

    results = execute_subcircuits(subcircuits)
    merged = cutter.reconstruct_results(results, circuit.num_qubits)
    print("\nTop 5 merged bitstrings (naive merge, see docstring):")
    for state, prob in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"  {state}: {prob:.4f}")


if __name__ == "__main__":
    main()
