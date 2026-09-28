"""Split a wide/deep qiskit circuit into a sequence of smaller subcircuits.

The cutter picks gate indices at which to slice the circuit into consecutive
segments, remaps each segment onto only the qubits it touches, and records
which qubits carry state across each cut.

``reconstruct_results`` performs a naive merge of per-segment measurement
counts. It is not a quasi-probability reconstruction and does not recover the
full circuit's output distribution in general; it exists to give a rough
picture of where probability mass lands.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx
from qiskit import QuantumCircuit


@dataclass
class CircuitCut:
    """A cut in the circuit."""

    position: int  # gate index at which the cut occurs
    affected_qubits: Set[int]
    subcircuit_mapping: Dict[int, int]  # original qubit -> local index


@dataclass
class SubCircuit:
    """A segment of the original circuit remapped onto its own qubits."""

    circuit: QuantumCircuit
    input_qubits: Set[int]  # local qubits that receive state from the previous segment
    output_qubits: Set[int]  # local qubits that hand state to the next segment
    original_qubits: Dict[int, int]  # local index -> original qubit


def _qubit_indices(circuit: QuantumCircuit, inst) -> Tuple[int, ...]:
    return tuple(circuit.find_bit(q).index for q in inst.qubits)


class QuantumCircuitCutter:
    def __init__(self, max_subcircuit_width: int = 5, max_subcircuit_depth: int = 50):
        self.max_subcircuit_width = max_subcircuit_width
        self.max_subcircuit_depth = max_subcircuit_depth
        self.cut_points: List[CircuitCut] = []
        self.subcircuits: List[SubCircuit] = []

    def cut_circuit(self, circuit: QuantumCircuit) -> List[SubCircuit]:
        """Cut ``circuit`` into consecutive subcircuits."""
        self.cut_points = []
        self.subcircuits = []
        dep_graph = self._build_dependency_graph(circuit)
        self._find_cut_points(dep_graph, circuit)
        self.subcircuits = self._generate_subcircuits(circuit)
        return self.subcircuits

    # ------------------------------------------------------------- analysis

    def _build_dependency_graph(self, circuit: QuantumCircuit) -> nx.DiGraph:
        G = nx.DiGraph()
        qubit_sets = [set(_qubit_indices(circuit, inst)) for inst in circuit.data]
        for i, inst in enumerate(circuit.data):
            G.add_node(i, gate=inst.operation.name, qubits=tuple(sorted(qubit_sets[i])))
        for i in range(len(qubit_sets)):
            for j in range(i + 1, len(qubit_sets)):
                if qubit_sets[i] & qubit_sets[j]:
                    G.add_edge(i, j)
        return G

    def _find_cut_points(self, dep_graph: nx.DiGraph, circuit: QuantumCircuit) -> None:
        cut_metrics = {}
        n = len(circuit.data)
        for node in dep_graph.nodes():
            if node < 5 or node > n - 5:
                continue
            cut_metrics[node] = {
                "size_balance": self._evaluate_cut_point(node, dep_graph),
                "entanglement_cost": self._calculate_entanglement_cost(node, circuit),
            }
        self.cut_points = self._select_optimal_cuts(cut_metrics, circuit)

    def _evaluate_cut_point(self, node: int, dep_graph: nx.DiGraph) -> float:
        before = set(nx.ancestors(dep_graph, node)) | {node}
        after = set(dep_graph.nodes()) - before
        return abs(len(before) - len(after))

    def _calculate_entanglement_cost(self, node: int, circuit: QuantumCircuit) -> float:
        affected = set(_qubit_indices(circuit, circuit.data[node]))
        window = 3
        for i in range(max(0, node - window), min(len(circuit.data), node + window + 1)):
            if i != node:
                affected.update(_qubit_indices(circuit, circuit.data[i]))
        return len(affected)

    def _select_optimal_cuts(self, cut_metrics: Dict, circuit: QuantumCircuit) -> List[CircuitCut]:
        cuts: List[CircuitCut] = []
        circuit_length = len(circuit.data)
        min_subcircuit_size = self.max_subcircuit_depth // 2
        sorted_points = sorted(
            cut_metrics,
            key=lambda x: cut_metrics[x]["size_balance"] * 0.7 + cut_metrics[x]["entanglement_cost"] * 0.3,
        )
        current_pos = 0
        while current_pos < circuit_length:
            valid = [
                p
                for p in sorted_points
                if current_pos + min_subcircuit_size < p < current_pos + self.max_subcircuit_depth
            ]
            if not valid:
                break
            cut_point = valid[0]
            affected = set(_qubit_indices(circuit, circuit.data[cut_point]))
            cuts.append(
                CircuitCut(
                    position=cut_point,
                    affected_qubits=affected,
                    subcircuit_mapping={q: idx for idx, q in enumerate(sorted(affected))},
                )
            )
            current_pos = cut_point
        return cuts

    # ----------------------------------------------------------- generation

    def _generate_subcircuits(self, circuit: QuantumCircuit) -> List[SubCircuit]:
        subcircuits: List[SubCircuit] = []
        positions = [0] + [c.position for c in self.cut_points] + [len(circuit.data)]
        for i in range(len(positions) - 1):
            start, end = positions[i], positions[i + 1]
            segment = circuit.data[start:end]

            used: Set[int] = set()
            for inst in segment:
                used.update(_qubit_indices(circuit, inst))
            mapping = {old: new for new, old in enumerate(sorted(used))}

            sub = QuantumCircuit(len(used))
            for inst in segment:
                sub.append(inst.operation, [mapping[q] for q in _qubit_indices(circuit, inst)])

            input_qubits: Set[int] = set()
            output_qubits: Set[int] = set()
            if i > 0:
                input_qubits = {mapping[q] for q in self.cut_points[i - 1].affected_qubits if q in used}
            if i < len(self.cut_points):
                output_qubits = {mapping[q] for q in self.cut_points[i].affected_qubits if q in used}

            subcircuits.append(
                SubCircuit(
                    circuit=sub,
                    input_qubits=input_qubits,
                    output_qubits=output_qubits,
                    original_qubits={new: old for old, new in mapping.items()},
                )
            )
        return subcircuits

    # ------------------------------------------------------- reconstruction

    def reconstruct_results(self, subcircuit_results: List[Dict[str, float]], num_qubits: int) -> Dict[str, float]:
        """Naively merge per-segment counts onto the original qubit labels.

        Bitstrings follow qiskit's little-endian convention: the rightmost
        character is qubit 0.
        """
        full = defaultdict(float)
        for i, results in enumerate(subcircuit_results):
            sub = self.subcircuits[i]
            for bitstring, count in results.items():
                full[self._map_bitstring_to_original(bitstring, sub.original_qubits, num_qubits)] += count
        total = sum(full.values())
        if total > 0:
            for k in full:
                full[k] /= total
        return dict(full)

    @staticmethod
    def _map_bitstring_to_original(bitstring: str, qubit_mapping: Dict[int, int], num_qubits: int) -> str:
        bits = ["0"] * num_qubits
        local = bitstring.replace(" ", "")[::-1]  # local index k is the k-th char from the right
        for local_idx, bit in enumerate(local):
            if local_idx in qubit_mapping:
                bits[qubit_mapping[local_idx]] = bit
        return "".join(bits[::-1])

    # -------------------------------------------------------- visualisation

    def visualize_cuts(self, circuit: QuantumCircuit, path: str = "circuit_cuts.png") -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, inst in enumerate(circuit.data):
            qubits = list(_qubit_indices(circuit, inst))
            ax.scatter([i] * len(qubits), qubits, color="tab:blue", alpha=0.5)
            if len(qubits) > 1:
                ax.plot([i] * len(qubits), qubits, color="tab:blue", alpha=0.3)
        for cut in self.cut_points:
            ax.axvline(x=cut.position, color="tab:red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Gate index")
        ax.set_ylabel("Qubit index")
        ax.set_title("Circuit cuts")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
