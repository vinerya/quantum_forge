from typing import List, Dict, Tuple, Set
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class CircuitCut:
    """Represents a cut in the quantum circuit"""
    position: int  # Gate index where cut occurs
    affected_qubits: Set[int]  # Qubits affected by the cut
    subcircuit_mapping: Dict[int, int]  # Maps original qubit indices to subcircuit indices

@dataclass
class SubCircuit:
    """Represents a subcircuit after cutting"""
    circuit: QuantumCircuit
    input_qubits: Set[int]  # Qubits that receive input from previous subcircuit
    output_qubits: Set[int]  # Qubits that send output to next subcircuit
    original_qubits: Dict[int, int]  # Maps subcircuit qubit indices to original indices

class QuantumCircuitCutter:
    def __init__(self, max_subcircuit_width: int = 5, max_subcircuit_depth: int = 50):
        self.max_subcircuit_width = max_subcircuit_width
        self.max_subcircuit_depth = max_subcircuit_depth
        self.cut_points: List[CircuitCut] = []
        self.subcircuits: List[SubCircuit] = []

    def cut_circuit(self, circuit: QuantumCircuit) -> List[SubCircuit]:
        """Cut a large quantum circuit into smaller subcircuits"""
        # Reset previous cuts and subcircuits
        self.cut_points = []
        self.subcircuits = []

        # Build dependency graph
        dep_graph = self._build_dependency_graph(circuit)
        
        # Find optimal cut points
        self._find_cut_points(dep_graph, circuit)
        
        # Generate subcircuits
        self.subcircuits = self._generate_subcircuits(circuit)
        
        return self.subcircuits

    def _build_dependency_graph(self, circuit: QuantumCircuit) -> nx.DiGraph:
        """Build a directed graph representing gate dependencies"""
        G = nx.DiGraph()
        
        # Add nodes for each gate
        for i, instruction in enumerate(circuit.data):
            G.add_node(i, 
                      gate=instruction.operation.name,
                      qubits=tuple(q.index for q in instruction.qubits))
        
        # Add edges for dependencies
        for i in range(len(circuit.data)):
            for j in range(i + 1, len(circuit.data)):
                if self._gates_dependent(circuit.data[i], circuit.data[j]):
                    G.add_edge(i, j)
        
        return G

    def _gates_dependent(self, gate1, gate2) -> bool:
        """Check if two gates have dependencies"""
        qubits1 = set(q.index for q in gate1.qubits)
        qubits2 = set(q.index for q in gate2.qubits)
        return bool(qubits1.intersection(qubits2))

    def _find_cut_points(self, dep_graph: nx.DiGraph, circuit: QuantumCircuit):
        """Find optimal points to cut the circuit"""
        # Initialize metrics for each potential cut point
        cut_metrics = {}
        
        for node in dep_graph.nodes():
            # Skip first and last few gates
            if node < 5 or node > len(circuit.data) - 5:
                continue
                
            # Calculate metrics for this cut point
            subcircuit_sizes = self._evaluate_cut_point(node, dep_graph)
            entanglement_cost = self._calculate_entanglement_cost(node, circuit)
            
            # Combine metrics into a single score
            cut_metrics[node] = {
                'size_balance': subcircuit_sizes,
                'entanglement_cost': entanglement_cost
            }
        
        # Select optimal cut points
        selected_cuts = self._select_optimal_cuts(cut_metrics, circuit)
        
        # Store cut points
        self.cut_points = selected_cuts

    def _evaluate_cut_point(self, node: int, dep_graph: nx.DiGraph) -> float:
        """Evaluate the quality of a cut point based on resulting subcircuit sizes"""
        # Get subgraphs before and after cut
        before = set(nx.ancestors(dep_graph, node)).union({node})
        after = set(dep_graph.nodes()).difference(before)
        
        # Calculate size difference (aim for balanced sizes)
        size_diff = abs(len(before) - len(after))
        return size_diff

    def _calculate_entanglement_cost(self, node: int, circuit: QuantumCircuit) -> float:
        """Calculate the entanglement cost of cutting at a specific point"""
        instruction = circuit.data[node]
        affected_qubits = set(q.index for q in instruction.qubits)
        
        # Check neighboring gates for additional entanglement
        window = 3  # Look at gates within this window
        for i in range(max(0, node - window), min(len(circuit.data), node + window + 1)):
            if i != node:
                affected_qubits.update(q.index for q in circuit.data[i].qubits)
        
        return len(affected_qubits)

    def _select_optimal_cuts(self, cut_metrics: Dict, 
                           circuit: QuantumCircuit) -> List[CircuitCut]:
        """Select the optimal set of cut points"""
        cuts = []
        circuit_length = len(circuit.data)
        min_subcircuit_size = self.max_subcircuit_depth // 2
        
        # Sort points by combined metric
        sorted_points = sorted(
            cut_metrics.keys(),
            key=lambda x: (
                cut_metrics[x]['size_balance'] * 0.7 +
                cut_metrics[x]['entanglement_cost'] * 0.3
            )
        )
        
        current_pos = 0
        while current_pos < circuit_length:
            # Find next valid cut point
            valid_cuts = [
                p for p in sorted_points
                if p > current_pos + min_subcircuit_size and
                p < current_pos + self.max_subcircuit_depth
            ]
            
            if not valid_cuts:
                break
                
            cut_point = valid_cuts[0]
            affected_qubits = set(
                q.index for q in circuit.data[cut_point].qubits
            )
            
            # Create subcircuit mapping
            mapping = {
                q: idx for idx, q in enumerate(sorted(affected_qubits))
            }
            
            cuts.append(CircuitCut(
                position=cut_point,
                affected_qubits=affected_qubits,
                subcircuit_mapping=mapping
            ))
            
            current_pos = cut_point
            
        return cuts

    def _generate_subcircuits(self, circuit: QuantumCircuit) -> List[SubCircuit]:
        """Generate subcircuits based on cut points"""
        subcircuits = []
        cut_positions = [0] + [cut.position for cut in self.cut_points] + [len(circuit.data)]
        
        for i in range(len(cut_positions) - 1):
            start = cut_positions[i]
            end = cut_positions[i + 1]
            
            # Determine qubits used in this section
            used_qubits = set()
            for inst in circuit.data[start:end]:
                used_qubits.update(q.index for q in inst.qubits)
            
            # Create mapping for qubits
            qubit_mapping = {
                old: new for new, old in enumerate(sorted(used_qubits))
            }
            
            # Create new circuit
            subcircuit = QuantumCircuit(len(used_qubits))
            
            # Add gates with remapped qubits
            for inst in circuit.data[start:end]:
                gate = inst.operation
                qubits = [qubit_mapping[q.index] for q in inst.qubits]
                if hasattr(gate, 'params'):
                    subcircuit.append(gate, qubits, inst.clbits)
                else:
                    getattr(subcircuit, gate.name)(*qubits)
            
            # Determine input and output qubits
            input_qubits = set()
            output_qubits = set()
            
            if i > 0:
                input_qubits.update(
                    qubit_mapping[q] for q in self.cut_points[i-1].affected_qubits
                    if q in used_qubits
                )
            
            if i < len(self.cut_points):
                output_qubits.update(
                    qubit_mapping[q] for q in self.cut_points[i].affected_qubits
                    if q in used_qubits
                )
            
            subcircuits.append(SubCircuit(
                circuit=subcircuit,
                input_qubits=input_qubits,
                output_qubits=output_qubits,
                original_qubits={new: old for old, new in qubit_mapping.items()}
            ))
        
        return subcircuits

    def reconstruct_results(self, subcircuit_results: List[Dict]) -> Dict:
        """Reconstruct the results of the full circuit from subcircuit results"""
        # Initialize reconstruction
        full_results = defaultdict(float)
        
        # Process each subcircuit result
        for i, results in enumerate(subcircuit_results):
            subcircuit = self.subcircuits[i]
            
            # Map subcircuit results back to original qubit indices
            for bitstring, count in results.items():
                mapped_bitstring = self._map_bitstring_to_original(
                    bitstring, subcircuit.original_qubits
                )
                full_results[mapped_bitstring] += count
        
        # Normalize probabilities
        total = sum(full_results.values())
        if total > 0:
            for k in full_results:
                full_results[k] /= total
        
        return dict(full_results)

    def _map_bitstring_to_original(self, bitstring: str, 
                                 qubit_mapping: Dict[int, int]) -> str:
        """Map a bitstring from subcircuit qubits to original circuit qubits"""
        # Convert bitstring to list of bits
        bits = list(bitstring)
        
        # Create mapped bitstring
        mapped_bits = ['0'] * (max(qubit_mapping.values()) + 1)
        for subcircuit_idx, bit in enumerate(bits):
            if subcircuit_idx in qubit_mapping:
                mapped_bits[qubit_mapping[subcircuit_idx]] = bit
        
        return ''.join(mapped_bits)

    def visualize_cuts(self, circuit: QuantumCircuit) -> None:
        """Visualize the circuit cuts using matplotlib"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw circuit gates
        for i, inst in enumerate(circuit.data):
            qubits = [q.index for q in inst.qubits]
            y_positions = qubits
            
            # Draw gate
            ax.scatter([i] * len(qubits), y_positions, color='blue', alpha=0.5)
            
            # Draw connections for multi-qubit gates
            if len(qubits) > 1:
                ax.plot([i] * len(qubits), y_positions, color='blue', alpha=0.3)
        
        # Draw cut lines
        for cut in self.cut_points:
            ax.axvline(x=cut.position, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Gate Index')
        ax.set_ylabel('Qubit Index')
        ax.set_title('Quantum Circuit Cuts Visualization')
        
        plt.tight_layout()
        plt.savefig('circuit_cuts.png')
        plt.close()
