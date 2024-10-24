from qiskit import QuantumCircuit, execute, Aer
from circuit_cutter import QuantumCircuitCutter
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

def create_large_circuit(num_qubits: int = 8, depth: int = 100) -> QuantumCircuit:
    """Create a large quantum circuit for demonstration"""
    circuit = QuantumCircuit(num_qubits)
    
    # Add a mix of single and multi-qubit gates
    for _ in range(depth):
        # Random single-qubit gates
        for qubit in range(num_qubits):
            if np.random.random() < 0.3:
                gate_choice = np.random.choice(['h', 'x', 'ry'])
                if gate_choice == 'ry':
                    circuit.ry(np.random.random() * np.pi, qubit)
                else:
                    getattr(circuit, gate_choice)(qubit)
        
        # Random two-qubit gates
        for _ in range(num_qubits // 2):
            if np.random.random() < 0.3:
                q1, q2 = np.random.choice(num_qubits, size=2, replace=False)
                gate_choice = np.random.choice(['cx', 'cz'])
                getattr(circuit, gate_choice)(q1, q2)
    
    return circuit

def execute_subcircuits(subcircuits: List[QuantumCircuit], 
                       shots: int = 1000) -> List[Dict]:
    """Execute subcircuits and return their results"""
    backend = Aer.get_backend('qasm_simulator')
    results = []
    
    for subcircuit in subcircuits:
        # Add measurements to output qubits
        meas_circuit = subcircuit.circuit.copy()
        meas_circuit.measure_all()
        
        # Execute circuit
        job = execute(meas_circuit, backend, shots=shots)
        counts = job.result().get_counts()
        results.append(counts)
    
    return results

def analyze_cutting_performance(original_circuit: QuantumCircuit,
                              subcircuits: List[QuantumCircuit]):
    """Analyze and visualize circuit cutting performance"""
    # Calculate size metrics
    original_size = {
        'width': original_circuit.num_qubits,
        'depth': original_circuit.depth(),
        'gates': len(original_circuit.data)
    }
    
    subcircuit_sizes = [{
        'width': sc.circuit.num_qubits,
        'depth': sc.circuit.depth(),
        'gates': len(sc.circuit.data)
    } for sc in subcircuits]
    
    # Plot size comparison
    metrics = ['width', 'depth', 'gates']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Original circuit
        ax.bar(0, original_size[metric], label='Original', alpha=0.7)
        
        # Subcircuits
        subcircuit_values = [size[metric] for size in subcircuit_sizes]
        ax.bar(range(1, len(subcircuit_values) + 1), 
               subcircuit_values, 
               label='Subcircuits',
               alpha=0.7)
        
        ax.set_title(f'Circuit {metric.capitalize()}')
        ax.set_xticks(range(len(subcircuit_values) + 1))
        ax.set_xticklabels(['Original'] + [f'Sub {i+1}' for i in range(len(subcircuit_values))])
        
    plt.tight_layout()
    plt.savefig('cutting_performance.png')
    plt.close()

def demonstrate_circuit_cutting():
    print("\nDemonstrating Quantum Circuit Cutting:")
    
    # Create a large quantum circuit
    print("1. Creating large quantum circuit...")
    circuit = create_large_circuit(num_qubits=8, depth=100)
    print(f"Original circuit size: {len(circuit.data)} gates, "
          f"depth: {circuit.depth()}, "
          f"qubits: {circuit.num_qubits}")
    
    # Initialize circuit cutter
    cutter = QuantumCircuitCutter(max_subcircuit_width=5, max_subcircuit_depth=50)
    
    # Cut the circuit
    print("\n2. Cutting circuit into smaller subcircuits...")
    subcircuits = cutter.cut_circuit(circuit)
    print(f"Generated {len(subcircuits)} subcircuits")
    
    # Print subcircuit information
    for i, sc in enumerate(subcircuits):
        print(f"\nSubcircuit {i+1}:")
        print(f"- Number of qubits: {sc.circuit.num_qubits}")
        print(f"- Circuit depth: {sc.circuit.depth()}")
        print(f"- Number of gates: {len(sc.circuit.data)}")
        print(f"- Input qubits: {sc.input_qubits}")
        print(f"- Output qubits: {sc.output_qubits}")
    
    # Visualize the cuts
    print("\n3. Generating circuit cutting visualization...")
    cutter.visualize_cuts(circuit)
    print("Circuit cuts visualization saved as 'circuit_cuts.png'")
    
    # Execute subcircuits
    print("\n4. Executing subcircuits...")
    subcircuit_results = execute_subcircuits(subcircuits)
    
    # Reconstruct results
    print("\n5. Reconstructing full circuit results...")
    full_results = cutter.reconstruct_results(subcircuit_results)
    print("Top 5 most probable states:")
    for state, prob in sorted(full_results.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:5]:
        print(f"State {state}: probability {prob:.4f}")
    
    # Analyze cutting performance
    print("\n6. Analyzing cutting performance...")
    analyze_cutting_performance(circuit, subcircuits)
    print("Cutting performance analysis saved as 'cutting_performance.png'")

if __name__ == "__main__":
    demonstrate_circuit_cutting()
    print("\nCircuit cutting demonstration completed.")
