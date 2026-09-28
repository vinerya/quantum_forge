from quantum_forge import QuantumForgeEnv, HardwareConstraints
from circuit_optimizer import QuantumCircuitOptimizer, OptimizationTarget
from qiskit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_advanced_optimization():
    print("\nDemonstrating advanced quantum circuit optimization:")
    
    # Initialize hardware constraints
    hardware_constraints = HardwareConstraints(
        connectivity_map={
            0: [1, 2],
            1: [0, 3],
            2: [0, 4],
            3: [1, 5],
            4: [2, 5],
            5: [3, 4]
        },
        gate_times={
            'x': 50,
            'h': 50,
            'cx': 300,
            'rz': 0,
            'ry': 50
        },
        error_rates={(i, j): 0.01 for i in range(6) for j in range(6) if abs(i-j) == 1},
        max_parallel_gates=4
    )
    
    # Initialize circuit optimizer
    optimizer = QuantumCircuitOptimizer(num_qubits=6, hardware_constraints=hardware_constraints)
    
    # Example 1: Optimize QFT
    print("\n1. Optimizing Quantum Fourier Transform:")
    qft_circuit = optimizer.decompose_algorithm('qft', {'n_qubits': 6})
    qft_resources = optimizer.estimate_resources(qft_circuit)
    print("QFT Resources:")
    for metric, value in qft_resources.items():
        print(f"{metric}: {value}")
    
    # Example 2: Optimize Grover's Algorithm
    print("\n2. Optimizing Grover's Algorithm:")
    # Create simple oracle for demonstration
    oracle = QuantumCircuit(6)
    oracle.x(0)
    oracle.h(5)
    oracle.mcx([0,1,2,3,4], 5)
    oracle.h(5)
    oracle.x(0)
    
    grover_circuit = optimizer.decompose_algorithm(
        'grover',
        {
            'oracle': oracle,
            'iterations': 2
        }
    )
    grover_resources = optimizer.estimate_resources(grover_circuit)
    print("Grover Resources:")
    for metric, value in grover_resources.items():
        print(f"{metric}: {value}")
    
    # Example 3: Optimize VQE
    print("\n3. Optimizing VQE Circuit:")
    vqe_circuit = optimizer.decompose_algorithm(
        'vqe',
        {
            'depth': 3,
            'hamiltonian': None  # Simple demonstration without specific Hamiltonian
        }
    )
    vqe_resources = optimizer.estimate_resources(vqe_circuit)
    print("VQE Resources:")
    for metric, value in vqe_resources.items():
        print(f"{metric}: {value}")
    
    # Compare optimization levels
    print("\n4. Comparing Optimization Levels:")
    # Create a test circuit
    test_circuit = QuantumCircuit(6)
    for i in range(6):
        test_circuit.h(i)
    for i in range(5):
        test_circuit.cx(i, i+1)
    for i in range(6):
        test_circuit.rz(np.pi/4, i)
    
    results = {}
    for level in range(3):
        optimized = optimizer.optimize_circuit(
            test_circuit,
            OptimizationTarget(optimization_level=level)
        )
        results[f"Level {level}"] = optimizer.estimate_resources(optimized)
    
    # Plot comparison
    metrics = ['depth', 'gate_count', 'two_qubit_gate_count']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[f"Level {level}"][metric] for level in range(3)]
        axes[i].bar(range(3), values)
        axes[i].set_title(metric)
        axes[i].set_xlabel('Optimization Level')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png')
    plt.close()
    
    print("\nOptimization comparison plot saved as 'optimization_comparison.png'")
    
    # Verify circuit equivalence
    print("\n5. Verifying Circuit Equivalence:")
    level0 = optimizer.optimize_circuit(
        test_circuit,
        OptimizationTarget(optimization_level=0)
    )
    level2 = optimizer.optimize_circuit(
        test_circuit,
        OptimizationTarget(optimization_level=2)
    )
    
    is_equivalent = optimizer.verify_equivalence(level0, level2)
    print(f"Circuits are functionally equivalent: {is_equivalent}")

if __name__ == "__main__":
    demonstrate_advanced_optimization()
    print("\nAdvanced optimization demonstration completed.")
