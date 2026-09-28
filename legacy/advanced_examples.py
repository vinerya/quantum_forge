from qiskit import QuantumCircuit
from error_mitigation import QuantumErrorMitigator, NoiseModel
from dynamic_compiler import DynamicCircuitCompiler, CompilationConstraints
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_error_mitigation():
    print("\nDemonstrating Error Mitigation:")
    
    # Create a noise model
    noise_model = NoiseModel(
        single_qubit_error_rates={i: 0.001 for i in range(5)},
        two_qubit_error_rates={(i, i+1): 0.01 for i in range(4)},
        measurement_error_rates={i: 0.02 for i in range(5)},
        coherence_times={i: (50000, 70000) for i in range(5)}  # T1, T2 in ns
    )
    
    # Create error mitigator
    mitigator = QuantumErrorMitigator(noise_model)
    
    # Create test circuit
    circuit = QuantumCircuit(5)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(np.pi/4, 2)
    circuit.cx(2, 3)
    circuit.cx(3, 4)
    circuit.measure_all()
    
    # Execute circuit with different mitigation strategies
    print("\n1. Zero-noise extrapolation:")
    zne_results = mitigator.zero_noise_extrapolation(
        circuit,
        {'00000': 0.8, '00001': 0.1, '00010': 0.1}
    )
    print(f"Fidelity improvement: {zne_results.fidelity_improvement:.3f}")
    
    print("\n2. Probabilistic error cancellation:")
    pec_results = mitigator.probabilistic_error_cancellation(circuit)
    print(f"Fidelity improvement: {pec_results.fidelity_improvement:.3f}")
    
    print("\n3. Measurement error mitigation:")
    mem_results = mitigator.measurement_error_mitigation(
        {'00000': 0.7, '00001': 0.15, '00010': 0.15}
    )
    print(f"Fidelity improvement: {mem_results.fidelity_improvement:.3f}")
    
    # Visualize results
    mitigator.visualize_mitigation_results(
        {'00000': 0.7, '00001': 0.15, '00010': 0.15},
        mem_results
    )
    print("\nError mitigation visualization saved as 'error_mitigation_results.png'")

def demonstrate_dynamic_compilation():
    print("\nDemonstrating Dynamic Circuit Compilation:")
    
    # Create compilation constraints
    constraints = CompilationConstraints(
        max_depth=20,
        max_two_qubit_gates=10,
        preferred_basis_gates=['x', 'h', 'cx', 'rz'],
        connectivity_map={
            0: [1, 2],
            1: [0, 3],
            2: [0, 4],
            3: [1],
            4: [2]
        },
        timing_constraints={
            'cx': 300,  # ns
            'x': 50,
            'h': 50,
            'rz': 0
        }
    )
    
    # Create compiler
    compiler = DynamicCircuitCompiler(optimization_level=3)
    
    # Create test circuit
    circuit = QuantumCircuit(5)
    for i in range(5):
        circuit.h(i)
    for i in range(4):
        circuit.cx(i, i+1)
    for i in range(5):
        circuit.rz(np.pi/4, i)
    for i in range(3):
        circuit.ccx(i, i+1, i+2)
    
    print("\n1. Compiling circuit...")
    compiled_circuit = compiler.compile_circuit(circuit, constraints)
    
    print("\n2. Analyzing compilation results:")
    analysis = compiler.analyze_compilation(circuit, compiled_circuit)
    print(f"Depth reduction: {analysis['depth_reduction']:.2%}")
    print(f"Gate count reduction: {analysis['gate_count_reduction']:.2%}")
    print(f"Execution time improvement: {analysis['execution_time_improvement']:.2%}")
    
    print("\n3. Visualizing compilation results...")
    compiler.visualize_compilation_results(circuit, compiled_circuit)
    print("Compilation results visualization saved as 'compilation_results.png'")

if __name__ == "__main__":
    demonstrate_error_mitigation()
    demonstrate_dynamic_compilation()
    print("\nAdvanced features demonstration completed.")
