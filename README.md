# QuantumForge: Advanced Quantum Circuit Design and Optimization Framework

QuantumForge is a comprehensive quantum circuit design and optimization framework that combines reinforcement learning, circuit cutting, error mitigation, and dynamic compilation to create optimal quantum circuits.

## Features

### Core Circuit Design
- Support for both Qiskit and Cirq backends
- Gymnasium-compatible environment
- Advanced action space including multi-qubit gates
- Realistic noise simulation
- Reinforcement learning integration with Stable Baselines3
- Hyperparameter optimization using Optuna
- Circuit optimization and visualization

### Circuit Optimization
- Template-based optimization
- Quantum Shannon Decomposition
- Gate commutation analysis
- Multi-objective optimization
- Hardware-aware optimization
- Resource estimation
- Circuit equivalence verification

### Circuit Cutting
- Intelligent cut-point selection
- Dependency graph analysis
- Balanced subcircuit generation
- Entanglement cost minimization
- Automated qubit remapping
- Result reconstruction
- Cutting visualization

### Error Mitigation
- Zero-noise extrapolation
- Probabilistic error cancellation
- Measurement error mitigation
- Noise characterization
- Confidence interval calculation
- Error analysis visualization
- Fidelity improvement tracking

### Dynamic Compilation
- Runtime optimization with caching
- Pulse-level optimization
- Hardware-specific compilation
- Automated gate decomposition
- Timing constraint optimization
- Qubit mapping optimization
- Compilation analysis tools

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Example
See `example.py` for basic usage demonstrating:
- Training a PPO agent on the QuantumForge environment
- Evaluating the trained agent
- Visualizing rewards and quantum states

```bash
python example.py
```

### Advanced Examples

#### Hardware-Aware Optimization
See `hardware_aware_example.py` for:
- Hardware constraint integration
- Noise-aware optimization
- Connectivity optimization

```bash
python hardware_aware_example.py
```

#### Circuit Cutting
See `circuit_cutting_example.py` for:
- Large circuit decomposition
- Subcircuit execution
- Result reconstruction

```bash
python circuit_cutting_example.py
```

#### Error Mitigation
See `advanced_examples.py` for:
- Error mitigation techniques
- Dynamic compilation
- Advanced optimization strategies

```bash
python advanced_examples.py
```

## Components

### Environment (QuantumForgeEnv)
- Observations: Current quantum state as complex vector
- Actions: (operation, qubit1, qubit2, parameter)
  - Operations: X, Z, H, RY, CNOT, CZ, RXX, RZZ, etc.
  - Qubits: Indices for operation application
  - Parameters: Used for parameterized gates

### Circuit Optimizer
- Gate sequence optimization
- Template matching
- Quantum Shannon Decomposition
- Resource estimation
- Circuit equivalence checking

### Circuit Cutter
- Dependency analysis
- Cut-point selection
- Subcircuit generation
- Result reconstruction
- Performance analysis

### Error Mitigator
- Multiple mitigation strategies
- Noise characterization
- Error analysis
- Result improvement tracking

### Dynamic Compiler
- Runtime optimization
- Pulse-level control
- Hardware adaptation
- Performance analysis
- Visualization tools

## Backends

QuantumForge supports two quantum computing backends:

1. Qiskit: IBM's quantum computing framework
   - Full noise simulation
   - Hardware-specific optimization
   - Pulse-level control

2. Cirq: Google's quantum computing framework
   - Noise modeling
   - Device specification
   - Custom gate sets

## Visualization

The framework generates various visualizations:
- Reward plots
- Quantum state visualizations
- Circuit cut diagrams
- Error mitigation results
- Compilation analysis
- Resource usage comparisons

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use QuantumForge in your research, please cite:

```bibtex
@software{quantumforge2024,
  title = {QuantumForge: Advanced Quantum Circuit Design and Optimization Framework},
  year = {2024},
  author = {Moudather Chelbi},
  url = {https://github.com/vinerya/QuantumForge}
}
